# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import atexit
import importlib
import inspect
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, AsyncGenerator

import torch
import yaml
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.common.utils.video_utils import parse_size
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.cf_dispatch import CFDispatchError, CFSessionDispatcher
from dynamo.vllm.omni.cf_pipeline import cf_put_key
from dynamo.vllm.omni.cf_session import (
    CFSessionRequestError,
    has_cf_session,
    parse_cf_session,
)
from dynamo.vllm.omni.connectors import register_dynamoomni_nixl_connector
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import (
    DEFAULT_VIDEO_SIZE,
    _build_sampling_params,
    ensure_awaited,
    is_empty_payload,
    parse_omni_request,
    resolve_stage_configs_compat,
    unwrap_connector_payload,
)

logger = logging.getLogger(__name__)


@dataclass
class _Proxy:
    """Satisfies stage_list[i].engine_outputs for processor functions.

    Processor functions (e.g. ar2diffusion) access stage_list[i].engine_outputs
    as a list of OmniRequestOutput objects.
    """

    engine_outputs: Any = None


class OmniStageWorker:
    """Single-stage worker: fetches inputs → runs processor → runs engine → writes output.

    For stage 0: gets engine_inputs directly from request.
    For stage N > 0: fetches previous stage outputs from connectors via stage_connector_refs,
    runs the pre-processor (e.g. thinker2talker) to produce this stage's engine inputs,
    then runs the engine.

    Non-final stages write output to a connector and yield stage_connector_refs for the router.
    Final stages write to SHM and yield shm_meta for the router to format.
    """

    def __init__(
        self,
        engine: StageEngine,
        stage_config: Any,
        connectors: dict,
        stage_id: int,
        output_modalities: list | None = None,
        default_video_fps: int = 16,
    ) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.connectors = connectors  # {(from_stage, to_stage): vllm_omni connector}
        self._output_modalities = output_modalities or []
        self._default_video_fps = default_video_fps
        self.stage_config = stage_config

        func_path = getattr(stage_config, "custom_process_input_func", None)
        self._processor = _load_processor(func_path)
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._requires_mm: bool = getattr(
            stage_config, "requires_multimodal_data", False
        )

    async def generate(self, request: dict, context) -> AsyncGenerator[dict, None]:
        req = StageRequest.model_validate(request)
        request_id = req.request_id or context.id()

        # Causal-Forcing pipelined streaming: one chunk per rollout block instead
        # of one per request. Branches early and stays entirely out of the batch
        # path below, which is deliberate -- the one-shot chain is the tested
        # thing and a session request is recognizable up front, so there is no
        # reason for the two to share a code path they would both have to guard.
        if cf_streaming_stage(req, request) is not None:
            async for chunk in self._generate_cf_stream(req, request, request_id):
                yield chunk
            return

        original_prompt = req.original_prompt
        # JSON sends dict keys as strings; normalize to int for stage_connector_refs.
        stage_connector_refs = _int_keyed(req.stage_connector_refs)

        # --- Resolve engine inputs ---
        sampling_params_list_override: dict | None = None
        if stage_connector_refs:
            # Stage N > 0: fetch previous stage outputs from connectors, run pre-processor.
            sampling_params_list_override = req.sampling_params_list
            try:
                stage_list = await ensure_awaited(
                    self._fetch_stage_inputs(stage_connector_refs, request_id)
                )
            except RuntimeError as e:
                yield {"error": str(e), "finished": True}
                return

            if len(stage_list) != len(
                self._engine_input_source or stage_connector_refs
            ):
                logger.warning(
                    "Stage %d: expected %d stage inputs, got %d",
                    self.stage_id,
                    len(self._engine_input_source or stage_connector_refs),
                    len(stage_list),
                )

            if self._processor is not None:
                prompt = self._process_stage_inputs(stage_list, original_prompt)
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
            else:
                # No processor: check if the upstream output has the
                # structure needed to build an OmniEngineCoreRequest
                # (e.g. code2wav receiving token_ids from talker).
                # Otherwise fall back to passing the raw data directly.
                upstream = stage_list[-1].engine_outputs[0]
                if hasattr(upstream, "outputs") and upstream.outputs:
                    try:
                        prompt = self._build_engine_core_request_from_upstream(
                            stage_list, request_id, sampling_params_list_override
                        )
                    except RuntimeError as e:
                        yield {"error": str(e), "finished": True}
                        return
                else:
                    prompt = upstream
        elif req.request_id is not None:
            # Stage 0 via router: raw request forwarded with request_id — parse it.
            parsed = await parse_omni_request(
                request,
                self._output_modalities,
                self._default_video_fps,
                tokenizer_getter=self.engine.get_tokenizer,
            )
            prompt = parsed["engine_inputs"]
            original_prompt = parsed["original_prompt"]
            sampling_params_list_override = parsed["sampling_params_list"]
        else:
            # Direct frontend → stage (single-stage, no router).
            prompt = request

        logger.debug(
            "Stage %d: engine.generate for %s — prompt type=%s",
            self.stage_id,
            request_id,
            type(prompt).__name__,
        )

        sp = _build_sampling_params(self.stage_config, sampling_params_list_override)
        last_result = None

        # --- Write output ---
        # Check for a downstream connector first, regardless of final_output.
        # In vllm-omni's native mode, multiple stages can set final_output=True
        # (meaning "produces user-visible output"). In Dynamo's disaggregated
        # mode the actual pipeline topology — connector edges from the YAML —
        # determines whether output should go to a connector or to SHM.
        from_s, to_s = _connector_key(self.stage_id, self.stage_id + 1)
        connector = self.connectors.get((from_s, to_s))

        # Per-chunk block-stream lane (work-item b, §6.2/§8). The DiT engine can
        # emit N non-terminal latent blocks (finished=False) followed by one
        # terminal sentinel. When such a block arrives — and a downstream
        # connector exists — deliver it immediately under the per-chunk key
        # f"{request_id}_c{n}" and RPC-yield a tiny control signal (no tensor).
        # The default (aggregated) engine emits a single finished=True output, so
        # n_streamed stays 0 and the untouched post-loop path runs as before.
        n_streamed = 0
        try:
            async for chunk in self.engine.generate(
                prompt, request_id=request_id, sampling_params_list=sp
            ):
                # ``finished`` defaults True, matching DiffusionOutput: an output
                # that does not carry the field is a whole result, not a stream
                # block. Defaulting False instead would route every engine output
                # without the attribute -- LLM RequestOutputs, plain dicts -- down
                # the per-chunk lane, so the terminal aggregated result would never
                # be emitted and the request would hang waiting for a chunk that
                # never comes.
                if not bool(_chunk_finished(chunk)):
                    if connector is not None:
                        # Inter-stage lane (work-item b): DiT streams latent
                        # blocks to the next stage under per-chunk connector keys.
                        if not await self._put_stream_chunk(
                            connector, from_s, to_s, request_id, n_streamed, chunk
                        ):
                            yield {"error": "per-chunk connector.put() failed", "finished": True}
                            return
                        logger.info(
                            "[cmaf-timing] stage %d (DiT) put latent block %d -> connector for %s",
                            self.stage_id, n_streamed, request_id,
                        )
                        yield {"chunk_index": n_streamed, "is_last": False, "finished": False}
                    else:
                        # Final-stage lane (work-item c): VAE streams pixel chunks
                        # to the router. Pixel tensors can't ride the JSON RPC, so
                        # each chunk goes to SHM under its per-chunk name and the
                        # RPC yields only the ref for the router to deserialize.
                        try:
                            shm_meta = self._write_stream_chunk_shm(
                                request_id, n_streamed, chunk
                            )
                        except Exception as e:
                            yield {"error": f"per-chunk SHM write failed: {e}", "finished": True}
                            return
                        logger.info(
                            "[cmaf-timing] stage %d (VAE) decoded pixel chunk %d -> SHM for %s",
                            self.stage_id, n_streamed, request_id,
                        )
                        yield {
                            "chunk_index": n_streamed,
                            "is_last": False,
                            "finished": False,
                            "shm_meta": shm_meta,
                        }
                    n_streamed += 1
                    continue
                last_result = chunk
        except Exception as e:
            logger.error(
                "Stage %d engine error for %s: %s",
                self.stage_id,
                request_id,
                e,
                exc_info=True,
            )
            yield {"error": str(e), "finished": True}
            return

        if n_streamed > 0:
            if connector is not None:
                # Inter-stage streamed (b): every block is already on the
                # connector under its per-chunk key. Emit only a terminal control
                # signal whose ref carries num_chunks so the consumer knows how
                # many per-chunk keys to fetch (§8). No tensor travels over RPC.
                out: dict = {
                    "original_prompt": original_prompt,
                    "stage_connector_refs": {
                        **{str(k): v for k, v in stage_connector_refs.items()},
                        str(self.stage_id): {"num_chunks": n_streamed, "chunked": True},
                    },
                    "chunk_index": n_streamed - 1,
                    "is_last": True,
                    "finished": True,
                }
                if sampling_params_list_override is not None:
                    out["sampling_params_list"] = sampling_params_list_override
                yield out
            else:
                # Final-stage streamed (c): pixel chunks were already delivered to
                # the router via per-chunk SHM. The terminal is a bare sentinel.
                yield {"chunk_index": n_streamed - 1, "is_last": True, "finished": True}
            return

        _ensure_cumulative_token_ids(last_result)

        if connector is not None:
            try:
                put_result = await ensure_awaited(
                    connector.put(  # type: ignore[arg-type]
                        from_s,
                        to_s,
                        request_id,
                        _prepare_connector_payload(
                            last_result,
                            from_stage=self.stage_id,
                            to_stage=self.stage_id + 1,
                        ),
                    )
                )
                ok, _, metadata = put_result
            except Exception as e:
                logger.error(
                    "Stage %d: connector.put() raised %s: %s",
                    self.stage_id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                yield {"error": f"connector.put() raised: {e}", "finished": True}
                return
            if not ok:
                yield {"error": "connector.put() failed", "finished": True}
                return
            out: dict = {
                "original_prompt": original_prompt,
                "stage_connector_refs": {
                    **{str(k): v for k, v in stage_connector_refs.items()},
                    str(self.stage_id): metadata,
                },
                "finished": True,
            }
            if sampling_params_list_override is not None:
                out["sampling_params_list"] = sampling_params_list_override
            yield out
            return

        # Final stage -> router: check for a YAML-configured connector for the
        # (stage_id -> "router") edge before falling back to SHM.  A connector
        # here enables multi-node deployments where the router and final stage
        # worker reside on different machines (SHM requires same host).
        router_connector = self.connectors.get(_connector_key(self.stage_id, "router"))
        if router_connector is not None:
            try:
                rput_result = await ensure_awaited(
                    router_connector.put(  # type: ignore[arg-type]
                        from_s,
                        "router",
                        request_id,
                        _prepare_connector_payload(
                            last_result,
                            from_stage=self.stage_id,
                            to_stage="router",
                        ),
                    )
                )
                ok, _, metadata = rput_result
            except Exception as e:
                logger.error(
                    "Stage %d: router connector.put() raised %s: %s",
                    self.stage_id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                yield {"error": f"router connector.put() raised: {e}", "finished": True}
                return
            if not ok:
                yield {"error": "router connector.put() failed", "finished": True}
                return
            yield {
                "stage_connector_refs": {str(self.stage_id): metadata},
                "finished": True,
            }
            return

        # SHM fallback -- only works when router and final stage are on the same node.
        shm_meta = shm_write_bytes(serialize_obj(last_result), name=request_id)
        yield {"shm_meta": shm_meta, "finished": True}

    async def _put_stream_chunk(
        self,
        connector: Any,
        from_s: str,
        to_s: str,
        request_id: str,
        chunk_index: int,
        chunk: Any,
    ) -> bool:
        """Write one streamed block under the per-chunk key ``{request_id}_c{n}``.

        The per-chunk key is the actual fix (§8): the SHM connector overwrites
        on a repeated key, so a single ``request_id`` key would clobber every
        block but the last. Returns the connector's ok flag; errors are logged
        and reported as a failed put (POC: no retry).
        """
        _ensure_cumulative_token_ids(chunk)
        try:
            ok, _, _ = await ensure_awaited(
                connector.put(  # type: ignore[arg-type]
                    from_s,
                    to_s,
                    _chunk_key(request_id, chunk_index),
                    _prepare_connector_payload(
                        chunk,
                        from_stage=self.stage_id,
                        to_stage=self.stage_id + 1,
                    ),
                )
            )
        except Exception as e:
            logger.error(
                "Stage %d: per-chunk connector.put() raised for %s c%d: %s: %s",
                self.stage_id,
                request_id,
                chunk_index,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return False
        return bool(ok)

    def _write_stream_chunk_shm(
        self, request_id: str, chunk_index: int, chunk: Any
    ) -> dict:
        """Write one streamed pixel chunk to SHM under its per-chunk name (§c).

        The final (VAE) stage has no downstream connector, so pixel chunks reach
        the router the same way the aggregated final output does — via SHM — but
        under the per-chunk name ``{request_id}_c{n}`` so streamed frames don't
        clobber each other. Returns the shm_meta ref for the router to read with
        ``shm_deserialize``.
        """
        return shm_write_bytes(
            serialize_obj(chunk), name=_chunk_key(request_id, chunk_index)
        )
    # -- Causal-Forcing pipelined streaming ---------------------------------

    async def _generate_cf_stream(
        self, req: StageRequest, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Serve one Causal-Forcing session request, one chunk per block.

        Which half of the stream this is depends on where the session id came
        from, and that is not a preference -- it is the only signal available.
        Stage 0 reads it off the request's ``nvext`` annotations, because that is
        the one channel a client can extend end to end. Stage N>0 reads the
        ``cf_session`` field the previous stage put on the hop, because ``nvext``
        never reaches it.
        """
        stage = cf_streaming_stage(req, request)
        if stage is None:  # pragma: no cover -- generate() already checked
            raise RuntimeError("_generate_cf_stream called for a non-session request")

        if stage == "dit":
            gen = self._cf_stream_rollout(request, request_id)
        else:
            gen = self._cf_stream_decode(req, request_id)

        try:
            async for chunk in gen:
                yield chunk
        except CFDispatchError as e:
            # A routing or delivery failure: the session is not where we thought,
            # or the worker never ran the call. Not the client's fault, and not
            # something a retry of this request would fix.
            logger.error("Stage %d: CF session dispatch failed: %s", self.stage_id, e)
            yield {"error": str(e), "finished": True, "cf_last": True}
        except CFSessionRequestError as e:
            logger.warning(
                "Stage %d: malformed CF session request: %s", self.stage_id, e
            )
            yield {"error": str(e), "finished": True, "cf_last": True}
        except Exception as e:
            logger.error(
                "Stage %d: CF stream failed for %s: %s",
                self.stage_id,
                request_id,
                e,
                exc_info=True,
            )
            yield {"error": str(e), "finished": True, "cf_last": True}

    async def _cf_stream_rollout(
        self, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Stage 0: roll the session out a block at a time, putting each downstream.

        ``max_blocks=1`` is what makes this a pipeline rather than a batch with
        extra steps. Draining everything queued would produce the same tensors and
        hand them over all at once, leaving the VAE stage idle for the whole
        rollout and then the DiT stage idle for the whole decode. One block per
        RPC costs a host round-trip each (~585 KB pickled at 480x832, against
        ~0.96 s of compute) and buys the overlap.

        Serves the aggregated (single-stage) deploy as well, where this worker is
        also the last one. What ``drain`` returns is decided by the pipeline's
        role, not by this method: role ``dit`` has no VAE so its blocks are
        latents bound for stage 1, while role ``full`` decodes inline and yields
        pixels that are already final. Both are handed to the next hop by the
        same ``_cf_put_block`` call -- only the destination differs, and
        ``_is_final_stage`` picks it. There is still no DiT/VAE overlap in the
        aggregated case (the inline decode is serial inside the rollout), but the
        encoder still overlaps, which is exactly the part this lane contributes.
        """
        cf_req = parse_cf_session(request.get("nvext"))
        if cf_req is None:  # pragma: no cover -- the caller established this
            raise RuntimeError("no cf_session annotation on a session request")

        prompt = (request.get("prompt") or "").strip()
        if not prompt:
            raise CFSessionRequestError("a session scene needs a non-empty prompt.")

        dispatcher = CFSessionDispatcher(self.engine, self.stage_id)
        opened = await dispatcher.ensure_open(
            cf_req.session_id,
            continues=cf_req.scene.transition == "continue",
            **_cf_open_kwargs(request),
        )
        logger.info(
            "[CF_STREAM] stage %d session %s: %s, scene=%s",
            self.stage_id,
            cf_req.session_id,
            "opened" if opened else "already open",
            cf_req.scene.transition,
        )
        await dispatcher.push(cf_req.session_id, prompt, **cf_req.scene.push_kwargs())

        block = 0
        try:
            while True:
                blocks = await dispatcher.drain(cf_req.session_id, max_blocks=1)
                if not blocks:
                    # The queue is dry, which ends *this request's* rollout, not
                    # the session: the KV window stays live so the next scene can
                    # continue the shot.
                    break
                latents = blocks[0]
                metadata = await self._cf_put_block(
                    latents, request_id, block, to_router=self._is_final_stage()
                )
                yield {
                    "original_prompt": {"prompt": prompt},
                    "stage_connector_refs": {str(self.stage_id): metadata},
                    "cf_session": cf_req.session_id,
                    "cf_block": block,
                    "finished": True,
                }
                block += 1
        finally:
            if cf_req.close:
                # Honoured even on failure: a session that outlives its request
                # holds a KV window, and because the VAE's temporal cache is
                # per-module rather than per-session it blocks the whole card
                # until the idle timeout reclaims it.
                await dispatcher.close(cf_req.session_id, missing_ok=True)

        logger.info(
            "[CF_STREAM] stage %d session %s emitted %d block(s)",
            self.stage_id,
            cf_req.session_id,
            block,
        )
        # A terminal marker of its own rather than a flag on the last block: the
        # block count is not known until the rollout runs dry, so there is no
        # earlier chunk that could have carried it.
        yield {
            "cf_session": cf_req.session_id,
            "cf_block": block,
            "cf_last": True,
            "finished": True,
        }

    async def _cf_stream_decode(
        self, req: StageRequest, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Stage 1: decode one block through the session's live temporal cache.

        One request per block, so this is a single decode and a single put -- the
        pipelining is the router's, not this method's. What makes it a *session*
        call rather than an ordinary stage run is ``session_decode_step``: it
        keeps the decoder's ``feat_cache`` alive between calls, so this block's
        first frame attends to the context the previous block left behind. Decode
        each block through the batch path instead and every block boundary would
        be a visible seam.
        """
        session_id = req.cf_session
        block = req.cf_block or 0
        dispatcher = CFSessionDispatcher(self.engine, self.stage_id)

        if req.cf_last:
            # The rollout's terminal marker. Nothing to decode; release the
            # decode cursor so the next stream starts from a clean feat_cache.
            await dispatcher.close(session_id, missing_ok=True)
            yield {
                "cf_session": session_id,
                "cf_block": block,
                "cf_last": True,
                "finished": True,
            }
            return

        # A decode cursor is cheap to open and has no history to lose, so unlike a
        # rollout it may be opened on any block -- which is what lets the VAE
        # stage recover from having been restarted mid-stream.
        #
        # Timed in four parts because an aggregate per-block number cannot tell a
        # busy card from a busy queue, and this stage was for a while suspected of
        # the former when it was doing the latter. ensure_open is broken out on its
        # own precisely because it is the non-obvious cost: it is a BROADCAST to
        # every replica (session_info, via open_sessions) issued once per block, so
        # it cannot return until the slowest replica answers -- and a replica
        # mid-decode answers only when its decode is done.
        t0 = time.monotonic()
        await dispatcher.ensure_open(session_id)
        t_ensure = time.monotonic()

        latents = await self._cf_get_block(req, request_id, block)
        t_get = time.monotonic()
        frames = await dispatcher.decode_step(session_id, latents)
        t_decode = time.monotonic()

        metadata = await self._cf_put_block(frames, request_id, block, to_router=True)
        logger.info(
            "[CF_BLOCK_TIME] session=%s block=%d ensure=%.0fms get=%.0fms "
            "decode_rpc=%.0fms put=%.0fms total=%.0fms",
            session_id,
            block,
            (t_ensure - t0) * 1e3,
            (t_get - t_ensure) * 1e3,
            (t_decode - t_get) * 1e3,
            (time.monotonic() - t_decode) * 1e3,
            (time.monotonic() - t0) * 1e3,
        )
        yield {
            "stage_connector_refs": {str(self.stage_id): metadata},
            "cf_session": session_id,
            "cf_block": block,
            "finished": True,
        }

    def _is_final_stage(self) -> bool:
        """Whether this worker's output goes to the router rather than to a stage.

        Decided by the connector edges the YAML declares, which is the same signal
        the batch path uses at the top of ``generate`` and deliberately not
        ``final_output``: in vllm-omni's native mode several stages can set
        ``final_output=True`` (it means "produces user-visible output"), so it does
        not identify the last one. An edge to ``stage_id + 1`` exists only when
        there is a stage there to receive it, so its absence is what "last" means
        here -- and it makes the aggregated deploy fall out of the topology rather
        than out of a stage-count special case.
        """
        return (
            self.connectors.get(_connector_key(self.stage_id, self.stage_id + 1))
            is None
        )

    async def _cf_put_block(
        self, payload: Any, request_id: str, block: int, *, to_router: bool = False
    ) -> Any:
        """Hand one block to the next hop, keyed per block.

        The per-block key is not cosmetic: ``SharedMemoryConnector`` names both
        the segment and its lockfile after the key, so every block of a stream
        under one request id would overwrite its predecessor while contending on
        a single lock.
        """
        to_stage: int | str = "router" if to_router else self.stage_id + 1
        connector = self.connectors.get(_connector_key(self.stage_id, to_stage))
        if connector is None:
            raise CFDispatchError(
                f"Stage {self.stage_id}: no connector for edge "
                f"({self.stage_id}→{to_stage}); a pipelined stream cannot fall back "
                "to SHM-by-request-id, because every block would reuse one key."
            )
        ok, _, metadata = await ensure_awaited(
            connector.put(
                str(self.stage_id),
                str(to_stage),
                cf_put_key(request_id, block),
                payload,
            )
        )
        if not ok:
            raise CFDispatchError(
                f"Stage {self.stage_id}: connector.put() failed for block {block}"
            )
        return metadata

    async def _cf_get_block(
        self, req: StageRequest, request_id: str, block: int
    ) -> Any:
        """Fetch one block's latents from the upstream stage's connector."""
        refs = _int_keyed(req.stage_connector_refs)
        from_stage = (
            self._engine_input_source[0]
            if self._engine_input_source
            else min(refs, default=None)
        )
        if from_stage is None or from_stage not in refs:
            raise CFDispatchError(
                f"Stage {self.stage_id}: block {block} carries no connector ref "
                f"(refs={sorted(refs)})"
            )
        connector = self.connectors.get(_connector_key(from_stage, self.stage_id))
        if connector is None:
            raise CFDispatchError(
                f"Stage {self.stage_id}: no connector for edge "
                f"({from_stage}→{self.stage_id})"
            )
        payload = unwrap_connector_payload(
            await ensure_awaited(
                connector.get(
                    str(from_stage),
                    str(self.stage_id),
                    cf_put_key(request_id, block),
                    metadata=refs[from_stage],
                )
            )
        )
        if is_empty_payload(payload):
            raise CFDispatchError(
                f"Stage {self.stage_id}: empty payload for block {block} "
                f"from stage {from_stage}"
            )
        return payload

    def _build_engine_core_request_from_upstream(
        self,
        stage_list: list[_Proxy],
        request_id: str,
        sampling_params_list_override: dict | None,
    ):
        """Build an OmniEngineCoreRequest from the upstream stage output.

        Used for stages without a custom processor (e.g. code2wav).  Mirrors
        what the native orchestrator does via ``build_engine_core_request_from_tokens``
        and ``_forward_to_next_stage``.  Building an ``EngineCoreRequest``
        bypasses ``InputProcessor.process_inputs()`` which would fail for
        non-autoregressive stages (``worker_type: generation``) with
        "This model does not support generation".

        Raises RuntimeError on unexpected upstream output structure.
        """
        try:
            # engine_outputs[0]: first (and only) RequestOutput — Dynamo
            # processes one request at a time per stage.
            # outputs[0]: first CompletionOutput (n=1 sampling).
            # Matches native orchestrator's process_engine_inputs pattern.
            upstream = stage_list[-1].engine_outputs[0]
            token_ids = upstream.outputs[0].token_ids
        except (IndexError, AttributeError) as e:
            raise RuntimeError(
                f"Stage {self.stage_id}: cannot extract token_ids from "
                f"upstream output: {e}"
            ) from e

        tokens_prompt = OmniTokensPrompt(prompt_token_ids=list(token_ids))
        sp_list = _build_sampling_params(
            self.stage_config, sampling_params_list_override
        )
        params = sp_list[0] if sp_list else None
        prompt = build_engine_core_request_from_tokens(
            request_id=request_id,
            prompt=tokens_prompt,
            params=params,
        )
        # Pre-built EngineCoreRequests skip the output processor registration
        # in _build_add_request_message (the isinstance(prompt, EngineCoreRequest)
        # branch bypasses that block).  Register manually so that the engine's
        # output processor can match the response back to this request.
        prompt.external_req_id = prompt.request_id
        self.engine.engine.output_processors[0].add_request(
            request=prompt,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
        )
        return prompt

    def _process_stage_inputs(self, stage_list: list[_Proxy], original_prompt: Any):
        """Call vLLM-Omni stage processors using the v0.20 transition API."""
        if self._processor is None:
            raise RuntimeError(f"Stage {self.stage_id}: no processor configured")

        signature = inspect.signature(self._processor)
        positional_params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        parameter_names = [parameter.name for parameter in positional_params]

        if parameter_names[:2] == ["stage_list", "engine_input_source"]:
            logger.debug(
                "Stage %d: processor dispatch branch=stage_list parameters=%s",
                self.stage_id,
                parameter_names,
            )
            return self._processor(
                stage_list,
                self._engine_input_source,
                [original_prompt],
                self._requires_mm,
            )

        source_outputs = [
            output
            for stage_input in stage_list
            for output in (stage_input.engine_outputs or [])
        ]
        if _accepts_source_outputs_processor(parameter_names):
            logger.debug(
                "Stage %d: processor dispatch branch=source_outputs parameters=%s",
                self.stage_id,
                parameter_names,
            )
            if len(parameter_names) >= 4:
                return self._processor(
                    source_outputs,
                    original_prompt,
                    self._requires_mm,
                    None,
                )
            return self._processor(
                source_outputs,
                original_prompt,
                self._requires_mm,
            )

        raise TypeError(
            f"Stage {self.stage_id}: unsupported processor signature for "
            f"{self._processor!r}; expected stage-list parameters "
            "('stage_list', 'engine_input_source', ...) or source-output "
            "parameters ('source_outputs', 'original_prompt', ...), got "
            f"{parameter_names}"
        )

    def _fetch_stage_inputs(
        self, stage_connector_refs: dict[int, Any], request_id: str
    ) -> list[_Proxy]:
        """Backward-compatible synchronous wrapper for unit tests/callers.

        Runtime pipeline code should use ``_fetch_stage_inputs_async``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._fetch_stage_inputs_async(stage_connector_refs, request_id)
            )
        return self._fetch_stage_inputs_async(stage_connector_refs, request_id)  # type: ignore[return-value]

    async def _fetch_stage_inputs_async(
        self, stage_connector_refs: dict[int, Any], request_id: str
    ) -> list[_Proxy]:
        """Fetch previous stage outputs from connectors for the processor/engine.

        Fetches only the stages listed in engine_input_source (or all refs if empty).
        Returns _Proxy objects in engine_input_source order.
        Raises RuntimeError on any failure so the caller can propagate it as an error chunk.
        """
        sources = self._engine_input_source or sorted(stage_connector_refs.keys())
        stage_list = []
        for stage_k in sources:
            if (meta_k := stage_connector_refs.get(stage_k)) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector ref for source stage {stage_k}"
                )
            if (
                connector := self.connectors.get(_connector_key(stage_k, self.stage_id))
            ) is None:
                raise RuntimeError(
                    f"Stage {self.stage_id}: no connector for edge ({stage_k}→{self.stage_id})"
                )
            # Chunked upstream (work-item c): the producer streamed N blocks under
            # per-chunk keys instead of one payload. Gather them in order and
            # reassemble a single engine_inputs so the processor/engine sees one
            # combined input (the VAE decodes the whole clip in one streaming
            # forward — feat_cache must persist across all latent frames).
            if isinstance(meta_k, dict) and meta_k.get("chunked"):
                engine_inputs = await self._fetch_chunked_stage_input(
                    stage_k, request_id, int(meta_k.get("num_chunks", 0))
                )
                stage_list.append(_Proxy(engine_outputs=[engine_inputs]))
                continue
            try:
                get_result = await ensure_awaited(
                    connector.get(
                        str(stage_k),
                        str(self.stage_id),
                        request_id,
                        metadata=meta_k,
                    )
                )
            except Exception as e:
                raise RuntimeError(
                    f"Stage {self.stage_id}: connector.get() failed: {e}"
                ) from e
            engine_inputs = self._unwrap_stage_payload(
                get_result, f"({stage_k}→{self.stage_id})"
            )
            stage_list.append(_Proxy(engine_outputs=[engine_inputs]))
        return stage_list

    async def _fetch_chunked_stage_input(
        self, from_stage: int, request_id: str, num_chunks: int
    ) -> Any:
        """Reassemble a chunk-streamed upstream output into one engine_inputs.

        The producer (DiT) streamed ``num_chunks`` latent blocks under the
        per-chunk keys ``{request_id}_c{n}`` (work-item b). Fetch them in order
        and concatenate their latent tensors along the temporal axis (dim=2 of
        ``[B, C, T, H, W]``) into the first chunk's output, so the downstream
        processor (``dit2vae``) reads one full-clip latent exactly as it would
        from a single-payload upstream. Raises RuntimeError on a missing chunk.
        """
        if num_chunks <= 0:
            raise RuntimeError(
                f"Stage {self.stage_id}: chunked ref from stage {from_stage} has num_chunks={num_chunks}"
            )
        chunks = [
            await self.fetch_stage_chunk(from_stage, request_id, n)
            for n in range(num_chunks)
        ]
        combined = chunks[0]
        latents = [_primary_latent(c) for c in chunks]
        if any(lat is None for lat in latents):
            # No latent field (non-video upstream): fall back to the first chunk
            # untouched rather than guessing how to merge opaque payloads.
            return combined
        _set_primary_latent(combined, torch.cat(latents, dim=2))
        return combined

    async def fetch_stage_chunk(
        self, from_stage: int, request_id: str, chunk_index: int
    ) -> Any:
        """Fetch one streamed block by its per-chunk key (work-item b, §8).

        The consumer decodes each chunk exactly once, in order, so it reads a
        single key ``{request_id}_c{n}`` with no window and no metadata ticket —
        the SHM connector's ``get`` falls back to ``_get_by_key`` when metadata
        is omitted. The router (work-item c) drives this per block. Raises
        RuntimeError on a missing connector edge or an empty/absent chunk.
        """
        connector = self.connectors.get(_connector_key(from_stage, self.stage_id))
        if connector is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: no connector for edge ({from_stage}→{self.stage_id})"
            )
        try:
            get_result = await ensure_awaited(
                connector.get(
                    str(from_stage),
                    str(self.stage_id),
                    _chunk_key(request_id, chunk_index),
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Stage {self.stage_id}: connector.get() failed for chunk c{chunk_index}: {e}"
            ) from e
        return self._unwrap_stage_payload(
            get_result, f"({from_stage}→{self.stage_id}) chunk c{chunk_index}"
        )

    def _unwrap_stage_payload(self, get_result: Any, where: str) -> Any:
        """Unwrap a connector get result into engine_inputs (shared by both fetch paths)."""
        payload_data = unwrap_connector_payload(get_result)
        if is_empty_payload(payload_data):
            raise RuntimeError(
                f"Stage {self.stage_id}: empty payload from connector {where}"
            )
        if isinstance(payload_data, dict) and "engine_inputs" in payload_data:
            engine_inputs = payload_data["engine_inputs"]
            _restore_completion_output_attrs(
                engine_inputs,
                payload_data.get("_dynamo_completion_output_attrs"),
            )
        else:
            engine_inputs = payload_data
        _ensure_cumulative_token_ids(engine_inputs)
        return engine_inputs


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Initialize a single omni stage worker.

    Mirrors init_omni() setup pattern exactly to avoid routing/handler issues.
    """
    if config.stage_id is None:
        raise ValueError("--stage-id is required for stage worker initialization")
    stage_id: int = config.stage_id

    resolved_stage_configs_path, stage_configs = resolve_stage_configs_compat(
        config.model,
        config.stage_configs_path,
        trust_remote_code=getattr(
            getattr(config, "engine_args", None), "trust_remote_code", False
        ),
    )
    connector_configs_path = _ensure_stage_connectors(
        resolved_stage_configs_path,
        stage_configs,
    )
    # Only register NixlConnector if it's actually used in stage configs
    if _uses_nixl_connector(connector_configs_path, stage_configs):
        try:
            register_dynamoomni_nixl_connector()
        except Exception as e:
            logger.error("Stage %d: failed to register NixlConnector: %s", stage_id, e)
            raise

    if stage_id >= len(stage_configs):
        raise ValueError(
            f"--stage-id {stage_id} out of range (YAML has {len(stage_configs)} stages)"
        )
    my_config = stage_configs[stage_id]
    stage_type: str = getattr(my_config, "stage_type", "llm")

    # Stage worker registers at {ns}.{model_stage}.generate — NOT {ns}.backend.generate.
    # Router registers at {ns}.backend.generate and discovers workers by model_stage.
    model_stage = getattr(my_config.engine_args, "model_stage", f"stage{stage_id}")
    generate_endpoint = runtime.endpoint(f"{config.namespace}.{model_stage}.generate")
    shutdown_endpoints[:] = [generate_endpoint]

    engine = _create_engine(config.model, my_config, stage_type)
    logger.info("Stage %d: engine created (type=%s)", stage_id, stage_type)

    # Connectors for inter-stage output transfer — type determined by YAML config
    # (SharedMemoryConnector, MooncakeConnector, etc.)
    _, connectors = initialize_orchestrator_connectors(connector_configs_path)  # type: ignore[arg-type]

    worker = OmniStageWorker(
        engine=engine,
        stage_config=my_config,
        connectors=connectors,
        output_modalities=config.output_modalities,
        default_video_fps=config.default_video_fps,
        stage_id=stage_id,
    )

    setup_metrics_collection(config, generate_endpoint, logger)

    if config.engine_args.data_parallel_rank:
        logger.info(
            "Stage %d: non-leader DP rank %d; waiting for shutdown",
            stage_id,
            config.engine_args.data_parallel_rank,
        )
        if shutdown_event is not None:
            await shutdown_event.wait()
        return

    logger.info(
        "Stage %d: serving internal stage endpoint '%s' (not registering model)",
        stage_id,
        generate_endpoint,
    )
    health_check_payload = (
        await VllmOmniHealthCheckPayload.create(engine)  # type: ignore[arg-type]
    ).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            worker.generate,
            graceful_shutdown=True,
            metrics_labels=[
                (
                    prometheus_names.labels.MODEL,
                    config.served_model_name or config.model,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    config.served_model_name or config.model,
                ),
            ],
            health_check_payload=health_check_payload,
        )
    except Exception as e:
        logger.error("Stage %d: endpoint failed: %s", stage_id, e)
        raise


def _connector_key(from_stage: int | str, to_stage: int | str) -> tuple[str, str]:
    """Build the connector dict key used by initialize_orchestrator_connectors."""
    return (str(from_stage), str(to_stage))


def _chunk_key(request_id: str, chunk_index: int) -> str:
    """Per-chunk connector address (§8): ``{request_id}_c{n}``.

    Both producer (DiT) and consumer (VAE) derive this from ``request_id`` + the
    0-based chunk index alone, so the SHM connector can locate the block purely
    by key (``get(metadata=None)`` → ``_get_by_key``) with no handshake.
    """
    return f"{request_id}_c{chunk_index}"


def _chunk_finished(chunk: Any) -> bool:
    """Whether an engine output is a whole result rather than a stream block.

    Defaults to True for anything that does not report the field at all, which
    is the only safe default: ``DiffusionOutput.finished`` is itself ``True`` by
    default, and every non-diffusion engine output (an LLM ``RequestOutput``, a
    plain dict from a test double or a direct-API caller) is a complete result.
    Treating those as non-terminal would send them down the per-chunk lane and
    the request would then never see its terminal chunk.

    Attribute first, then mapping key, because the two engine paths disagree on
    shape and neither is wrong: diffusion yields dataclass-like outputs, while
    some callers hand back dicts.
    """
    if hasattr(chunk, "finished"):
        return bool(chunk.finished)
    if isinstance(chunk, Mapping):
        return bool(chunk.get("finished", True))
    return True


def cf_streaming_stage(req: StageRequest, request: dict) -> str | None:
    """Which half of a Causal-Forcing stream this request is, or None if it is not one.

    Two different signals, because a session id reaches the two stages by
    different routes and neither is available at both. Stage 0 gets it from
    ``nvext.annotations``, the only channel a client can extend without a Rust
    change. Stage N>0 gets it from the ``cf_session`` field the previous stage set
    on the hop, because ``nvext`` is not forwarded past stage 0.

    Returns ``"dit"`` for the rollout half, ``"vae"`` for the decode half. Total
    and cheap: an ordinary one-shot request carries neither signal and returns
    None, which is what keeps the batch path untouched.
    """
    if req.cf_session:
        return "vae"
    if has_cf_session(request.get("nvext")):
        return "dit"
    return None


def _cf_open_kwargs(request: dict) -> dict[str, Any]:
    """Geometry and seed for ``session_open``, read off the frontend request.

    Only what the *session* owns, which is what outlives one scene: the KV window
    is sized by height/width, and the stream seed is the base every scene's own
    seed derives from. Per-scene parameters travel on the scene instead.
    """
    nvext = request.get("nvext")
    nvext = nvext if isinstance(nvext, dict) else {}
    width, height = parse_size(request.get("size") or DEFAULT_VIDEO_SIZE)
    kwargs: dict[str, Any] = {
        "height": int(nvext.get("height") or height),
        "width": int(nvext.get("width") or width),
    }
    if nvext.get("seed") is not None:
        kwargs["seed"] = int(nvext["seed"])
    return kwargs


def _uses_nixl_connector(stage_configs_path: str, stage_configs: list[Any]) -> bool:
    """Check if any stage connector uses NixlConnector."""
    try:
        with open(stage_configs_path) as f:
            raw = f.read()
    except OSError:
        return False

    try:
        deploy_config = yaml.safe_load(raw) or {}
    except Exception as exc:
        logger.error(
            "_uses_nixl_connector: failed to parse %s: %s", stage_configs_path, exc
        )
        raise

    if not isinstance(deploy_config, dict):
        raise ValueError(
            f"_uses_nixl_connector: {stage_configs_path} did not yield a mapping "
            f"(got {type(deploy_config).__name__})"
        )

    # Check both root-level connectors and runtime.connectors (YAML structure varies)
    connectors_list = []

    # Root-level connectors (synthesized by _ensure_stage_connectors)
    if isinstance(deploy_config.get("connectors"), dict):
        connectors_list.append(deploy_config["connectors"])

    # Runtime.connectors (user-defined in stage config YAML)
    runtime = deploy_config.get("runtime")
    if isinstance(runtime, dict) and isinstance(runtime.get("connectors"), dict):
        connectors_list.append(runtime["connectors"])

    for connectors in connectors_list:
        for connector_config in connectors.values():
            if not isinstance(connector_config, dict):
                continue
            connector_type = connector_config.get("name", "")
            if connector_type == "NixlConnector":
                return True

    return False


def _load_processor(func_path: str | None) -> Any:
    """Load a processor function from a dotted module path, or return None."""
    if not func_path:
        return None
    module_path, func_name = func_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), func_name)


def _ensure_stage_connectors(stage_configs_path: str, stage_configs: list[Any]) -> str:
    """Add default SHM connector edges for stage configs that omit them.

    Two kinds of edge are synthesized: the inter-stage ``from_stage_N`` inputs implied
    by each stage's ``engine_input_source``, and the final stage's output edge to the
    router.

    The router edge exists for the pipelined Causal-Forcing path. A serial request may
    fall back to ``shm_write_bytes(..., name=request_id)``, but a stream may not: it
    emits many blocks under one request id, and ``SharedMemoryConnector`` names both
    the segment and its lockfile after the key, so every block would overwrite its
    predecessor while contending on a single lock. ``_cf_put_block`` therefore refuses
    that fallback rather than corrupting the stream, which leaves this the place the
    edge has to come from. Both the router and the stage worker call this function, so
    the two ends stay in agreement about which edges exist.
    """
    try:
        with open(stage_configs_path) as f:
            deploy_config = yaml.safe_load(f) or {}
    except OSError:
        logger.warning(
            "Could not read stage config %s; using it without connector synthesis",
            stage_configs_path,
        )
        return stage_configs_path

    if not isinstance(deploy_config, dict):
        return stage_configs_path

    stages = deploy_config.get("stages")
    if not isinstance(stages, list):
        return stage_configs_path

    stages_by_id = {
        int(stage.get("stage_id", idx)): stage
        for idx, stage in enumerate(stages)
        if isinstance(stage, dict)
    }
    connector_name = "connector_of_shared_memory"
    changed = False

    for stage_config in stage_configs:
        to_stage = int(getattr(stage_config, "stage_id", -1))
        if to_stage < 0:
            continue
        stage = stages_by_id.get(to_stage)
        if stage is None:
            continue
        input_connectors = stage.setdefault("input_connectors", {})
        if not isinstance(input_connectors, dict):
            continue
        for from_stage in getattr(stage_config, "engine_input_source", []) or []:
            connector_key = f"from_stage_{int(from_stage)}"
            if connector_key not in input_connectors:
                input_connectors[connector_key] = connector_name
                changed = True

    # The final stage's edge to the router. Keyed "to_stage_router" because
    # initialize_connectors_from_config strips the "to_stage_" prefix to name the
    # far end, giving the ("<final>", "router") key that _connector_key builds and
    # both _cf_put_block and the router's own fetch look up.
    final_stage_id = max(stages_by_id, default=None)
    if final_stage_id is not None:
        final_stage = stages_by_id[final_stage_id]
        output_connectors = final_stage.setdefault("output_connectors", {})
        if isinstance(output_connectors, dict):
            if "to_stage_router" not in output_connectors:
                output_connectors["to_stage_router"] = connector_name
                changed = True

    if not changed:
        return stage_configs_path

    connectors = deploy_config.setdefault("connectors", {})
    if not isinstance(connectors, dict):
        raise ValueError(
            f"'connectors' in {stage_configs_path} must be a mapping to "
            f"synthesize {connector_name}; got {type(connectors).__name__}"
        )
    connectors.setdefault(
        connector_name,
        {
            "name": "SharedMemoryConnector",
            "extra": {},
        },
    )

    tmp_dir = tempfile.mkdtemp(prefix=f"dynamo_omni_stage_{os.getpid()}_")
    tmp_path = os.path.join(tmp_dir, "stage_config.yaml")
    with open(tmp_path, "w") as tmp:
        yaml.safe_dump(deploy_config, tmp, sort_keys=False)

    atexit.register(_cleanup_temp_stage_config, tmp_dir)
    logger.info(
        "Synthesized default SharedMemoryConnector edges in %s from %s",
        tmp_path,
        stage_configs_path,
    )
    return tmp_path


def _cleanup_temp_stage_config(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)
    except OSError:
        pass


def _prepare_connector_payload(
    engine_inputs: Any,
    from_stage: int | None = None,
    to_stage: int | str | None = None,
) -> Any:
    """Build connector payload for inter-stage transfer.

    Connector payloads are regular Python objects. Connectors that advertise
    raw-data support (including NIXL) can serialize/deserialize these payloads
    directly
    """
    _ = (from_stage, to_stage)
    # Preserve completion-only fields that some serializers may drop.
    _promote_request_multimodal_output(engine_inputs)
    output_attrs = _collect_completion_output_attrs(engine_inputs)
    if len(output_attrs) == 0:
        return engine_inputs
    return {
        "engine_inputs": engine_inputs,
        "_dynamo_completion_output_attrs": output_attrs,
    }


def _collect_completion_output_attrs(engine_inputs: Any) -> list[dict[str, Any]]:
    output_attrs: list[dict[str, Any]] = []
    for output in _iter_completion_outputs(engine_inputs):
        attrs: dict[str, Any] = {}
        cumulative_token_ids = getattr(output, "cumulative_token_ids", None)
        if cumulative_token_ids is not None:
            attrs["cumulative_token_ids"] = list(cumulative_token_ids)
        multimodal_output = getattr(output, "multimodal_output", None)
        if multimodal_output is not None and not is_empty_payload(multimodal_output):
            attrs["multimodal_output"] = multimodal_output
        output_attrs.append(attrs)
    return output_attrs


def _promote_request_multimodal_output(engine_inputs: Any) -> None:
    """Expose request-level multimodal payloads on the sole completion output."""
    request_multimodal_output = getattr(engine_inputs, "multimodal_output", None)
    if request_multimodal_output is None or is_empty_payload(request_multimodal_output):
        return

    outputs = _iter_completion_outputs(engine_inputs)
    if len(outputs) != 1:
        return

    completion = outputs[0]
    completion_mm = getattr(completion, "multimodal_output", None)
    if completion_mm is None or is_empty_payload(completion_mm):
        completion.multimodal_output = request_multimodal_output


def _restore_completion_output_attrs(
    engine_inputs: Any, output_attrs: Any | None
) -> None:
    if not isinstance(output_attrs, list):
        return
    for output, attrs in zip(
        _iter_completion_outputs(engine_inputs), output_attrs, strict=False
    ):
        if not isinstance(attrs, dict):
            continue
        if "cumulative_token_ids" in attrs:
            output.cumulative_token_ids = list(attrs["cumulative_token_ids"])
        if "multimodal_output" in attrs:
            output.multimodal_output = attrs["multimodal_output"]


def _primary_latent(engine_inputs: Any) -> torch.Tensor | None:
    """Read the DiT latent tensor off a stage output (work-item c reassembly).

    Mirrors ``dit2vae._latent_from_output``: the streamed DiT block is a diffusion
    ``OmniRequestOutput`` whose post-processed latent lands on the *top-level*
    ``multimodal_output['latent']`` (backed by ``_multimodal_output``) — the only
    channel the inter-stage connector preserves (``.images`` is dropped). Read
    that first, then any completion-output ``multimodal_output`` (the pipeline-
    stage shape), then ``.images[0]`` (the in-process no-connector channel).
    Returns None when no latent tensor is present so the caller can fall back.
    """
    mm = getattr(engine_inputs, "multimodal_output", None)
    if isinstance(mm, dict) and isinstance(mm.get("latent"), torch.Tensor):
        return mm["latent"]
    for output in _iter_completion_outputs(engine_inputs):
        mm = getattr(output, "multimodal_output", None)
        if isinstance(mm, dict) and isinstance(mm.get("latent"), torch.Tensor):
            return mm["latent"]
    images = getattr(engine_inputs, "images", None)
    if images:
        first = images[0] if isinstance(images, (list, tuple)) else images
        if isinstance(first, torch.Tensor):
            return first
    return None


def _set_primary_latent(engine_inputs: Any, latent: torch.Tensor) -> None:
    """Write the combined latent back onto the reassembled stage output.

    Sets whichever channel ``_primary_latent`` reads (top-level
    multimodal_output first, then a completion output's, else ``.images[0]``) so
    ``dit2vae`` sees the full-clip latent.
    """
    mm = getattr(engine_inputs, "multimodal_output", None)
    if isinstance(mm, dict) and isinstance(mm.get("latent"), torch.Tensor):
        mm["latent"] = latent
        return
    for output in _iter_completion_outputs(engine_inputs):
        mm = getattr(output, "multimodal_output", None)
        if isinstance(mm, dict) and isinstance(mm.get("latent"), torch.Tensor):
            mm["latent"] = latent
            return
    images = getattr(engine_inputs, "images", None)
    if images and isinstance(images, list) and isinstance(images[0], torch.Tensor):
        images[0] = latent


def _ensure_cumulative_token_ids(engine_inputs: Any) -> None:
    """Bridge vLLM 0.20 CompletionOutput into vLLM-Omni stage processors."""
    for output in _iter_completion_outputs(engine_inputs):
        if not hasattr(output, "cumulative_token_ids") and hasattr(output, "token_ids"):
            output.cumulative_token_ids = list(output.token_ids)


def _iter_completion_outputs(engine_inputs: Any):
    outputs = getattr(engine_inputs, "outputs", None)
    if outputs is None:
        request_output = getattr(engine_inputs, "request_output", None)
        outputs = getattr(request_output, "outputs", None)
    if outputs is None:
        return []
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    if isinstance(outputs, torch.Tensor):
        return []
    try:
        return list(outputs)
    except TypeError:
        return []


def _accepts_source_outputs_processor(parameter_names: list[str]) -> bool:
    if len(parameter_names) < 3:
        return False
    return (
        parameter_names[0] == "source_outputs"
        and (parameter_names[1] in {"original_prompt", "prompt"})
        and (parameter_names[2] in {"requires_mm", "requires_multimodal_data"})
    )


def _create_engine(model: str, stage_config: Any, stage_type: str) -> StageEngine:
    """Create AsyncOmni with a single-stage YAML."""
    stage_arg = _stage_config_to_dict(stage_config, stage_type)
    _normalize_single_stage_runtime_devices(stage_arg)
    single_stage_config = {
        "stage_args": [stage_arg],
        "runtime": {"edges": []},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(single_stage_config, tmp)
        tmp_path = tmp.name

    try:
        return AsyncOmni(model=model, stage_configs_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def _stage_config_to_dict(stage_config: Any, stage_type: str) -> dict:
    """Convert a parsed stage config to a single-stage YAML dict."""
    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    def _to_plain(obj: Any) -> Any:
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        return obj

    result: dict = {
        "stage_id": 0,
        "stage_type": stage_type,
        "engine_args": _to_plain(stage_config.engine_args),
        "final_output": True,
        "final_output_type": getattr(stage_config, "final_output_type", "text"),
    }

    for key in ("default_sampling_params", "is_comprehension"):
        val = getattr(stage_config, key, None)
        if val is not None:
            result[key] = _to_plain(val)

    engine_input_source = getattr(stage_config, "engine_input_source", None)
    if engine_input_source is not None:
        result["engine_input_source"] = _to_plain(engine_input_source)

    runtime = getattr(stage_config, "runtime", None)
    if runtime is not None:
        rt = _to_plain(runtime)
        rt.setdefault("devices", "0")
        result["runtime"] = rt

    return result


def _normalize_single_stage_runtime_devices(stage_arg: dict) -> None:
    """Map stage-local device visibility to vLLM-Omni logical device IDs."""
    runtime = stage_arg.get("runtime")
    if not isinstance(runtime, dict):
        return

    devices = runtime.get("devices")
    visible_devices = _get_visible_devices()
    if devices in (None, "cpu") or not visible_devices:
        return

    requested_devices = _parse_runtime_devices(devices)
    if requested_devices != visible_devices:
        return

    # Dynamo starts each stage worker with the process visibility already
    # narrowed to that stage's devices. vLLM-Omni then interprets runtime.devices
    # as logical indexes inside that visible set.
    runtime["devices"] = ",".join(str(i) for i in range(len(requested_devices)))


def _get_visible_devices() -> list[str]:
    for env_var in (
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
    ):
        if devices := os.environ.get(env_var):
            return _parse_runtime_devices(devices)
    return []


def _parse_runtime_devices(devices: Any) -> list[str]:
    if isinstance(devices, int):
        return [str(devices)]
    if isinstance(devices, str):
        return [device.strip() for device in devices.split(",") if device.strip()]
    if isinstance(devices, (list, tuple)):
        return [str(device).strip() for device in devices if str(device).strip()]
    return []


def _resolve_model_type(final_output_type: str) -> ModelType:
    return {
        "image": ModelType.Images,
        "video": ModelType.Videos,
    }.get(final_output_type, ModelType.Chat)
