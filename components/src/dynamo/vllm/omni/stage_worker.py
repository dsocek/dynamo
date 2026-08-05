# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-stage omni worker for disaggregated pipelines."""

import asyncio
import atexit
import copy
import importlib
import inspect
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import torch
import yaml
from vllm_omni.diffusion.block_stream import (
    FIRST_TIMEOUT_S as BLOCK_FIRST_TIMEOUT_S,
)
from vllm_omni.diffusion.block_stream import (
    POLL_MAX_S as BLOCK_POLL_MAX_S,
)
from vllm_omni.diffusion.block_stream import (
    POLL_MIN_S as BLOCK_POLL_MIN_S,
)
from vllm_omni.diffusion.block_stream import (
    STALL_WARN_S as BLOCK_STALL_WARN_S,
)
from vllm_omni.diffusion.block_stream import (
    ShmLatentBlockStream,
    chunk_end_key,
    chunk_key,
)
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.stage_utils import serialize_obj, shm_write_bytes
from vllm_omni.inputs.data import OmniTokensPrompt

from dynamo import prometheus_names
from dynamo.llm import ModelType
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.health_check import VllmOmniHealthCheckPayload
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.connectors import register_dynamoomni_nixl_connector
from dynamo.vllm.omni.types import StageEngine, StageRequest, _int_keyed
from dynamo.vllm.omni.utils import (
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
        n_engine_outputs = 0
        try:
            async for chunk in self.engine.generate(
                prompt, request_id=request_id, sampling_params_list=sp
            ):
                # [cmaf-trace] Every engine output as this worker sees it, before
                # any lane decision. `finished` alone selects the streaming lane
                # vs. the terminal path, and a chunk carrying no images takes the
                # streaming lane but writes an empty pixel chunk — so log both,
                # plus the type, since a mis-shaped output is otherwise silent.
                n_engine_outputs += 1
                _imgs = getattr(chunk, "images", None)
                logger.info(
                    "[cmaf-trace] stage %d: engine output #%d type=%s finished=%s "
                    "n_images=%s has_connector=%s",
                    self.stage_id,
                    n_engine_outputs - 1,
                    type(chunk).__name__,
                    getattr(chunk, "finished", "<absent>"),
                    len(_imgs) if isinstance(_imgs, (list, tuple)) else ("1" if _imgs is not None else 0),
                    connector is not None,
                )
                if not bool(getattr(chunk, "finished", False)):
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
                        ready: dict = {
                            "chunk_index": n_streamed,
                            "is_last": False,
                            "finished": False,
                        }
                        if n_streamed == 0:
                            # [cmaf-step2] The router dispatches the VAE concurrently
                            # and so can no longer wait for the DiT terminal to learn
                            # the request facts. Carry them on the *first* ready
                            # signal — by then block c0 is on the connector, which is
                            # the earliest the VAE could start anyway.
                            ready["original_prompt"] = original_prompt
                            if sampling_params_list_override is not None:
                                ready["sampling_params_list"] = (
                                    sampling_params_list_override
                                )
                            logger.info(
                                "[cmaf-step2] stage %d (DiT): first block out for %s; "
                                "ready signal carries request facts for concurrent VAE dispatch",
                                self.stage_id, request_id,
                            )
                        yield ready
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

        # [cmaf-trace] Which exit this worker takes. n_streamed==0 falls through to
        # the whole-clip path even in a streaming deployment — a silent mode
        # downgrade that looks like a normal batch run in the log, so state the
        # branch outright rather than leaving it to be inferred.
        logger.info(
            "[cmaf-trace] stage %d: engine stream ended for %s — %d engine outputs, "
            "%d streamed chunks; taking %s path",
            self.stage_id,
            request_id,
            n_engine_outputs,
            n_streamed,
            "STREAMED terminal"
            if n_streamed > 0
            else "WHOLE-CLIP (no chunks were streamed)",
        )

        if n_streamed > 0:
            if connector is not None:
                # [cmaf-step2] Publish the stream-end marker under its own key.
                # A concurrently-running consumer polls {rid}_c{n} and cannot
                # distinguish "block not written yet" from "stream over" (the
                # connector's get is non-blocking and returns None for both), and
                # it is not waiting for this RPC terminal any more. The marker is
                # the authoritative termination signal: it is written only after
                # every block is on the connector, and carries the final count.
                await self._put_stream_end(
                    connector, from_s, to_s, request_id, n_streamed
                )
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
        # Scrubbed for the same reason as the per-chunk write: a live block-stream
        # descriptor on the echoed-back request state is not serializable, and this
        # exit is reachable on the streaming config whenever no chunk was streamed.
        # The graph dump is here too — this is the terminal reply, so a failing
        # serialize ends the request and is exactly as expensive to re-diagnose.
        scrubbed_final = _strip_unserializable_prompt(last_result)
        try:
            final_payload = serialize_obj(scrubbed_final)
        except Exception as e:
            logger.error(
                "[cmaf-trace] serialize failed for the terminal reply of %s — object "
                "graph of the scrubbed result follows:\n%s",
                request_id,
                "\n".join(_describe_object_graph(scrubbed_final, path="last_result")),
            )
            yield {"error": f"terminal SHM write failed: {e}", "finished": True}
            return
        shm_meta = shm_write_bytes(final_payload, name=request_id)
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

    async def _put_stream_end(
        self,
        connector: Any,
        from_s: str,
        to_s: str,
        request_id: str,
        num_chunks: int,
    ) -> bool:
        """Publish the stream-end marker for a concurrent consumer (§4.2 Step 2).

        Written under ``{request_id}_cend`` once every block is already on the
        connector, so a consumer that sees it knows blocks ``c0..c{num_chunks-1}``
        are all present. This is the sole termination signal for the concurrent
        path, where ``num_chunks`` is not known when the VAE is dispatched.

        Why a separate key rather than an ``is_last`` flag stamped into the last
        block's payload: at put time the producer does not yet know a block is the
        last one (the engine reveals that only with the following terminal
        output), so stamping in-payload would mean holding every block back by one
        — a full block of added latency on the very path this step exists to speed
        up. A marker key costs nothing and keeps blocks flowing immediately.
        """
        payload = {"stream_end": True, "num_chunks": int(num_chunks)}
        try:
            ok, _, _ = await ensure_awaited(
                connector.put(  # type: ignore[arg-type]
                    from_s, to_s, _chunk_end_key(request_id), payload
                )
            )
        except Exception as e:
            logger.error(
                "Stage %d: stream-end marker put() raised for %s: %s: %s",
                self.stage_id, request_id, type(e).__name__, e, exc_info=True,
            )
            return False
        logger.info(
            "[cmaf-step2] stage %d (DiT): stream-end marker written for %s (%d blocks)",
            self.stage_id, request_id, num_chunks,
        )
        return bool(ok)

    def _write_stream_chunk_shm(
        self, request_id: str, chunk_index: int, chunk: Any
    ) -> dict:
        """Write one streamed pixel chunk to SHM under its per-chunk name (§c).

        The final (VAE) stage has no downstream connector, so pixel chunks reach
        the router the same way the aggregated final output does — via SHM — but
        under a per-chunk name so streamed frames don't clobber each other.

        The name is namespaced ``{request_id}_p{n}``, deliberately *not* the
        ``_c{n}`` latent-block key: both producers share one flat SHM namespace,
        and ``shm_write_bytes`` unlinks and recreates on a name clash. Under the
        concurrent path (§4.2 Step 2) latents and pixels are in flight at the same
        time, so a pixel chunk on the latent's key would destroy a latent block the
        decode has not read yet. Returns the shm_meta ref for the router to read
        with ``shm_deserialize``.

        The chunk is scrubbed of live block-stream descriptors first. Request state
        is echoed back onto every ``OmniRequestOutput`` (``prompt``,
        ``multimodal_output``, ``images``), and on this stage that state carries
        ``extra.latents`` — under §4.2 Step 2 a live
        :class:`ShmLatentBlockStream`, not a tensor. ``serialize_obj`` deep-walks
        dataclasses via ``asdict``, so it reaches the descriptor and raises
        ``TypeError: Object of type ShmLatentBlockStream is not serializable``,
        failing the very first pixel chunk. That state is echo-back only — both
        consumers of a pixel chunk read ``.images`` and nothing else — so clearing
        it here costs nothing and keeps the descriptor on the one side of the
        boundary where it is meant to live.

        If a serialize still fails, the chunk's whole object graph is logged before
        re-raising. The serializer names the offending *type* but not its location,
        and location is the fact needed to fix it; a remote run is too expensive to
        spend on learning it twice.
        """
        scrubbed = _strip_unserializable_prompt(chunk)
        try:
            payload = serialize_obj(scrubbed)
        except Exception:
            logger.error(
                "[cmaf-trace] serialize failed for pixel chunk %d of %s — object "
                "graph of the scrubbed chunk follows:\n%s",
                chunk_index,
                request_id,
                "\n".join(_describe_object_graph(scrubbed)),
            )
            raise
        return shm_write_bytes(
            payload, name=_pixel_chunk_key(request_id, chunk_index)
        )

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
                # num_chunks is absent on the concurrent path (§4.2 Step 2): the
                # router synthesizes {chunked: true} at dispatch time, before the
                # producer knows how many blocks it will emit. None => poll until
                # the stream-end marker appears.
                raw_count = meta_k.get("num_chunks")
                engine_inputs = await self._fetch_chunked_stage_input(
                    stage_k,
                    request_id,
                    int(raw_count) if raw_count is not None else None,
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
        self, from_stage: int, request_id: str, num_chunks: int | None
    ) -> Any:
        """Reassemble a chunk-streamed upstream output into one engine_inputs.

        The producer (DiT) streamed latent blocks under the per-chunk keys
        ``{request_id}_c{n}`` (work-item b). Their latent tensors are carried
        through as a **list of per-block tensors** on the first chunk's output, so
        the downstream processor (``dit2vae``) and the VAE pipeline consume the
        blocks one at a time into a persistent decode session (§4.2 Step 1). No
        ``torch.cat`` — the VAE's ``feat_cache`` spans the blocks in order, so a
        per-block list decodes byte-identically to a single cat'd tensor, without
        materializing the whole clip.

        Two modes, by whether ``num_chunks`` is known:

        - **known** (Step 1 / serialized): the producer already finished, so every
          block is on the connector. Fetch ``c0..c{num_chunks-1}`` eagerly and hand
          the pipeline a plain list. Unchanged behavior.
        - **None** (Step 2 / concurrent): the producer is *still running*. Blocks
          are streamed to the pipeline through a blocking queue as they land, so
          decode overlaps the rollout instead of waiting for it. Returns as soon as
          block ``c0`` is available — the rest arrive on the queue.
        """
        if num_chunks is not None:
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
            # Carry the per-block latents as a list (not a cat'd tensor); the VAE
            # pipeline flattens and streams them one frame at a time (§4.2 Step 1).
            logger.info(
                "[cmaf-step1] stage %d: carrying %d latent blocks as list (no torch.cat) for %s",
                self.stage_id,
                len(latents),
                request_id,
            )
            _set_primary_latent(combined, latents)
            return combined
        return await self._fetch_chunked_stage_input_streaming(from_stage, request_id)

    async def _fetch_chunked_stage_input_streaming(
        self, from_stage: int, request_id: str
    ) -> Any:
        """Hand the pipeline a live block stream, while the producer still runs.

        Step 2 (§4.2). The decode does **not** run in this process: the executor
        pickles the request through ``shm_broadcast`` to a diffusion worker process.
        So we cannot bridge blocks over a queue, a task, or a connector instance —
        none of those survive pickling. Instead we ship a ``ShmLatentBlockStream``:
        a plain description of *where to look* (request id, stage labels, connector
        config), which rebuilds its own reader after unpickling and polls the
        per-chunk keys itself, in the process that actually decodes.

        Block ``c0`` is awaited here, on the event loop, before the descriptor is
        built: the downstream ``dit2vae`` processor needs a real tensor to derive
        pixel geometry from, and it is what the engine's request payload is built
        around. Blocks ``c1..`` are never touched by this process.
        """
        first = await self._await_first_chunk(from_stage, request_id)
        if first is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: chunked stream from stage {from_stage} "
                f"for {request_id} ended before any block was produced"
            )
        first_latent = _primary_latent(first)
        if first_latent is None:
            # Non-video upstream: no latent to stream — nothing to bridge.
            return first

        connector = self.connectors.get(_connector_key(from_stage, self.stage_id))
        connector_config = getattr(connector, "config", None)
        if not isinstance(connector_config, dict):
            raise RuntimeError(
                f"Stage {self.stage_id}: connector for edge ({from_stage}→"
                f"{self.stage_id}) exposes no config dict, so the decode process "
                f"cannot rebuild a reader for the live block stream"
            )
        stream = ShmLatentBlockStream(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=self.stage_id,
            connector_config=connector_config,
            first_block=first_latent,
        )
        logger.info(
            "[cmaf-step2] stage %d: live block stream handed to decode for %s "
            "(first block %s in hand; remaining blocks polled from SHM inside the "
            "decode process)",
            self.stage_id, request_id, tuple(first_latent.shape),
        )
        _set_primary_latent(first, stream)
        return first

    async def _await_first_chunk(self, from_stage: int, request_id: str) -> Any | None:
        """Poll ``{rid}_c0`` on the event loop until it appears, or the stream ends.

        Only block ``c0`` is awaited here (the rest are polled inside the decode
        process, see ``_fetch_chunked_stage_input_streaming``), but the termination
        logic is the same and for the same reason: the connector's ``get`` is
        non-blocking and returns nothing both for "not written yet" and "never will
        be", so absence alone is ambiguous. The stream-end marker disambiguates — it
        is written only *after* every block is on the connector, so marker-present
        plus ``c0``-absent proves the producer emitted no blocks at all. Checking the
        marker *after* a failed block read (not before) closes the race where both
        land between the two reads.

        Returns the chunk's engine_inputs, or None if the stream ended empty.
        """
        connector = self.connectors.get(_connector_key(from_stage, self.stage_id))
        if connector is None:
            raise RuntimeError(
                f"Stage {self.stage_id}: no connector for edge ({from_stage}→{self.stage_id})"
            )
        delay = BLOCK_POLL_MIN_S
        t_start = time.monotonic()
        next_warn = BLOCK_STALL_WARN_S
        polls = 0
        while True:
            chunk = await self._try_get_key(
                connector, from_stage, _chunk_key(request_id, 0)
            )
            if chunk is not None:
                waited = time.monotonic() - t_start
                logger.info(
                    "[cmaf-step2] stage %d: first block c0 for %s available after "
                    "%.1fs (%d polls) — decode can start now, concurrent with the "
                    "rest of the upstream rollout",
                    self.stage_id, request_id, waited, polls,
                )
                return chunk
            end = await self._try_get_key(
                connector, from_stage, _chunk_end_key(request_id)
            )
            if end is not None:
                # Re-read once: c0 may have landed between the two reads above, in
                # which case the marker does not imply absence.
                chunk = await self._try_get_key(
                    connector, from_stage, _chunk_key(request_id, 0)
                )
                if chunk is not None:
                    logger.info(
                        "[cmaf-step2] stage %d: block c0 for %s landed in the "
                        "end-marker race window (recovered, not a lost block)",
                        self.stage_id, request_id,
                    )
                    return chunk
                logger.error(
                    "[cmaf-step2] stage %d: stream for %s ended with no blocks "
                    "(producer reported %s)",
                    self.stage_id,
                    request_id,
                    end.get("num_chunks") if isinstance(end, dict) else end,
                )
                return None

            waited = time.monotonic() - t_start
            if waited >= next_warn:
                # A producer that died before writing the marker looks exactly like
                # a slow one from here, so say so rather than hanging silently.
                logger.warning(
                    "[cmaf-step2] stage %d: still waiting for block c0 of %s after "
                    "%.1fs (%d polls, no stream-end marker). Upstream is either slow "
                    "or died without publishing the marker; will fail at %.0fs.",
                    self.stage_id, request_id, waited, polls, BLOCK_FIRST_TIMEOUT_S,
                )
                next_warn += BLOCK_STALL_WARN_S
            if waited >= BLOCK_FIRST_TIMEOUT_S:
                raise RuntimeError(
                    f"Stage {self.stage_id}: timed out after {waited:.0f}s waiting for "
                    f"block c0 of {request_id} with no stream-end marker — upstream "
                    f"stage {from_stage} never produced a block. Check that it started "
                    f"and is block-streaming (stream_dit_blocks); raise "
                    f"DYN_CMAF_FIRST_BLOCK_TIMEOUT_S if warmup is simply this slow."
                )
            polls += 1
            await asyncio.sleep(delay)
            delay = min(delay * 2, BLOCK_POLL_MAX_S)

    async def _try_get_key(
        self, connector: Any, from_stage: int, key: str
    ) -> Any | None:
        """One non-blocking connector read: engine_inputs, or None if absent."""
        try:
            get_result = await ensure_awaited(
                connector.get(str(from_stage), str(self.stage_id), key)
            )
        except Exception:
            # A miss can surface as a raise depending on connector; treat as absent.
            return None
        payload = unwrap_connector_payload(get_result)
        if is_empty_payload(payload):
            return None
        if isinstance(payload, dict) and payload.get("stream_end"):
            return payload
        if isinstance(payload, dict) and "engine_inputs" in payload:
            engine_inputs = payload["engine_inputs"]
            _restore_completion_output_attrs(
                engine_inputs, payload.get("_dynamo_completion_output_attrs")
            )
        else:
            engine_inputs = payload
        _ensure_cumulative_token_ids(engine_inputs)
        return engine_inputs

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


# Per-chunk connector addresses (§8, §4.2 Step 2). Both producer (this worker) and
# consumer (the decode process, via ShmLatentBlockStream) derive them from
# ``request_id`` alone, so the SHM connector can locate a block purely by key
# (``get(metadata=None)`` → ``_get_by_key``) with no handshake. Imported rather than
# redefined here precisely because the two sides must never drift apart.
_chunk_key = chunk_key
_chunk_end_key = chunk_end_key


def _pixel_chunk_key(request_id: str, chunk_index: int) -> str:
    """Final-stage pixel-chunk SHM name: ``{request_id}_p{n}``.

    A namespace of its own, so a pixel chunk can never land on a latent block's
    key while both are in flight (see ``_write_stream_chunk_shm``).
    """
    return f"{request_id}_p{chunk_index}"


_SCRUB_MAX_DEPTH = 12


def _is_scrub_leaf(obj: Any) -> bool:
    """True for values the serializer handles directly and never walks into.

    Mirrors ``OmniMsgpackEncoder``: tensors, ndarrays and images are encoded
    wholesale, and scalars are native msgpack types. Descending into them would
    cost real time on a pixel chunk and can never find a descriptor.
    """
    if obj is None or isinstance(obj, (str, bytes, bytearray, bool, int, float)):
        return True
    if isinstance(obj, torch.Tensor):
        return True
    module = type(obj).__module__ or ""
    return module.startswith("numpy") or module.startswith("PIL")


def _scrub_unserializable_refs(
    obj: Any,
    *,
    path: str = "chunk",
    found: list[str] | None = None,
    cycles: list[str] | None = None,
    depth: int = 0,
    memo: dict[int, Any] | None = None,
    in_progress: set[int] | None = None,
) -> Any:
    """Return ``obj`` with every live block-stream descriptor replaced by ``None``.

    A streamed pixel chunk crosses a serializing boundary (SHM), but carries
    request state echoed back on it — and on the VAE stage under the concurrent
    path (§4.2 Step 2) that state includes a live :class:`ShmLatentBlockStream`:
    a descriptor that exists to be *read* in this process, which the serializer
    rejects outright.

    The descriptor reaches the chunk through more than one channel, and which one
    carries it is not knowable from this side: ``dit2vae`` copies it into
    ``prompt['extra']['latents']``, ``_set_primary_latent`` writes it to
    ``multimodal_output['latent']`` or ``images[0]``, and ``_forward_vae_decode``
    also reads ``sampling_params.extra_args['latents']``. Clearing one named
    field therefore fixes one route and leaves the others live — so this walks the
    whole graph the serializer walks and clears the descriptor wherever it sits.

    Traversal mirrors ``OmniMsgpackEncoder``: dicts, sequences, and any object
    with a ``__dict__`` (which covers dataclass fields *and* the dynamic
    ``multimodal_output`` attributes the encoder reads by hand). Copy-on-write
    throughout: a container is rebuilt only if a descendant changed, so the live
    prompt the decode is still iterating is never mutated out from under it.

    ``memo`` maps visited object ids to their scrubbed result. Request state is
    graph-shaped, not tree-shaped — ``dit2vae`` and ``_set_primary_latent`` put
    the *same* descriptor on several channels — so memoizing does two jobs: it
    terminates the walk, and it returns the scrubbed replacement for a node
    reached a second time. A plain visited-set would return the original there and
    leave the descriptor live on every path but the first.

    A true *cycle* (a back-edge to a node still being walked) is not handled and
    cannot be: copy-on-write would have to patch a clone that does not exist yet,
    and the descriptor stays reachable through the back-edge. It is also moot —
    ``OmniMsgpackEncoder`` has no cycle support, so such a graph cannot be
    serialized either way. It is reported through ``cycles`` instead of being
    silently half-scrubbed.

    ``found`` collects the paths that were cleared, so the caller can log *which*
    channel carried the descriptor rather than leaving it to be inferred.
    ``cycles`` collects back-edge paths for the same reason.
    """
    if found is None:
        found = []
    if cycles is None:
        cycles = []
    if memo is None:
        memo = {}
    if in_progress is None:
        in_progress = set()

    if isinstance(obj, ShmLatentBlockStream):
        found.append(path)
        return None

    if _is_scrub_leaf(obj) or depth >= _SCRUB_MAX_DEPTH:
        return obj

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]
    if obj_id in in_progress:
        cycles.append(path)
        return obj
    in_progress.add(obj_id)

    def _child(value: Any, child_path: str) -> Any:
        return _scrub_unserializable_refs(
            value,
            path=child_path,
            found=found,
            cycles=cycles,
            depth=depth + 1,
            memo=memo,
            in_progress=in_progress,
        )

    def _record(result: Any) -> Any:
        in_progress.discard(obj_id)
        memo[obj_id] = result
        return result

    if isinstance(obj, dict):
        replacements = {}
        for key, value in obj.items():
            new_value = _child(value, f"{path}[{key!r}]")
            if new_value is not value:
                replacements[key] = new_value
        if not replacements:
            return _record(obj)
        return _record({**obj, **replacements})

    if isinstance(obj, (list, tuple)):
        new_items = [_child(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        if all(new is old for new, old in zip(new_items, obj)):
            return _record(obj)
        return _record(
            type(obj)(new_items) if isinstance(obj, tuple) else new_items
        )

    attrs = getattr(obj, "__dict__", None)
    if not isinstance(attrs, dict):
        return _record(obj)

    replacements = {}
    for name, value in list(attrs.items()):
        new_value = _child(value, f"{path}.{name}")
        if new_value is not value:
            replacements[name] = new_value
    if not replacements:
        return _record(obj)
    # Shallow-copy before writing: this object may be shared with the live
    # request. The copy shares every unchanged child, so this is cheap even when
    # the chunk carries pixel tensors.
    #
    # If the type refuses to be copied, fall back to writing in place rather than
    # failing the whole stream. That is safe for the one field that matters:
    # ``_forward_vae_decode`` reads ``latents`` into a local before the decode
    # loop starts, and ``_iter_frame_slices`` closes over that local — so clearing
    # the descriptor off the echoed-back state cannot disturb an in-flight decode.
    try:
        target = copy.copy(obj)
    except Exception as exc:
        logger.warning(
            "[cmaf-trace] %s (%s) could not be copied for scrubbing (%s); "
            "clearing in place",
            path,
            type(obj).__name__,
            exc,
        )
        target = obj
    for name, value in replacements.items():
        try:
            setattr(target, name, value)
        except Exception as exc:
            # Left live deliberately: the serialize site logs the full object
            # graph on failure, which names this path.
            logger.warning(
                "[cmaf-trace] could not clear %s.%s while scrubbing (%s)",
                path,
                name,
                exc,
            )
    return _record(target)


def _strip_unserializable_prompt(chunk: Any) -> Any:
    """Scrub live block-stream descriptors off ``chunk``, logging what was found.

    Thin wrapper over :func:`_scrub_unserializable_refs` that keeps the call
    sites terse and reports the channel that carried the descriptor. The log line
    is the whole point of naming the paths: a remote run is expensive, so the one
    run that fixes this should also record *why* the earlier, field-specific strip
    was not enough.
    """
    found: list[str] = []
    cycles: list[str] = []
    scrubbed = _scrub_unserializable_refs(chunk, found=found, cycles=cycles)
    if found:
        logger.info(
            "[cmaf-trace] scrubbed live block-stream descriptor from %d location(s) "
            "before serialize: %s",
            len(found),
            ", ".join(found),
        )
    if cycles:
        # The serializer cannot encode a cycle either, so this is a real problem
        # in its own right and not merely a scrub limitation — name it here rather
        # than letting it surface as an opaque recursion or type error.
        logger.warning(
            "[cmaf-trace] reference cycle(s) on the streamed chunk, not scrubbed: %s",
            ", ".join(cycles),
        )
    return scrubbed


def _describe_object_graph(
    obj: Any, *, depth: int = 0, path: str = "chunk"
) -> list[str]:
    """Render ``path -> type`` for every node the serializer would visit.

    Used only when a serialize call has already failed. The serializer's
    ``TypeError`` names the offending *type* but not where it sits, which is
    exactly the fact needed to fix it — and a remote run is too expensive to
    spend on learning it twice.
    """
    if _is_scrub_leaf(obj) or depth >= _SCRUB_MAX_DEPTH:
        return [f"{path}: {type(obj).__name__}"]
    lines = [f"{path}: {type(obj).__name__}"]
    if isinstance(obj, dict):
        children = [(f"{path}[{k!r}]", v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple)):
        # Sequences here are pixel/frame lists; a few entries prove the shape.
        children = [(f"{path}[{i}]", v) for i, v in enumerate(obj[:4])]
    elif isinstance(getattr(obj, "__dict__", None), dict):
        children = [(f"{path}.{k}", v) for k, v in vars(obj).items()]
    else:
        return lines
    for child_path, value in children:
        lines.extend(_describe_object_graph(value, depth=depth + 1, path=child_path))
    return lines


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
    """Add default SHM connector edges for stage configs that omit them."""
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
    # A connector serializes too, so clear any live block-stream descriptor off the
    # echoed-back prompt for the same reason the SHM writes do. Done here, at the
    # one chokepoint every connector payload passes through, rather than at each
    # call site.
    engine_inputs = _strip_unserializable_prompt(engine_inputs)
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
    """Read the DiT latent tensor off a *single* stage output (per-chunk read).

    Mirrors ``dit2vae._latent_from_output``: the streamed DiT block is a diffusion
    ``OmniRequestOutput`` whose post-processed latent lands on the *top-level*
    ``multimodal_output['latent']`` (backed by ``_multimodal_output``) — the only
    channel the inter-stage connector preserves (``.images`` is dropped). Read
    that first, then any completion-output ``multimodal_output`` (the pipeline-
    stage shape), then ``.images[0]`` (the in-process no-connector channel).
    Returns None when no latent tensor is present so the caller can fall back.

    Reads one per-block tensor per call — this runs against the individual
    ``{rid}_c{n}`` chunks, before they are carried through as a list by
    ``_set_primary_latent`` (§4.2 Step 1).
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


def _set_primary_latent(
    engine_inputs: Any,
    latent: "torch.Tensor | list[torch.Tensor] | ShmLatentBlockStream",
) -> None:
    """Write the reassembled latent(s) back onto the first chunk's stage output.

    Sets whichever channel ``_primary_latent`` reads (top-level
    multimodal_output first, then a completion output's, else ``.images[0]``) so
    ``dit2vae`` sees the reassembled latent. ``latent`` is a **list of per-block
    tensors** (§4.2 Step 1): the VAE pipeline flattens and streams them one frame
    at a time into a persistent decode session, so no whole-clip ``torch.cat`` is
    materialized here. On the concurrent path (§4.2 Step 2) it is instead a
    ``ShmLatentBlockStream`` — a picklable descriptor that iterates blocks still
    being produced, which the pipeline consumes identically. (A bare tensor is
    still accepted for callers that hand one.)
    The guards test the *existing* value — the first chunk's per-block tensor — so
    they pass on the first (and only) write regardless of what ``latent`` is.
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
