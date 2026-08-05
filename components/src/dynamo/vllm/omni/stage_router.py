# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

from dynamo import prometheus_names
from dynamo.common.protocols.audio_protocol import NvAudioSpeechResponse
from dynamo.common.protocols.video_protocol import NvVideosResponse
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.common.utils.video_utils import StreamingCmafEncoder
from dynamo.llm import ModelInput, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.cmaf_video import (
    CMAF_ANNOTATION,
    CMAF_FALLBACK_VIDEO_CODEC,
    CMAF_INIT_TAG,
    CMAF_METADATA_TAG,
    CMAF_SEGMENT_PREFIX,
    cmaf_gop_frames,
    cmaf_segment_seconds,
    metadata_bytes,
)
from dynamo.vllm.omni.connectors import register_dynamoomni_nixl_connector
from dynamo.vllm.omni.output_formatter import OutputFormatter
from dynamo.vllm.omni.stage_worker import (
    _connector_key,
    _ensure_stage_connectors,
    _resolve_model_type,
    _restore_completion_output_attrs,
    _uses_nixl_connector,
)
from dynamo.vllm.omni.types import StageOutput
from dynamo.vllm.omni.utils import (
    ensure_awaited,
    is_empty_payload,
    resolve_stage_configs_compat,
    shm_deserialize,
    unwrap_connector_payload,
)
from dynamo.vllm.omni.video_convert import to_canonical

logger = logging.getLogger(__name__)


class OmniStageRouter:
    """Pure message broker for multi-stage omni pipelines."""

    def __init__(
        self,
        config: OmniConfig,
        stage_configs_path: str,
    ) -> None:
        self.config = config
        self.connectors: dict[tuple[str, str], Any] = {}
        (
            resolved_stage_configs_path,
            self.stage_configs,
        ) = resolve_stage_configs_compat(
            config.model,
            stage_configs_path,
            trust_remote_code=getattr(
                getattr(config, "engine_args", None), "trust_remote_code", False
            ),
        )
        self.stage_clients: Dict[str, Any] = {}

        # Initialize connectors so the router can fetch final-stage output
        # via connector.get() instead of SHM -- enabling multi-node deployments.
        connector_configs_path = _ensure_stage_connectors(
            resolved_stage_configs_path,
            self.stage_configs,
        )
        # Only register NixlConnector if it's actually used in stage configs
        if _uses_nixl_connector(connector_configs_path, self.stage_configs):
            try:
                register_dynamoomni_nixl_connector()
            except Exception as e:
                logger.error("Router: failed to register NixlConnector: %s", e)
                raise

        try:
            _, self.connectors = initialize_orchestrator_connectors(connector_configs_path)  # type: ignore[arg-type]
        except FileNotFoundError:
            logger.warning(
                "Router: connector config %s not found; continuing without connectors",
                connector_configs_path,
            )
            self.connectors = {}
        logger.info("Router: initialized %d connector(s)", len(self.connectors))

        media_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._formatter = OutputFormatter(
            model_name=config.served_model_name or config.model,
            media_fs=media_fs,
            media_http_url=config.media_output_http_url,
            default_fps=config.default_video_fps,
        )

    def set_stage_client(self, model_stage: str, client: Any) -> None:
        self.stage_clients[model_stage] = client
        logger.info("Registered stage client: %s", model_stage)

    def _error_envelope(
        self,
        request_id: str,
        message: str,
        request_type: Any = None,
    ) -> Dict[str, Any]:
        """Build a client-visible error response in the shape the route expects.

        A bare ``{"error": ..., "finished": True}`` dict is the *worker -> router*
        error contract (``StageOutput``), not the *router -> frontend* one. The
        frontend deserializes this yield into the modality's response type, and
        for video that is ``NvVideosResponse``, whose ``id``/``model``/``created``
        fields have no serde defaults. A bare dict therefore fails to deserialize
        and the real message is replaced by ``missing field `id```, so every
        distinct failure on this endpoint reports the same useless string. Emit
        the full envelope instead: the Rust side already turns ``status ==
        "failed"`` into a proper error (an ``0x04`` frame on the binary CMAF
        route), so the message reaches the client verbatim.

        ``finished`` is kept for the batch paths, where the router's own reply
        loop still reads it; the response models ignore unknown keys.

        ``request_type`` is compared by *value* rather than by identity: a caller
        holding the enum and one holding the plain string must select the same
        envelope, because picking the wrong modality here silently reintroduces
        exactly the deserialization failure this method exists to prevent.
        """
        model_name = self.config.served_model_name or self.config.model
        created = int(time.time())
        rt = getattr(request_type, "value", request_type)
        if rt == RequestType.VIDEO_GENERATION.value:
            envelope = NvVideosResponse(
                id=request_id,
                object="video",
                model=model_name,
                status="failed",
                progress=0,
                created=created,
                data=[],
                error=message,
            ).model_dump()
        elif rt == RequestType.AUDIO_GENERATION.value:
            envelope = NvAudioSpeechResponse(
                id=request_id,
                model=model_name,
                status="failed",
                created=created,
                error=message,
            ).model_dump()
        else:
            # Chat and image generation share the chat.completion.chunk error
            # shape used by OutputFormatter._error_chunk and BaseHandler.
            envelope = {
                "id": request_id,
                "created": created,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": f"Error: {message}",
                        },
                        "finish_reason": "error",
                    }
                ],
                "error": message,
            }
        envelope["finished"] = True
        return envelope

    async def generate(
        self,
        request: dict,
        context,  # noqa: ARG002 — context unused; router generates its own request_id
    ) -> AsyncGenerator[dict, None]:
        request_id = str(uuid.uuid4())
        _, request_type = parse_request_type(request, self.config.output_modalities)

        # Binary CMAF live streaming: when the request is CMAF-annotated video,
        # the final (VAE) stage streams pixel chunks and the router pumps each
        # straight into a persistent CMAF encoder (work-item c). Detect it up
        # front so we can take the live path instead of the batch drain below.
        nvext_early = request.get("nvext") or {}
        annotations_early = (
            (nvext_early.get("annotations") or []) if isinstance(nvext_early, dict) else []
        )
        cmaf_live = (
            request_type == RequestType.VIDEO_GENERATION
            and CMAF_ANNOTATION in annotations_early
        )
        if cmaf_live:
            async for chunk in self._stream_cmaf_live(request, request_id, request_type):
                yield chunk
            return

        stage_outputs: List[StageOutput] = []
        for stage_idx, stage_cfg in enumerate(self.stage_configs):
            model_stage = getattr(
                stage_cfg.engine_args, "model_stage", f"stage{stage_idx}"
            )
            client = self.stage_clients.get(model_stage)
            if client is None:
                yield self._error_envelope(
                    request_id, f"No client for stage '{model_stage}'", request_type
                )
                return

            if stage_idx == 0:
                # This is a workaround for now to pass in the raw request to stage 0. StageRequest validates it but ignores any unknown keys, so it gets passed through.
                stage_request = {"request_id": request_id, **request}
            else:
                stage_request = stage_outputs[-1].to_next_stage_request(request_id)

            raw_stage_output = {}
            # A streamed final stage (VAE with streaming_output on) delivers pixel
            # chunks as multiple per-chunk-SHM yields; collect their refs so the
            # batch path can reassemble the full clip. Non-final streamed stages
            # (DiT) end with a chunked connector ref the next stage reassembles,
            # so their intermediate control signals are simply ignored here.
            pixel_chunk_metas: list[dict] = []
            logger.info(
                "Router: stage %d request keys=%s",
                stage_idx,
                list(stage_request.keys()),
            )
            async for chunk in await client.round_robin(stage_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                if not data.get("finished") and data.get("shm_meta") is not None:
                    pixel_chunk_metas.append(data["shm_meta"])
                    continue
                raw_stage_output.update(data)
            stage_outputs.append(StageOutput.model_validate(raw_stage_output))

            if stage_outputs[-1].error:
                yield self._error_envelope(
                    request_id, stage_outputs[-1].error, request_type
                )
                return

        final = stage_outputs[-1]
        # Streamed final-stage pixels: reassemble the per-chunk SHM outputs into
        # one full-clip result and format it directly (bypassing the single-
        # shm_meta read in _format_output, which a streamed stage never sets).
        if pixel_chunk_metas:
            result = _reassemble_pixel_chunks(pixel_chunk_metas)
            fmt_ctx_stream: Dict[str, Any] = {}
            nvext_s = request.get("nvext") or {}
            if nvext_s.get("fps") is not None:
                fmt_ctx_stream["fps"] = nvext_s["fps"]
            if request.get("response_format") is not None:
                fmt_ctx_stream["response_format"] = request["response_format"]
            if request.get("output_format") is not None:
                fmt_ctx_stream["output_format"] = request["output_format"]
            chunk = await self._formatter.format(
                result, request_id, request_type=request_type, **fmt_ctx_stream
            )
            if chunk:
                yield chunk
            else:
                yield self._error_envelope(
                    request_id,
                    "Formatter returned no output for streamed clip",
                    request_type,
                )
            return
        connectors = getattr(self, "connectors", {})
        # Accept either connector-based output (multi-node) or SHM (single-node legacy).
        # Connector path: final stage wrote via connector.put(to_stage="router") and
        # returned stage_connector_refs[last_stage_id] = metadata.
        final_stage_id = self.stage_configs[-1].stage_id
        has_connector_output = (
            final.stage_connector_refs is not None
            and str(final_stage_id) in final.stage_connector_refs
            and connectors.get(_connector_key(final_stage_id, "router")) is not None
        )
        if not has_connector_output and not final.shm_meta:
            error_msg = (
                "No output from final stage (no connector ref and no SHM)"
                if connectors
                else "No SHM output from final stage"
            )
            yield self._error_envelope(request_id, error_msg, request_type)
            return

        # Build formatting context from the original request
        nvext = request.get("nvext") or {}
        fmt_ctx: Dict[str, Any] = {}
        if nvext.get("fps") is not None:
            fmt_ctx["fps"] = nvext["fps"]
        if nvext.get("speed") is not None:
            fmt_ctx["speed"] = nvext["speed"]
        # If the request type is AUDIO_GENERATION,
        # we need to normalize the data_source and response_format to
        # align with other modalities.
        response_format = (
            request.get("data_source")
            if request_type == RequestType.AUDIO_GENERATION
            else request.get("response_format")
        )
        output_format = (
            request.get("response_format")
            if request_type == RequestType.AUDIO_GENERATION
            else request.get("output_format")
        )
        if response_format is not None:
            fmt_ctx["response_format"] = response_format
        if output_format is not None:
            fmt_ctx["output_format"] = output_format

        # Binary CMAF streaming: the frontend's /v1/videos/stream/binary/cmaf
        # route injects the experimental_binary_cmaf annotation. When present on
        # a video request, fragment the finished clip and stream CMAF pieces
        # instead of a single full-video response. ``nvext`` here is the raw
        # request dict, not a Pydantic model, so check the annotation directly.
        annotations = (nvext.get("annotations") or []) if isinstance(nvext, dict) else []
        cmaf_enabled = (
            request_type == RequestType.VIDEO_GENERATION
            and CMAF_ANNOTATION in annotations
        )

        async for chunk in self._format_output(
            final,
            request_id,
            request_type,
            fmt_ctx,
            final_stage_id=self.stage_configs[-1].stage_id,
            cmaf_enabled=cmaf_enabled,
        ):
            yield chunk

    async def _stream_cmaf_live(
        self,
        request: dict,
        request_id: str,
        request_type: RequestType,
    ) -> AsyncGenerator[dict, None]:
        """Live per-chunk CMAF pump (work-item c).

        DiT streams latent blocks to the connector (work-item b) and RPC-yields a
        terminal ref; the router drives it to completion exactly like the batch
        path (the intermediate control signals carry no data). The VAE stage then
        reassembles those latents and streams **pixel chunks** — one decoded frame
        at a time, feat_cache persisted so the stream is seam-free (§7.0) — each
        delivered via per-chunk SHM. The router deserializes each pixel chunk and
        pushes it straight into a single persistent CMAF encoder, yielding
        metadata/init/segment items on the same wire contract as the batch path
        (§5). Only the *source* of the tagged items changes: streamed, not batched.
        """
        # Two stages exactly: DiT (0) then VAE (1). This live path is registered
        # only for the disaggregated causal-forcing video pipeline.
        if len(self.stage_configs) < 2:
            yield self._error_envelope(
                request_id,
                "CMAF live streaming requires a 2-stage pipeline",
                request_type,
            )
            return

        dit_cfg, vae_cfg = self.stage_configs[0], self.stage_configs[1]
        dit_stage = getattr(dit_cfg.engine_args, "model_stage", "stage0")
        vae_stage = getattr(vae_cfg.engine_args, "model_stage", "stage1")
        dit_client = self.stage_clients.get(dit_stage)
        vae_client = self.stage_clients.get(vae_stage)
        if dit_client is None or vae_client is None:
            yield self._error_envelope(
                request_id,
                "CMAF live streaming: missing DiT or VAE stage client",
                request_type,
            )
            return

        # --- Stage 0 (DiT): drive in the background; do NOT wait for completion. ---
        # [cmaf-step2] The VAE is dispatched as soon as the DiT's *first* latent
        # block is on the connector, so pixel decode overlaps the remaining
        # rollout. The VAE decode is temporally stateful (feat_cache), so all
        # latent frames must still pass through ONE persistent forward — that
        # invariant is preserved inside the VAE worker, which feeds blocks into a
        # single decode session as they arrive (§4.2 Step 2), rather than by the
        # router serializing the two stages. The [cmaf-timing] logs make the
        # first-block -> VAE-start handoff and time-to-first-frame explicit.
        t0 = time.monotonic()
        logger.info("[cmaf-timing] router: DiT started for %s", request_id)

        dit_error: dict = {}
        dit_blocks_seen = 0
        # Overlap timeline, all relative to t0. dit_ended_at stays None while the
        # rollout is still running, which is itself the healthy signal.
        dit_ended_at: float | None = None
        first_pixel_at: float | None = None
        first_ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def _drive_dit() -> None:
            """Consume the DiT ready-signal stream to completion (no tensors here)."""
            nonlocal dit_blocks_seen, dit_ended_at
            try:
                async for chunk in await dit_client.round_robin(
                    {"request_id": request_id, **request}
                ):
                    data = chunk.data()
                    if isinstance(data, (str, bytes)):
                        data = json.loads(data)
                    if data.get("error"):
                        dit_error["error"] = data["error"]
                        logger.error(
                            "[cmaf-step2] router: DiT reported an error for %s after "
                            "%d blocks at +%.2fs: %s",
                            request_id, dit_blocks_seen, time.monotonic() - t0,
                            data["error"],
                        )
                        break
                    # The first ready signal carries the request facts the VAE
                    # needs and means block c0 is on the connector.
                    if data.get("chunk_index") is not None and not data.get("finished"):
                        dit_blocks_seen += 1
                    if not first_ready.done() and data.get("chunk_index") == 0:
                        if data.get("original_prompt") is None:
                            # The VAE would fall back to bare geometry defaults;
                            # worth flagging since it points at a version skew
                            # between router and DiT worker.
                            logger.warning(
                                "[cmaf-step2] router: DiT first ready signal for %s "
                                "carries no original_prompt — is the DiT worker running "
                                "the Step 2 code?",
                                request_id,
                            )
                        first_ready.set_result(data)
            except Exception as e:
                logger.error(
                    "Router: DiT stream failed for %s: %s", request_id, e, exc_info=True
                )
                dit_error["error"] = f"DiT stage failed: {e}"
            finally:
                dit_ended_at = time.monotonic() - t0
                logger.info(
                    "[cmaf-timing] router: DiT stream ended for %s at +%.2fs "
                    "(%d latent blocks streamed)",
                    request_id, dit_ended_at, dit_blocks_seen,
                )
                if not first_ready.done():
                    # DiT produced no block: unblock the waiter with the reason.
                    # Most likely the DiT ran in aggregated (non-streaming) mode, so
                    # say that rather than just "no blocks".
                    reason = dit_error.get("error") or (
                        "DiT stage produced no latent blocks — it likely ran "
                        "aggregated instead of block-streaming (check stream_dit_blocks)"
                    )
                    first_ready.set_exception(RuntimeError(reason))

        dit_task = asyncio.ensure_future(_drive_dit())

        try:
            dit_first = await first_ready
        except RuntimeError as e:
            dit_task.cancel()
            yield self._error_envelope(request_id, str(e), request_type)
            return
        first_block_at = time.monotonic() - t0
        logger.info(
            "[cmaf-step2] router: DiT first block ready for %s at +%.2fs; "
            "dispatching VAE concurrently (DiT %s)",
            request_id,
            first_block_at,
            "already finished — expect SERIALIZED"
            if dit_ended_at is not None
            else "rollout still running",
        )

        # --- Stage 1 (VAE): consume the live pixel-chunk stream. ---
        fps = int((request.get("nvext") or {}).get("fps") or self.config.default_video_fps)
        # [cmaf-step2] Synthesize the chunked upstream ref rather than taking it
        # from the DiT terminal (which no longer exists at this point). No
        # num_chunks — it is unknown while the rollout runs; the VAE terminates on
        # the stream-end marker instead. Geometry comes from original_prompt, which
        # the real VAE decode does not read anyway (it derives it from the latents).
        vae_request = {
            "request_id": request_id,
            "stage_connector_refs": {str(dit_cfg.stage_id): {"chunked": True}},
        }
        if (op := dit_first.get("original_prompt")) is not None:
            vae_request["original_prompt"] = op
        if (spl := dit_first.get("sampling_params_list")) is not None:
            vae_request["sampling_params_list"] = spl
        created = int(time.time())
        t_vae = time.monotonic()
        pixel_chunks_seen = 0
        enc: StreamingCmafEncoder | None = None
        metadata_sent = False
        seg_index = 0

        # _cmaf_chunk lives on the DiffusionFormatter ("image"), not the
        # dispatcher OutputFormatter; reach it the same way stream_video_cmaf does.
        diffusion_formatter = self._formatter._formatters["image"]

        async def _emit(kind: str, payload: bytes) -> AsyncGenerator[dict, None]:
            nonlocal metadata_sent, seg_index
            if kind == "init":
                if not metadata_sent:
                    yield diffusion_formatter._cmaf_chunk(
                        request_id,
                        created,
                        CMAF_METADATA_TAG,
                        metadata_bytes(
                            None,
                            cmaf_segment_seconds(),
                            (enc.codec_string() if enc else None) or CMAF_FALLBACK_VIDEO_CODEC,
                        ),
                        progress=0,
                    )
                    metadata_sent = True
                yield diffusion_formatter._cmaf_chunk(request_id, created, CMAF_INIT_TAG, payload, progress=1)
            else:
                yield diffusion_formatter._cmaf_chunk(
                    request_id,
                    created,
                    f"{CMAF_SEGMENT_PREFIX}{seg_index}",
                    payload,
                    progress=min(99, 2 + seg_index),
                )
                seg_index += 1

        try:
            vae_replies = 0
            async for chunk in await vae_client.round_robin(vae_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                vae_replies += 1
                # [cmaf-trace] Every VAE reply as the router sees it. The break
                # below is driven purely by these two fields, so record them for
                # each reply: it is the only way to tell "the VAE stopped sending"
                # from "the router stopped listening".
                logger.info(
                    "[cmaf-trace] router: VAE reply #%d for %s — finished=%s "
                    "has_shm_meta=%s chunk_index=%s error=%s",
                    vae_replies - 1, request_id,
                    data.get("finished"),
                    data.get("shm_meta") is not None,
                    data.get("chunk_index"),
                    data.get("error"),
                )
                if data.get("error"):
                    yield self._error_envelope(
                        request_id, data["error"], request_type
                    )
                    return
                # Terminal sentinel: no pixels — the tail is drained by finish().
                if data.get("finished") and data.get("shm_meta") is None:
                    logger.info(
                        "[cmaf-trace] router: terminal sentinel for %s after %d "
                        "pixel chunks (reply #%d) — leaving the VAE read loop",
                        request_id, pixel_chunks_seen, vae_replies - 1,
                    )
                    break
                shm_meta = data.get("shm_meta")
                if shm_meta is None:
                    continue
                pixel_output = shm_deserialize(shm_meta)
                images = getattr(pixel_output, "images", pixel_output)
                if is_empty_payload(images):
                    continue
                canonical = to_canonical(images)
                pixel_chunks_seen += 1
                if first_pixel_at is None:
                    first_pixel_at = time.monotonic() - t0
                    # Time-to-first-frame is the headline number Step 2 moves, and
                    # whether DiT was still running when it landed is the proof.
                    logger.info(
                        "[cmaf-step2] router: FIRST PIXEL CHUNK for %s at +%.2fs "
                        "(DiT %s) — this is time-to-first-frame",
                        request_id,
                        first_pixel_at,
                        "STILL RUNNING => overlapped"
                        if dit_ended_at is None
                        else f"already ended at +{dit_ended_at:.2f}s => serialized",
                    )
                logger.info(
                    "[cmaf-timing] router: VAE pixel chunk %d received for %s at +%.2fs (VAE start->here)",
                    pixel_chunks_seen - 1, request_id, time.monotonic() - t_vae,
                )
                if enc is None:
                    height, width = int(canonical.shape[1]), int(canonical.shape[2])
                    enc = StreamingCmafEncoder(fps, width, height, gop_frames=cmaf_gop_frames())
                    await enc.start()
                async for kind, payload in enc.push(canonical):
                    async for item in _emit(kind, payload):
                        yield item

            if enc is not None:
                async for kind, payload in enc.finish():
                    async for item in _emit(kind, payload):
                        yield item
            elif pixel_chunks_seen == 0:
                # The VAE stream ended without a single pixel chunk, so nothing
                # was ever yielded. Returning silently here leaves the request
                # with no response at all, which the frontend reports only as an
                # empty stream. Yield an explicit failed envelope so the real
                # condition reaches the client.
                logger.error(
                    "[cmaf-trace] router: VAE produced ZERO pixel chunks for %s "
                    "(%d replies, DiT %s) — emitting an explicit error instead of "
                    "an empty stream",
                    request_id, vae_replies,
                    f"ended at +{dit_ended_at:.2f}s" if dit_ended_at is not None
                    else "still running",
                )
                yield self._error_envelope(
                    request_id,
                    "CMAF live stream produced no pixel chunks: the VAE stage "
                    f"sent {vae_replies} replies but no pixel data. The VAE "
                    "terminated before decoding any frame — check the "
                    "[cmaf-trace] lines in the VAE worker log.",
                    request_type,
                )
                return
        except Exception as e:
            logger.error("Router: CMAF live stream failed for %s: %s", request_id, e, exc_info=True)
            yield self._error_envelope(
                request_id, f"CMAF live stream failed: {e}", request_type
            )
            return
        finally:
            # [cmaf-step2] The DiT drive runs concurrently, so reap it on every
            # exit path — including an early return or a client disconnect that
            # closes this generator — rather than leaving an orphan task behind.
            if not dit_task.done():
                dit_task.cancel()
            try:
                await dit_task
            except (asyncio.CancelledError, Exception):
                pass
        # A DiT failure after the first block still means a truncated video; the
        # pixel stream just ends early, so surface it rather than ending cleanly.
        if dit_error:
            logger.error(
                "Router: DiT stage reported an error for %s after streaming began: %s",
                request_id, dit_error["error"],
            )
            yield self._error_envelope(request_id, dit_error["error"], request_type)
            return
        # The overlap verdict, stated outright: this is what the whole step is for,
        # and reading it off interleaved timestamps by hand is error-prone. If the
        # DiT stream ended before the first pixel chunk arrived, the stages ran
        # serialized and Step 2 is not actually in effect.
        overlapped = dit_ended_at is None or (
            first_pixel_at is not None and first_pixel_at < dit_ended_at
        )
        logger.info(
            "[cmaf-step2] router: live stream complete for %s — %d pixel chunks in "
            "%.2fs; first block +%.2fs, first pixel %s, DiT end %s => %s",
            request_id,
            pixel_chunks_seen,
            time.monotonic() - t0,
            first_block_at,
            f"+{first_pixel_at:.2f}s" if first_pixel_at is not None else "never",
            f"+{dit_ended_at:.2f}s" if dit_ended_at is not None else "still running",
            "OVERLAPPED (Step 2 working)"
            if overlapped
            else "SERIALIZED — decode did not overlap the rollout",
        )
        if not overlapped:
            logger.warning(
                "[cmaf-step2] router: stages ran serialized for %s. The VAE waited for "
                "the full DiT rollout, so time-to-first-frame is unimproved. Likely "
                "causes: something materialized the block stream (a list()/len() on it), "
                "or the VAE fetched with a known num_chunks instead of polling.",
                request_id,
            )
        # Natural end: the Rust route emits the DONE(0x05) frame when this
        # generator closes (§9). Success items are modality-fixed by the CMAF wire
        # contract; request_type only shapes the error envelopes above.

    async def _format_output(
        self,
        stage_output: StageOutput,
        request_id: str,
        request_type: RequestType,
        ctx: dict,
        final_stage_id: int = 0,
        cmaf_enabled: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """Read OmniRequestOutput from connector (multi-node) or SHM (single-node) and format."""
        # --- Connector path (multi-node: router and final stage on different machines) ---
        router_connector = getattr(self, "connectors", {}).get(
            _connector_key(final_stage_id, "router")
        )
        meta_k = (
            stage_output.stage_connector_refs.get(str(final_stage_id))
            if stage_output.stage_connector_refs
            else None
        )
        if router_connector is not None and meta_k is not None:
            try:
                get_result = await ensure_awaited(
                    router_connector.get(
                        str(final_stage_id),
                        "router",
                        request_id,
                        metadata=meta_k,
                    )
                )
                payload_data = unwrap_connector_payload(get_result)
                if is_empty_payload(payload_data):
                    raise RuntimeError("empty payload returned by connector.get()")

                if isinstance(payload_data, dict) and "engine_inputs" in payload_data:
                    result = payload_data["engine_inputs"]
                    _restore_completion_output_attrs(
                        result,
                        payload_data.get("_dynamo_completion_output_attrs"),
                    )
                else:
                    result = payload_data
                logger.debug(
                    "Router: fetched final output via connector for %s", request_id
                )
            except Exception as e:
                logger.error("Router: connector.get() failed for %s: %s", request_id, e)
                yield self._error_envelope(
                    request_id, f"Router connector.get() failed: {e}", request_type
                )
                return
        else:
            # --- SHM fallback (single-node: router and final stage on same machine) ---
            shm_meta = stage_output.shm_meta
            if not shm_meta:
                # Unreachable via generate() (the caller already rejects a final
                # stage with neither a connector ref nor SHM), but returning
                # silently here would end the request with no response at all and
                # the client would see only an empty stream. Say what happened.
                logger.error("Router: no shm_meta in final stage output")
                yield self._error_envelope(
                    request_id,
                    "No output from final stage: neither a router connector ref "
                    "nor an SHM handle was returned",
                    request_type,
                )
                return
            result = shm_deserialize(shm_meta)

        # CMAF streaming path: encode + fragment the finished clip and stream
        # metadata/init/segment pieces instead of a single full-video response.
        if cmaf_enabled:
            fps = ctx.get("fps", self.config.default_video_fps)
            async for chunk in self._formatter.stream_video_cmaf(
                result, request_id, fps=fps
            ):
                yield chunk
            return

        chunk = await self._formatter.format(
            result, request_id, request_type=request_type, **ctx
        )
        if chunk:
            yield chunk
        else:
            final_output_type = getattr(result, "final_output_type", "unknown")
            logger.warning(
                "Router: formatter returned None, final_output_type=%s",
                final_output_type,
            )
            yield self._error_envelope(
                request_id,
                f"Formatter returned no output for type '{final_output_type}'",
                request_type,
            )


def _reassemble_pixel_chunks(pixel_chunk_metas: list[dict]) -> Any:
    """Concatenate streamed per-chunk pixel outputs into one full-clip result.

    Each ref points to an ``OmniRequestOutput`` carrying one decoded chunk's
    pixels on ``.images``. Read them in order, canonicalize to ``(t, H, W, 3)``
    uint8, concatenate along time, and hand the combined clip back on the first
    output object so the formatter treats it exactly like a batch result.
    """
    import numpy as np

    outputs = [shm_deserialize(m) for m in pixel_chunk_metas]
    frames = [
        to_canonical(getattr(o, "images", o))
        for o in outputs
        if not is_empty_payload(getattr(o, "images", o))
    ]
    if not frames:
        return outputs[0]
    combined = np.concatenate(frames, axis=0)  # (T, H, W, 3)
    result = outputs[0]
    if hasattr(result, "images"):
        result.images = [combined]
        return result
    return [combined]


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config: OmniConfig,
    shutdown_endpoints: list,
) -> None:
    """Initialize OmniStageRouter as a Dynamo backend endpoint."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint or 'generate'}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    router = OmniStageRouter(config, config.stage_configs_path)  # type: ignore[arg-type]

    setup_metrics_collection(config, generate_endpoint, logger)

    # Discover stage endpoints
    for stage_cfg in router.stage_configs:
        model_stage = getattr(
            stage_cfg.engine_args, "model_stage", f"stage{stage_cfg.stage_id}"
        )
        client = await runtime.endpoint(
            f"{config.namespace}.{model_stage}.generate"
        ).client()
        await client.wait_for_instances()
        router.set_stage_client(model_stage, client)

    final_cfg = router.stage_configs[-1]
    final_output_type = getattr(final_cfg, "final_output_type", "image")
    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        model_type = _resolve_model_type(final_output_type)

    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        # OmniStageRouter is the user-visible front for an internal
        # multi-stage pipeline; the per-stage workers are private. From
        # the frontend's topology view, the router serves end-to-end as
        # Aggregated with no peer dependencies.
        worker_type=WorkerType.Aggregated,
        needs=[],
    )
    logger.info("OmniStageRouter registered at '%s'", generate_endpoint)

    try:
        await generate_endpoint.serve_endpoint(
            router.generate,
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
        )
    except Exception as e:
        logger.error("OmniStageRouter endpoint failed: %s", e)
        raise
