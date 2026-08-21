# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage router for disaggregated omni pipelines."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

import numpy as np
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

from dynamo import prometheus_names
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import (
    RequestType,
    get_output_modalities,
    parse_request_type,
)
from dynamo.common.utils.video_utils import StreamingCmafEncoder, compute_num_frames
from dynamo.llm import ModelInput, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.vllm.main import setup_metrics_collection
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.omni.cf_pipeline import FrameQueue, Relay, cf_put_key
from dynamo.vllm.omni.cf_session import (
    CFSessionRequestError,
    has_cf_session,
    parse_cf_session,
)
from dynamo.vllm.omni.cmaf_video import (
    CMAF_ANNOTATION,
    CMAF_FALLBACK_VIDEO_CODEC,
    cmaf_gop_frames,
    resolve_cmaf_audio_file,
)
from dynamo.vllm.omni.connectors import register_dynamoomni_nixl_connector
from dynamo.vllm.omni.output_formatter import AudioTrack, OutputFormatter
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
from dynamo.vllm.omni.video_convert import decoded_chunk_to_canonical, to_canonical

logger = logging.getLogger(__name__)

# Fallback audio length for a session (Causal-Forcing) scene that did not state
# `latents`, leaving its clip length known only to the rollout. An upper bound,
# not an estimate: the encoder is lazy so unpulled seconds are never produced,
# and the A/V merge cuts audio at the video's end -- whereas asking for too
# little would end the audio track early, which stalls playback in MSE.
_CF_AUDIO_UPPER_BOUND_S = 3600.0


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

    async def generate(
        self,
        request: dict,
        context,  # noqa: ARG002 — context unused; router generates its own request_id
    ) -> AsyncGenerator[dict, None]:
        request_id = str(uuid.uuid4())

        # Causal-Forcing session streaming, dispatched before request parsing: the
        # streaming path never consults request_type -- a session is video by
        # construction -- so parsing first would only add a way for it to fail.
        if has_cf_session(request.get("nvext")):
            # Closed explicitly rather than left to ``async for``: when the client
            # disconnects, closing *this* generator does not close the one it is
            # delegating to -- an async generator's cleanup otherwise waits for the
            # event loop's finalizer, and until it runs the stream's two producer
            # tasks are still alive with nobody draining them.
            cf_stream = self._generate_cf_stream(request, request_id)
            try:
                async for chunk in cf_stream:
                    yield chunk
            finally:
                await cf_stream.aclose()
            return

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
                yield {
                    "error": f"No client for stage '{model_stage}'",
                    "finished": True,
                }
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
                yield {"error": stage_outputs[-1].error, "finished": True}
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
                yield {"error": "Formatter returned no output for streamed clip", "finished": True}
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
            yield {"error": error_msg, "finished": True}
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
            yield {"error": "CMAF live streaming requires a 2-stage pipeline", "finished": True}
            return

        dit_cfg, vae_cfg = self.stage_configs[0], self.stage_configs[1]
        dit_stage = getattr(dit_cfg.engine_args, "model_stage", "stage0")
        vae_stage = getattr(vae_cfg.engine_args, "model_stage", "stage1")
        dit_client = self.stage_clients.get(dit_stage)
        vae_client = self.stage_clients.get(vae_stage)
        if dit_client is None or vae_client is None:
            yield {"error": "CMAF live streaming: missing DiT or VAE stage client", "finished": True}
            return

        # --- Stage 0 (DiT): drive to completion, collect the terminal ref. ---
        # NOTE: the VAE is dispatched only AFTER the DiT rollout finishes (not
        # after the first latent block): the VAE decode is temporally stateful
        # (feat_cache), so all latent frames must pass through one persistent
        # forward. The [cmaf-timing] logs make the DiT-done -> VAE-start handoff
        # and time-to-first-frame explicit so a long startup can be attributed
        # to the DiT rollout rather than a handoff stall.
        t0 = time.monotonic()
        logger.info("[cmaf-timing] router: DiT started for %s", request_id)
        dit_raw: dict = {}
        async for chunk in await dit_client.round_robin({"request_id": request_id, **request}):
            data = chunk.data()
            if isinstance(data, (str, bytes)):
                data = json.loads(data)
            dit_raw.update(data)
        dit_output = StageOutput.model_validate(dit_raw)
        if dit_output.error:
            yield {"error": dit_output.error, "finished": True}
            return
        logger.info(
            "[cmaf-timing] router: DiT done for %s in %.2fs; dispatching VAE",
            request_id, time.monotonic() - t0,
        )

        # --- Stage 1 (VAE): consume the live pixel-chunk stream. ---
        fps = int((request.get("nvext") or {}).get("fps") or self.config.default_video_fps)
        vae_request = dit_output.to_next_stage_request(request_id)
        t_vae = time.monotonic()

        async def _video_pieces() -> AsyncGenerator[tuple[str, bytes, str], None]:
            """The VAE pixel stream as pull-based CMAF video pieces.

            An async generator, so it advances only when the merge below asks for
            the next piece -- which is what lets audio be interleaved against it.
            The pull reaches all the way back to ``round_robin``, so back-pressure
            is unchanged from the push-based version this replaces.

            Errors are raised rather than turned into router error dicts: by the
            time one can happen the client already holds init and some segments,
            and the merge reports that as a CMAF failure frame -- a partial asset
            declared partial, instead of a stream that just stops.
            """
            pixel_chunks_seen = 0
            enc: StreamingCmafEncoder | None = None
            async for chunk in await vae_client.round_robin(vae_request):
                data = chunk.data()
                if isinstance(data, (str, bytes)):
                    data = json.loads(data)
                if data.get("error"):
                    raise RuntimeError(f"VAE stage: {data['error']}")
                # Terminal sentinel: no pixels — the tail is drained by finish().
                if data.get("finished") and data.get("shm_meta") is None:
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
                logger.info(
                    "[cmaf-timing] router: VAE pixel chunk %d received for %s at +%.2fs (VAE start->here)",
                    pixel_chunks_seen - 1, request_id, time.monotonic() - t_vae,
                )
                if enc is None:
                    height, width = int(canonical.shape[1]), int(canonical.shape[2])
                    enc = StreamingCmafEncoder(fps, width, height, gop_frames=cmaf_gop_frames())
                    await enc.start()
                async for kind, payload in enc.push(canonical):
                    yield (kind, payload, enc.codec_string() or CMAF_FALLBACK_VIDEO_CODEC)

            if enc is not None:
                async for kind, payload in enc.finish():
                    yield (kind, payload, enc.codec_string() or CMAF_FALLBACK_VIDEO_CODEC)

        # One emit path whether or not there is audio: an audio-less request is a
        # request whose AudioTrack is disabled (source None), which the merge
        # reports honestly as `has_audio: false` and then never polls.
        # Segment cadence here is the encoder GOP, not DYN_CMAF_SEGMENT_SECONDS:
        # this path fragments every `cmaf_gop_frames()` frames (0.25s at 4 @ 16fps),
        # so that -- not the 2s the fragment-generator path uses -- is what the
        # audio watermark has to be measured against.
        async for item in self._formatter.merge_av_cmaf(
            _video_pieces(),
            request_id,
            audio=self._audio_track(request, fps),
            video_segment_seconds=cmaf_gop_frames() / max(1, fps),
            video_duration_s=self._video_duration_s(request, fps),
        ):
            yield item
        # Natural end: the Rust route emits the DONE(0x05) frame when this
        # generator closes (§9). request_type is accepted for symmetry with the
        # batch path; the CMAF wire contract is modality-fixed.
        _ = request_type

    def _audio_track(
        self,
        request: dict,
        fps: int,
        *,
        duration_s: float | None = None,
        start_s: float = 0.0,
    ) -> AudioTrack:
        """Build the request's audio track, disabled when no source is configured.

        POC sourcing: a file on the *router host* (``CF_POC_AUDIO_FILE``, else the
        example's bundled mp3), stood up here rather than in a worker so the shape
        downstream -- a lazy fragment generator merged against video -- is already
        the one a real audio worker will plug into.

        Duration is slaved to the video's. ``duration_s`` overrides the value
        derived from the request, for a caller that cannot know the clip length up
        front; over-asking is free, because the encoder is lazy and the merge cuts
        audio at the video's real end. Under-asking is not: the audio track would
        end early and MSE stalls playback wherever an active track runs dry.

        ``start_s`` is where in the source to begin, so a chained scene continues
        the bed instead of restarting it. See ``cf_scene.audio_offset``.
        """
        if duration_s is None:
            duration_s = self._video_duration_s(request, fps)
        return AudioTrack(resolve_cmaf_audio_file(), duration_s, start_s=start_s)

    def _cf_audio_bounds(
        self, request: dict, fps: int
    ) -> tuple[float | None, float]:
        """``(scene_duration_s, audio_start_s)`` for a session request.

        The duration is None when the scene did not state ``latents``, meaning
        only the pipeline's default knows the length and the caller has to fall
        back to an upper bound.

        The offset is how far into the bed this scene starts, which is what makes
        a storyboard sound like one continuous piece of music rather than the same
        few seconds restarted per scene. The client is the component that knows
        it, since it owns the presentation timeline; 0 when it did not say.

        Malformed annotations are not this method's problem to report -- the DiT
        worker parses the same annotations and fails the request properly. Audio
        degrades to the video-only-ish default instead of turning a render into a
        traceback from the audio-sizing path.
        """
        try:
            session = parse_cf_session(request.get("nvext"))
        except CFSessionRequestError as e:
            logger.warning("Ignoring unparseable cf_scene for audio sizing: %s", e)
            return None, 0.0
        if session is None:
            return None, 0.0
        num_frames = session.scene.num_frames()
        duration_s = None if num_frames is None else num_frames / max(1, fps)
        return duration_s, session.scene.audio_offset or 0.0

    def _video_duration_s(self, request: dict, fps: int) -> float:
        """The clip's length in seconds, as the request states it.

        One place, because two callers must agree: the audio track is encoded to
        this length and the A/V merge cuts audio at it. Deriving them separately
        would let them drift, and the failure mode is silent -- audio a little
        long is wasted encoding, audio a little short stalls MSE playback.
        """
        nvext = request.get("nvext") or {}
        return (
            compute_num_frames(
                num_frames=nvext.get("num_frames"),
                seconds=nvext.get("seconds"),
                fps=nvext.get("fps"),
                default_fps=self.config.default_video_fps,
            )
            / max(1, fps)
        )
    # -- Causal-Forcing pipelined streaming ---------------------------------

    async def _generate_cf_stream(
        self, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Run a session request as three overlapping stages instead of three phases.

        The serial path above is correct and is what a one-shot request wants: each
        stage's whole output is the next stage's whole input, so waiting is not
        waste. A rollout is different. It produces blocks, and a block is
        independently decodable and independently encodable, so DiT block N+1,
        VAE block N and the encoder's block N-1 can all be in flight at once.
        Measured per-latent cost is 0.957 s of DiT and 1.290 s of VAE; run serially
        that is their sum, pipelined it is the max, and the encoder disappears
        under both.

        Three concurrent tasks, chained by the handoffs in :mod:`cf_pipeline`::

            rollout ──Relay──> decode ──FrameQueue──> encoder thread ──> client

        The decode task is the one that has to stay strictly ordered: the VAE's
        temporal ``feat_cache`` makes block N's first frame depend on block N-1's
        last, so it awaits each block in turn. Reordering there would not fail --
        it would produce visible seams. The rollout ahead of it and the encoder
        behind it are free to run as fast as they can.
        """
        if len(self.stage_configs) < 2:
            yield {
                "error": "Causal-Forcing session streaming needs the 2-stage "
                "(DiT + VAE) disaggregated pipeline",
                "finished": True,
            }
            return

        dit_stage, vae_stage = self.stage_configs[0], self.stage_configs[1]
        dit_client = self.stage_clients.get(
            getattr(dit_stage.engine_args, "model_stage", "stage0")
        )
        vae_client = self.stage_clients.get(
            getattr(vae_stage.engine_args, "model_stage", "stage1")
        )
        if dit_client is None or vae_client is None:
            yield {
                "error": "Missing stage client for the CF pipeline",
                "finished": True,
            }
            return

        nvext = request.get("nvext") or {}
        fps = int(nvext.get("fps") or self.config.default_video_fps)
        cmaf = CMAF_ANNOTATION in (nvext.get("annotations") or [])

        blocks = Relay()
        frames = FrameQueue()

        async def rollout() -> None:
            """Stage 0: forward each emitted block to the decode task as it lands."""
            try:
                async for output in self._stage_chunks(
                    dit_client, {"request_id": request_id, **request}
                ):
                    if output.error:
                        raise RuntimeError(f"DiT stage: {output.error}")
                    await blocks.put(output)
                    if output.cf_last:
                        break
            except BaseException as e:  # noqa: BLE001 -- re-raised in the consumer
                blocks.close(e)
            else:
                blocks.close()

        async def decode() -> None:
            """Stage 1: decode blocks in order, feeding the encoder as it goes."""
            try:
                async for output in blocks:
                    stage_request = output.to_next_stage_request(request_id)
                    if output.cf_last:
                        # Forwarded so the VAE worker releases its decode cursor;
                        # it produces no frames, so nothing is queued for it.
                        async for _ in self._stage_chunks(vae_client, stage_request):
                            pass
                        break
                    decoded = await self._fetch_cf_block(
                        vae_client, stage_request, request_id
                    )
                    frames.put(decoded_chunk_to_canonical(decoded))
            except BaseException as e:  # noqa: BLE001 -- re-raised in the consumer
                frames.close(e)
            else:
                frames.close()

        rollout_task = asyncio.create_task(rollout())
        decode_task = asyncio.create_task(decode())
        try:
            if cmaf:
                # The live formatter reports its own failures as a CMAF failure
                # frame, because by then the client already holds a partial asset
                # and needs to be told it is partial rather than complete.
                # Scene length and audio offset, when the scene stated them.
                # `latents` gives the exact frame count (see
                # CFSceneRequest.num_frames), so this is not an estimate; only a
                # scene that omitted it falls back to the upper bound, which the
                # merge then cuts at the video's nominal end.
                scene_duration_s, audio_start_s = self._cf_audio_bounds(
                    request, fps
                )
                audio = self._audio_track(
                    request,
                    fps,
                    duration_s=(
                        _CF_AUDIO_UPPER_BOUND_S
                        if scene_duration_s is None
                        else scene_duration_s
                    ),
                    start_s=audio_start_s,
                )
                if audio.enabled:
                    stream = self._formatter.stream_av_cmaf_live(
                        frames,
                        request_id,
                        fps=fps,
                        audio=audio,
                        video_duration_s=scene_duration_s,
                    )
                else:
                    stream = self._formatter.stream_video_cmaf_live(
                        frames, request_id, fps=fps
                    )
                async for chunk in stream:
                    yield chunk
            else:
                # No CMAF opt-in: collect the stream and answer as one clip. The
                # pipelining still applies -- only the delivery is batched.
                try:
                    collected = await asyncio.to_thread(list, frames)
                except Exception as e:
                    # A producer's terminal error, re-raised here by the handoff.
                    # It has to become a chunk: an exception out of this generator
                    # would reach the client as a dropped connection, which is
                    # indistinguishable from a network fault.
                    logger.error("CF stream failed for %s: %s", request_id, e)
                    yield {"error": str(e), "finished": True}
                    return
                if not collected:
                    yield {"error": "CF stream produced no frames", "finished": True}
                    return
                chunk = await self._formatter.format_video_frames(
                    [np.concatenate(collected, axis=0)], request_id, fps=fps
                )
                if chunk:
                    yield chunk
        finally:
            # The consumer may have stopped early (client disconnect), which leaves
            # both producers blocked on a handoff nobody will drain again.
            for task in (rollout_task, decode_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(rollout_task, decode_task, return_exceptions=True)

    async def _stage_chunks(
        self, client: Any, stage_request: dict
    ) -> AsyncGenerator[StageOutput, None]:
        """Yield a stage's outputs one at a time rather than merging them.

        The serial path collapses a stage's chunks with ``dict.update`` because a
        one-shot stage emits exactly one. A pipelined stage emits one per block,
        and merging them would keep only the last -- silently, since each chunk is
        individually well-formed.
        """
        async for chunk in await client.round_robin(stage_request):
            data = chunk.data()
            if isinstance(data, (str, bytes)):
                data = json.loads(data)
            yield StageOutput.model_validate(data)

    async def _fetch_cf_block(
        self, client: Any, stage_request: dict, request_id: str
    ) -> Any:
        """Send one block to the VAE stage and read the decoded frames back."""
        block = stage_request.get("cf_block", 0)
        last: StageOutput | None = None
        async for output in self._stage_chunks(client, stage_request):
            if output.error:
                raise RuntimeError(f"VAE stage, block {block}: {output.error}")
            last = output
        if last is None:
            raise RuntimeError(f"VAE stage returned nothing for block {block}")

        final_stage_id = self.stage_configs[-1].stage_id
        connector = self.connectors.get(_connector_key(final_stage_id, "router"))
        metadata = (last.stage_connector_refs or {}).get(str(final_stage_id))
        if connector is None or metadata is None:
            raise RuntimeError(
                f"block {block}: the VAE stage produced no router connector ref; a "
                "pipelined stream cannot use the SHM-by-request-id fallback, since "
                "every block would collide on one key"
            )
        payload = unwrap_connector_payload(
            await ensure_awaited(
                connector.get(
                    str(final_stage_id),
                    "router",
                    cf_put_key(request_id, block),
                    metadata=metadata,
                )
            )
        )
        if is_empty_payload(payload):
            raise RuntimeError(f"block {block}: empty payload from the VAE stage")
        return payload

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
                yield {
                    "error": f"Router connector.get() failed: {e}",
                    "finished": True,
                }
                return
        else:
            # --- SHM fallback (single-node: router and final stage on same machine) ---
            shm_meta = stage_output.shm_meta
            if not shm_meta:
                logger.warning("Router: no shm_meta in stage output")
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
            yield {
                "error": f"Formatter returned no output for type '{final_output_type}'",
                "finished": True,
            }


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
