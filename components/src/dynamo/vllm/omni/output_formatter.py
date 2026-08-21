# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modality-specific output formatters for vLLM-Omni.

Extracted from OmniHandler and AudioGenerationHandler so that any consumer
(aggregated handler, disaggregated router, test harness) can format engine
output without creating an engine or loading model weights.
"""

import asyncio
import base64
import logging
import time
import uuid
from io import BytesIO
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np
import soundfile as sf
import torch

from dynamo.common.protocols.audio_protocol import AudioData, NvAudioSpeechResponse
from dynamo.common.protocols.image_protocol import ImageData, NvImagesResponse
from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.output_modalities import RequestType
from dynamo.common.utils.video_utils import StreamingCmafEncoder, encode_video
from dynamo.vllm.handlers import build_prompt_tokens_details
from dynamo.vllm.omni.cmaf_video import (
    CMAF_AUDIO_ERROR_TAG,
    CMAF_AUDIO_INIT_TAG,
    CMAF_AUDIO_SEGMENT_PREFIX,
    CMAF_FALLBACK_VIDEO_CODEC,
    CMAF_INIT_TAG,
    CMAF_METADATA_TAG,
    CMAF_SEGMENT_PREFIX,
    audio_error_bytes,
    cmaf_audio_frag_seconds,
    cmaf_audio_lead_seconds,
    cmaf_emit_cadence_s,
    cmaf_gop_frames,
    cmaf_segment_seconds,
    metadata_bytes,
    stream_audio_file_to_cmaf,
    stream_frames_to_cmaf,
)
from dynamo.vllm.omni.utils import is_empty_payload
from dynamo.vllm.omni.video_convert import to_canonical

logger = logging.getLogger(__name__)

_FRAGMENTS_DONE = object()


class _AsyncFragments:
    """Adapts a blocking fragment generator to ``async for``, one step per thread hop.

    The encoder generator blocks on a subprocess pipe. Iterating it directly on
    the event loop would stall every other request in this worker for the whole
    encode -- which in a streaming deploy is unbounded. Each ``next()`` therefore
    runs via :func:`asyncio.to_thread`.

    One step at a time rather than a producer thread with a queue, deliberately:
    the generator is only ever touched by one thread at a time, so there is no
    shared mutable state to get wrong, and back-pressure stays natural -- nothing
    is encoded ahead of what the client has taken. The cost is a thread hop per
    segment, which is nothing against encoding one.
    """

    def __init__(self, generator) -> None:
        self._gen = generator
        self._closed = False

    def __aiter__(self) -> "_AsyncFragments":
        return self

    async def __anext__(self):
        item = await asyncio.to_thread(next, self._gen, _FRAGMENTS_DONE)
        if item is _FRAGMENTS_DONE:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        """Close the generator so its cleanup runs (it reaps the encoder process).

        Idempotent, and safe to call after exhaustion -- ``stream_video_cmaf``
        calls it from a ``finally``, where the normal-completion path has already
        run the generator to the end.
        """
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._gen.close)


class AudioTrack:
    """The audio side of a two-track CMAF stream, pulled one fragment at a time.

    Owns a lazy :func:`stream_audio_file_to_cmaf` generator plus the bookkeeping a
    caller needs to interleave it against video: how many seconds have been handed
    out, and whether the track is still live.

    Deliberately *not* a producer task. Nothing is encoded ahead of what the caller
    has taken, so a track that is being throttled costs no memory, and the same
    object works whether the source outruns realtime (a file) or lags it (a
    worker). It is shared by both live CMAF paths in the router, which differ in
    their video segment cadence but not in how audio is paced against it.

    Failures are *soft*: :meth:`open` and :meth:`next_fragment` return None rather
    than raising, recording the reason in :attr:`error`. A dead audio track must
    never take a healthy video track with it.

    ``source=None`` builds a permanently-disabled track. That is not a special
    case bolted on -- it is what lets the A/V merge be the *only* live path in the
    router: a video-only request is a request whose audio track is disabled, so
    there is no second emit loop to keep in step with this one.
    """

    def __init__(
        self,
        source: Optional[str],
        duration_s: float,
        *,
        frag_seconds: float | None = None,
        start_s: float = 0.0,
    ) -> None:
        self._source = source
        self._duration_s = duration_s
        # Where in the source to start. Non-zero for a chained scene, so the bed
        # continues across requests instead of restarting; invisible downstream,
        # since the fragments produced still begin at timestamp 0.
        self._start_s = start_s
        self._frag_seconds = (
            cmaf_audio_frag_seconds() if frag_seconds is None else frag_seconds
        )
        self._pieces: Optional[_AsyncFragments] = None
        self._index = 0
        self.codec: Optional[str] = None
        self.init_bytes: Optional[bytes] = None
        self.emitted_seconds = 0.0
        self.done = source is None
        self.error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._source is not None

    @property
    def frag_seconds(self) -> float:
        """Nominal fragment duration used for the interleave watermark.

        Nominal, not measured: the muxer cuts on codec frame boundaries so real
        fragments differ slightly (an AAC-LC frame is 21.33 ms, so a requested
        0.5 s lands near 0.512 s). That inaccuracy is affordable because the
        watermark decides *delivery order* only -- both tracks carry their own
        ``tfdt`` timestamps, so the browser's A/V sync is unaffected by it.
        """
        return self._frag_seconds

    @property
    def live(self) -> bool:
        return not self.done and self.error is None

    async def open(self) -> Optional[str]:
        """Start the encoder and take its init segment. Returns the codec string.

        Returns None on failure, having set :attr:`error`. Callers must treat that
        as "no audio track" and carry on with video: this runs *before* the
        metadata frame is sent precisely so ``has_audio`` can be truthful rather
        than advertised and then retracted.

        A disabled track (``source=None``) reports no codec and no error: nothing
        went wrong, there is simply no audio.
        """
        if self._source is None:
            return None
        try:
            self._pieces = _AsyncFragments(
                stream_audio_file_to_cmaf(
                    self._source,
                    self._duration_s,
                    frag_seconds=self._frag_seconds,
                    start_s=self._start_s,
                )
            )
            kind, payload, extra = await self._pieces.__anext__()
        except StopAsyncIteration:
            self._fail("audio encoder produced no output")
            return None
        except Exception as e:  # noqa: BLE001 -- degrade to video-only
            self._fail(str(e))
            return None

        if kind != "init":
            self._fail(f"expected an audio init segment first, got {kind!r}")
            return None
        self.codec, self.init_bytes = extra, payload
        return self.codec

    async def next_fragment(self) -> Optional[tuple[int, bytes]]:
        """Take the next media fragment, or None when the track ends or fails.

        Advances :attr:`emitted_seconds` by the nominal fragment duration, so the
        caller's watermark stays in step without having to parse durations back
        out of the fragment.
        """
        if self._pieces is None or not self.live:
            return None
        try:
            kind, payload, extra = await self._pieces.__anext__()
        except StopAsyncIteration:
            self.done = True
            return None
        except Exception as e:  # noqa: BLE001 -- degrade to video-only
            self._fail(str(e))
            return None
        if kind != "segment":
            # A second init cannot happen with one encoder, but silently
            # appending one to a SourceBuffer would reset its timeline.
            self._fail(f"unexpected audio piece {kind!r} mid-stream")
            return None
        self.emitted_seconds += self._frag_seconds
        self._index = extra + 1
        return extra, payload

    async def aclose(self) -> None:
        """Reap the encoder. Safe to call whether or not the track was opened."""
        if self._pieces is not None:
            await self._pieces.aclose()

    def _fail(self, message: str) -> None:
        self.error = message
        self.done = True
        logger.warning("CMAF audio track (%s): %s", self._source, message)


class TextFormatter:
    """Formats LLM text output as OpenAI chat completion chunks."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def format(
        self,
        request_output: Any,
        request_id: str,
        *,
        previous_text: str = "",
    ) -> Dict[str, Any] | None:
        if not request_output.outputs:
            return _error_chunk(request_id, self._model_name, "No outputs from engine")

        output = request_output.outputs[0]
        delta_text = output.text[len(previous_text) :]

        chunk: Dict[str, Any] = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": (
                        normalize_finish_reason(output.finish_reason)
                        if output.finish_reason
                        else None
                    ),
                }
            ],
        }

        if output.finish_reason:
            chunk["usage"] = _build_completion_usage(request_output)

        return chunk


class DiffusionFormatter:
    """Formats diffusion output (images/video frames) for the frontend.

    Handles both image and video — routes by request_type since vllm-omni
    reports final_output_type="image" for all diffusion outputs.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any,
        media_http_url: Optional[str],
        default_fps: int = 16,
    ) -> None:
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._default_fps = default_fps

    async def format(
        self, stage_output: Any, request_id: str, *, request_type: Any, **ctx: Any
    ) -> Dict[str, Any] | None:
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )
        if is_empty_payload(images):
            return None

        if request_type == RequestType.VIDEO_GENERATION:
            return await self._encode_video(
                images,
                request_id,
                fps=ctx.get("fps", self._default_fps),
                response_format=ctx.get("response_format"),
                output_format=ctx.get("output_format"),
            )
        return await self._encode_image(
            images,
            request_id,
            request_type=request_type,
            response_format=ctx.get("response_format"),
        )

    async def _encode_video(
        self,
        images: list,
        request_id: str,
        fps: int,
        response_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        output_format = output_format or "mp4"
        response_format = response_format or "url"
        if response_format not in ("url", "b64_json"):
            raise ValueError(
                f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'"
            )
        if output_format != "mp4":
            raise ValueError(
                f"Unsupported output_format: {output_format!r}; only 'mp4' is supported"
            )
        try:
            start_time = time.time()
            canonical = to_canonical(images)
            video_bytes = await asyncio.to_thread(
                encode_video, canonical, fps, container=output_format
            )

            if response_format == "b64_json":
                video_data = VideoData(
                    output_format=output_format,
                    b64_json=base64.b64encode(video_bytes).decode("utf-8"),
                )
            else:
                video_url = await upload_to_fs(
                    self._media_fs,
                    f"videos/{request_id}.{output_format}",
                    video_bytes,
                    self._media_http_url,
                )
                video_data = VideoData(output_format=output_format, url=video_url)

            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[video_data],
                inference_time_s=time.time() - start_time,
            ).model_dump()
        except Exception as e:
            logger.error("Failed to encode video for request %s: %s", request_id, e)
            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            ).model_dump()

    def _cmaf_chunk(
        self, request_id: str, created: int, tag: str, payload: bytes, progress: int
    ) -> Dict[str, Any]:
        """Build one binary-CMAF NvVideosResponse item (tag in url, b64 payload)."""
        return NvVideosResponse(
            id=request_id,
            object="video",
            model=self._model_name,
            status="in_progress",
            progress=progress,
            created=created,
            data=[
                VideoData(
                    output_format="mp4",
                    url=tag,
                    b64_json=base64.b64encode(payload).decode("ascii"),
                )
            ],
        ).model_dump()

    def _cmaf_failure(
        self, request_id: str, created: int, error: str
    ) -> Dict[str, Any]:
        return NvVideosResponse(
            id=request_id,
            object="video",
            model=self._model_name,
            status="failed",
            progress=0,
            created=created,
            data=[],
            error=error,
        ).model_dump()

    async def stream_video_cmaf(
        self, stage_output: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Encode a generated clip incrementally, streaming CMAF as it emerges.

        Yields a ``cmaf:metadata`` item, then ``cmaf:init``, then one
        ``cmaf:segment:{n}`` item per fMP4 media fragment. The frontend's
        binary-CMAF route re-frames these into a single binary response body.
        Non-final stage outputs (empty payloads) yield nothing.

        Pieces are emitted **as the encoder produces them** rather than after the
        whole clip is packaged, so the client can start decoding on segment 0.
        The frames still arrive here in one batch today -- the router hands the
        formatter a single finished ``StageOutput`` -- so what this removes is
        only the encode+split tail (~0.5 s of a ~55 s request). It matters
        because it is the half that has to exist first: once a session rollout
        feeds frames chunk by chunk, this same path ships segment 0 while the
        later frames are still being generated, with no further change here.

        Because the total is unknown while encoding, the metadata frame declares
        an open-ended stream and progress cannot be a true percentage -- see
        :meth:`_cmaf_progress`.

        :meth:`stream_video_cmaf_persistent` is the alternative encoder for this
        same wire protocol -- one long-lived ffmpeg instead of a per-call
        fragment generator. See its docstring for when each is preferable.
        """
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )
        if is_empty_payload(images):
            return

        created = int(time.time())
        try:
            canonical = to_canonical(images)
        except Exception as e:
            logger.error("Failed to convert frames for request %s: %s", request_id, e)
            yield self._cmaf_failure(request_id, created, str(e))
            return

        segment_seconds = cmaf_segment_seconds()
        cadence = cmaf_emit_cadence_s()
        # Known here only because the frames arrive whole; a live rollout would
        # not know it, which is exactly why the protocol carries open_ended.
        expected = self._expected_segments(len(canonical), fps, segment_seconds)

        pieces = _AsyncFragments(
            stream_frames_to_cmaf(
                self._frame_chunks(canonical, fps, segment_seconds),
                fps,
                segment_seconds,
            )
        )
        emitted = 0
        try:
            async for kind, payload, extra in pieces:
                if kind == "init":
                    yield self._cmaf_chunk(
                        request_id,
                        created,
                        CMAF_METADATA_TAG,
                        metadata_bytes(None, segment_seconds, extra),
                        progress=0,
                    )
                    yield self._cmaf_chunk(
                        request_id, created, CMAF_INIT_TAG, payload, progress=1
                    )
                    continue
                if cadence > 0:
                    await asyncio.sleep(cadence)
                emitted += 1
                yield self._cmaf_chunk(
                    request_id,
                    created,
                    f"{CMAF_SEGMENT_PREFIX}{extra}",
                    payload,
                    self._cmaf_progress(emitted, expected),
                )
        except Exception as e:
            # A mid-stream failure has already sent init and possibly segments,
            # so the client holds a partial but valid asset. Say so rather than
            # letting the stream just stop, which is indistinguishable from a
            # completed short clip.
            logger.error(
                "CMAF streaming failed for request %s after %d segment(s): %s",
                request_id,
                emitted,
                e,
            )
            yield self._cmaf_failure(request_id, created, str(e))
            return
        finally:
            await pieces.aclose()

        logger.info("CMAF stream for %s complete: %d segment(s)", request_id, emitted)

    async def stream_video_cmaf_persistent(
        self, stage_output: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """:meth:`stream_video_cmaf` over one long-lived ffmpeg process.

        Identical wire output; different encoder ownership. The default path
        drives :func:`encode_video_fragments`, a generator that owns an ffmpeg
        for the duration of one call -- simple, and it back-pressures naturally
        because nothing is encoded ahead of what the consumer has taken.
        ``StreamingCmafEncoder`` instead keeps a process alive across pushes, so
        the muxer -- not us -- owns ``mfhd.sequence_number`` and
        ``tfdt.baseMediaDecodeTime`` across the whole presentation, and there is
        exactly one init segment no matter how many pushes arrive.

        That property is what a *multi-request* stream needs: many pushes, one
        continuous timeline. It is unused by the router today, which gets its
        continuity from a single generator per presentation instead, so this
        stays available rather than default -- there is no reason to prefer a
        persistent subprocess when one call has all the frames.
        """
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )
        if is_empty_payload(images):
            return

        created = int(time.time())
        cadence = cmaf_emit_cadence_s()
        try:
            canonical = to_canonical(images)
            height, width = canonical.shape[1], canonical.shape[2]
            enc = StreamingCmafEncoder(fps, width, height, gop_frames=cmaf_gop_frames())
            await enc.start()

            seg_index = 0

            async def emit(kind: str, payload: bytes):
                nonlocal seg_index
                if kind == "init":
                    # The codec string is parsed FROM the init, and metadata must
                    # precede the init append on the client -- so send it here.
                    yield self._cmaf_chunk(
                        request_id,
                        created,
                        CMAF_METADATA_TAG,
                        metadata_bytes(
                            None,
                            cmaf_segment_seconds(),
                            enc.codec_string() or CMAF_FALLBACK_VIDEO_CODEC,
                        ),
                        progress=0,
                    )
                    yield self._cmaf_chunk(
                        request_id, created, CMAF_INIT_TAG, payload, progress=1
                    )
                else:
                    if cadence > 0:
                        await asyncio.sleep(cadence)
                    yield self._cmaf_chunk(
                        request_id,
                        created,
                        f"{CMAF_SEGMENT_PREFIX}{seg_index}",
                        payload,
                        progress=min(99, 2 + seg_index),
                    )
                    seg_index += 1

            async for kind, payload in enc.push(canonical):
                async for chunk in emit(kind, payload):
                    yield chunk
            async for kind, payload in enc.finish():
                async for chunk in emit(kind, payload):
                    yield chunk
        except Exception as e:
            logger.error("Failed to stream CMAF for request %s: %s", request_id, e)
            yield self._cmaf_failure(request_id, created, str(e))
            return

        logger.info(
            "Persistent CMAF stream for %s complete: %d segment(s)",
            request_id,
            seg_index,
        )

    async def stream_video_cmaf_live(
        self, frame_chunks: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream binary-CMAF pieces from frames that are still being generated.

        Same wire protocol as :meth:`stream_video_cmaf`, and the same encoder --
        the only difference is where the frames come from. There, they arrive as
        one finished clip, so the segment count is known and only the encode tail
        overlaps generation. Here ``frame_chunks`` is a blocking iterable the
        producer fills as the rollout decodes, so segment 0 ships while later
        frames do not exist yet. That is the pipelining this whole path was built
        for, and the reason the batch version chunks its input at all.

        ``expected`` is therefore genuinely unknown, not merely unstated. Progress
        is a monotonic count capped below 100 rather than a fraction, because
        there is no denominator to divide by -- see :meth:`_cmaf_progress`.
        """
        created = int(time.time())
        segment_seconds = cmaf_segment_seconds()
        cadence = cmaf_emit_cadence_s()

        pieces = _AsyncFragments(
            stream_frames_to_cmaf(frame_chunks, fps, segment_seconds)
        )
        emitted = 0
        try:
            async for kind, payload, extra in pieces:
                if kind == "init":
                    yield self._cmaf_chunk(
                        request_id,
                        created,
                        CMAF_METADATA_TAG,
                        metadata_bytes(None, segment_seconds, extra),
                        progress=0,
                    )
                    yield self._cmaf_chunk(
                        request_id, created, CMAF_INIT_TAG, payload, progress=1
                    )
                    continue
                if cadence > 0:
                    await asyncio.sleep(cadence)
                emitted += 1
                yield self._cmaf_chunk(
                    request_id,
                    created,
                    f"{CMAF_SEGMENT_PREFIX}{extra}",
                    payload,
                    self._live_progress(emitted),
                )
        except Exception as e:
            # init and some segments may already be with the client, so it holds a
            # partial but valid asset. Saying so beats stopping silently, which is
            # indistinguishable from a clip that simply ended.
            logger.error(
                "Live CMAF streaming failed for request %s after %d segment(s): %s",
                request_id,
                emitted,
                e,
            )
            yield self._cmaf_failure(request_id, created, str(e))
            return
        finally:
            await pieces.aclose()

        logger.info(
            "Live CMAF stream for %s complete: %d segment(s)", request_id, emitted
        )

    async def stream_av_cmaf_live(
        self,
        frame_chunks: Any,
        request_id: str,
        *,
        fps: int,
        audio: "AudioTrack",
        video_duration_s: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """:meth:`stream_video_cmaf_live` with a second, independent audio track.

        Sibling rather than an optional parameter, so a video-only request runs
        exactly the code it runs today. The video half is identical -- the same
        fragment generator over the same live ``frame_chunks``; all that is added
        is a second track merged onto the same wire by
        :meth:`_merge_av_cmaf`.

        ``video_duration_s`` is the clip's real length when the caller knows it,
        and it decides where audio is cut. See :meth:`_merge_av_cmaf`.
        """
        pieces = _AsyncFragments(
            stream_frames_to_cmaf(frame_chunks, fps, cmaf_segment_seconds())
        )
        async for chunk in self._merge_av_cmaf(
            pieces,
            request_id,
            audio=audio,
            video_segment_seconds=float(cmaf_segment_seconds()),
            video_duration_s=video_duration_s,
        ):
            yield chunk

    async def _merge_av_cmaf(
        self,
        video_pieces: Any,
        request_id: str,
        *,
        audio: "AudioTrack",
        video_segment_seconds: float,
        video_duration_s: Optional[float] = None,
        created: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Merge a video piece stream and an audio track into one CMAF wire stream.

        ``video_pieces`` is any async iterator of ``("init", bytes, codec)`` /
        ``("segment", bytes, index)`` with an ``aclose()`` -- either
        :class:`_AsyncFragments` over :func:`stream_frames_to_cmaf` or an adapter
        around a persistent encoder. Both live router paths use this, which is why
        the video cadence is a parameter: the fragment generator closes a segment
        every ``DYN_CMAF_SEGMENT_SECONDS`` (2 s), the persistent encoder every
        ``DYN_CMAF_GOP_FRAMES`` frames (0.25 s at the default 4 @ 16 fps).

        **Ordering.** Metadata is gated on *both* inits, so ``has_audio`` and
        ``audio_codec`` describe what will actually arrive rather than what was
        hoped for -- the client creates both ``SourceBuffer``s from that one frame
        and browsers are unreliable about adding a second one later::

            METADATA -> VIDEO_INIT -> AUDIO_INIT -> <interleaved media> -> DONE

        Gating costs nothing: an init is the first thing either muxer emits, ahead
        of any media.

        **Interleaving** is a watermark on the presentation timeline: emit from
        whichever track has handed out fewer seconds, letting audio sit up to
        ``DYN_CMAF_AUDIO_LEAD_SECONDS`` ahead. The lead is not symmetric because
        MSE is not: playback stalls when *any* active ``SourceBuffer`` lacks data
        at ``currentTime``, so audio starving is as fatal as video starving while
        being far cheaper to prevent.

        The watermark uses nominal durations, not durations parsed back out of
        each fragment, and that is sound because it decides *delivery order only*.
        Each track carries its own ``tfdt``/``trun`` written by its own muxer, so
        the browser syncs by presentation time whatever order the bytes arrived
        in. A drifting watermark costs smoothness; it cannot cost A/V sync.

        **Where audio stops** is a different question, and nominal durations are
        *not* good enough for it -- an over-long audio track is either wasted
        encoding or, for a client stitching scenes onto one timeline, audible
        drift. So ``video_duration_s`` is the authoritative total when the caller
        knows it, and audio is cut there. Only when it is unknown does the cut
        fall back to the accumulated nominal ``video_t``, which measurably
        over-runs: 2 s nominal segments against ~1.7 s real ones over-produced
        audio by 29% on a 69-frame scene.

        A slow audio source therefore also throttles video delivery, since the
        loop blocks on whichever track is behind. That is deliberate for now: it
        keeps the wire ordered by presentation time and makes an audio underrun
        show up as reduced throughput rather than as a mystery stall in the
        browser. A real generator will want a silence-fill policy here instead.

        Failure is per-track. A dead audio track yields one non-fatal
        ``cmaf:audio:error`` item and stops; video runs to completion. A failed
        :class:`NvVideosResponse` could not be used for that -- the frontend
        aborts the whole stream on one.
        """
        created = int(time.time()) if created is None else created
        cadence = cmaf_emit_cadence_s()
        lead = cmaf_audio_lead_seconds()

        video_t = 0.0
        audio_t = 0.0
        emitted = 0
        audio_index = 0
        video_done = False
        audio_error_sent = False

        try:
            # --- Preamble: video init, then audio init, then metadata. ---
            try:
                kind, payload, extra = await video_pieces.__anext__()
            except StopAsyncIteration:
                raise RuntimeError("video encoder produced no output") from None
            if kind != "init":
                raise RuntimeError(f"expected a video init segment first, got {kind!r}")
            video_codec = extra or CMAF_FALLBACK_VIDEO_CODEC
            video_init = payload

            audio_codec = await audio.open()
            if audio_codec is None and audio.enabled:
                logger.warning(
                    "CMAF audio unavailable for %s (%s); streaming video-only",
                    request_id,
                    audio.error,
                )

            yield self._cmaf_chunk(
                request_id,
                created,
                CMAF_METADATA_TAG,
                metadata_bytes(
                    None, cmaf_segment_seconds(), video_codec, audio_codec
                ),
                progress=0,
            )
            yield self._cmaf_chunk(
                request_id, created, CMAF_INIT_TAG, video_init, progress=1
            )
            if audio_codec is not None and audio.init_bytes is not None:
                yield self._cmaf_chunk(
                    request_id,
                    created,
                    CMAF_AUDIO_INIT_TAG,
                    audio.init_bytes,
                    progress=1,
                )

            def audio_covered() -> bool:
                """True once audio spans the whole video, so it should stop.

                Audio is slaved to the video's duration and the encoder is only
                ever asked for an upper bound, so somebody has to decide where to
                cut. ``video_duration_s`` is that decision when the caller knows
                it -- a scene of known length -- and it is checked even while
                video is still arriving, because the nominal watermark runs ahead
                of real fragments and would otherwise keep asking for audio past
                the end of the clip.

                Without it the only available answer is the accumulated nominal
                ``video_t``, and that is usable only once video has finished: it
                is a lower bound while segments are still coming.
                """
                if video_duration_s is not None:
                    return audio_t >= video_duration_s
                return video_done and audio_t >= video_t

            # --- Media: whichever track is behind on the timeline. ---
            while not video_done or (audio.live and not audio_covered()):
                if not audio.live or audio_covered():
                    take_audio = False
                else:
                    # A finished video track takes audio unconditionally: audio
                    # already past the lead would otherwise spin taking neither.
                    take_audio = video_done or audio_t <= video_t + lead

                if take_audio:
                    fragment = await audio.next_fragment()
                    if fragment is None:
                        if audio.error and not audio_error_sent:
                            audio_error_sent = True
                            yield self._cmaf_chunk(
                                request_id,
                                created,
                                CMAF_AUDIO_ERROR_TAG,
                                audio_error_bytes(audio.error),
                                self._live_progress(emitted),
                            )
                        continue
                    index, audio_payload = fragment
                    audio_t = audio.emitted_seconds
                    audio_index = index + 1
                    yield self._cmaf_chunk(
                        request_id,
                        created,
                        f"{CMAF_AUDIO_SEGMENT_PREFIX}{index}",
                        audio_payload,
                        self._live_progress(emitted),
                    )
                    continue

                try:
                    kind, payload, extra = await video_pieces.__anext__()
                except StopAsyncIteration:
                    video_done = True
                    continue
                if kind == "init":
                    # One encoder emits one init; forwarded rather than dropped
                    # only because the video-only paths do the same.
                    yield self._cmaf_chunk(
                        request_id, created, CMAF_INIT_TAG, payload, progress=1
                    )
                    continue
                if cadence > 0:
                    await asyncio.sleep(cadence)
                emitted += 1
                video_t += video_segment_seconds
                yield self._cmaf_chunk(
                    request_id,
                    created,
                    f"{CMAF_SEGMENT_PREFIX}{emitted - 1}",
                    payload,
                    self._live_progress(emitted),
                )
        except Exception as e:
            # init and some segments are already with the client, so it holds a
            # partial but valid asset. Saying so beats stopping silently, which
            # is indistinguishable from a clip that simply ended.
            logger.error(
                "Live A/V CMAF streaming failed for request %s after %d video "
                "segment(s) and %d audio fragment(s): %s",
                request_id,
                emitted,
                audio_index,
                e,
            )
            yield self._cmaf_failure(request_id, created, str(e))
            return
        finally:
            await audio.aclose()
            await video_pieces.aclose()

        logger.info(
            "Live A/V CMAF stream for %s complete: %d video segment(s), "
            "%d audio fragment(s)",
            request_id,
            emitted,
            audio_index,
        )

    @staticmethod
    def _live_progress(emitted: int) -> int:
        """Progress for a stream whose length is not knowable.

        Monotonic and asymptotic rather than a percentage: there is no total to be
        a fraction of. It approaches but never reaches 100, so a client can show
        motion without ever being told the stream finished before it did.
        """
        return max(1, min(99, 100 - int(90 / (1 + emitted * 0.15))))

    @staticmethod
    def _expected_segments(num_frames: int, fps: int, segment_seconds: int) -> int:
        """Best-effort segment count, used only to shape the progress number."""
        per_segment = max(1, fps * max(1, segment_seconds))
        return max(1, -(-num_frames // per_segment))

    @staticmethod
    def _cmaf_progress(emitted: int, expected: int) -> int:
        """Progress for an open-ended stream.

        An estimate, deliberately capped below 100: hardware encoders do not
        always honor the requested GOP, so the real segment count can exceed the
        prediction, and a progress field that reached 100 mid-stream would be
        worse than one that merely lags.
        """
        return max(1, min(99, int((emitted / max(1, expected)) * 100)))

    @staticmethod
    def _frame_chunks(canonical: np.ndarray, fps: int, segment_seconds: int):
        """Feed the encoder one segment's worth of frames at a time.

        Chunking matters even though the frames are all in hand: writing the
        whole array in one ``write()`` would fill the pipe buffer and block until
        the encoder drained it, which serializes the very overlap this exists
        for. It also mirrors the shape a live rollout will deliver.
        """
        step = max(1, fps * max(1, segment_seconds))
        for start in range(0, len(canonical), step):
            yield canonical[start : start + step]

    async def _encode_image(
        self,
        images: list,
        request_id: str,
        *,
        request_type: Any,
        response_format: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        if is_empty_payload(images):
            return _error_chunk(request_id, self._model_name, "No images generated")

        data_urls = await self._prepare_images(images, request_id, response_format)

        if request_type == RequestType.CHAT_COMPLETION:
            return {
                "id": request_id,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "model": self._model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": u}}
                                for u in data_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        if request_type == RequestType.IMAGE_GENERATION:
            image_data_list = []
            for data_url in data_urls:
                if response_format == "url":
                    image_data_list.append(ImageData(url=data_url))
                elif response_format == "b64_json" or response_format is None:
                    b64 = (
                        data_url.split(",", 1)[1]
                        if data_url.startswith("data:")
                        else data_url
                    )
                    image_data_list.append(ImageData(b64_json=b64))
                else:
                    raise ValueError(f"Invalid response format: {response_format}")
            return NvImagesResponse(
                created=int(time.time()), data=image_data_list
            ).model_dump()

        return None

    async def _prepare_images(
        self, images: list, request_id: str, response_format: Optional[str] = None
    ) -> list:
        outlist = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            if response_format == "url":
                url = await upload_to_fs(
                    self._media_fs,
                    f"images/{request_id}/{uuid.uuid4()}.png",
                    image_bytes,
                    self._media_http_url,
                )
                outlist.append(url)
            elif response_format == "b64_json" or response_format is None:
                outlist.append(
                    f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                )
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return outlist


class AudioFormatter:
    """Formats audio multimodal_output → NvAudioSpeechResponse."""

    def __init__(
        self, model_name: str, media_fs: Any, media_http_url: Optional[str]
    ) -> None:
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._AudioData = AudioData  # stored for use in format()

    async def format(
        self, stage_output: Any, request_id: str, **ctx: Any
    ) -> Dict[str, Any] | None:
        mm_output = (
            stage_output.multimodal_output
            if hasattr(stage_output, "multimodal_output")
            else stage_output
        )
        if is_empty_payload(mm_output):
            return self._error_response(request_id, "No audio generated")

        response_format = ctx.get("response_format")
        output_format = ctx.get("output_format")
        speed = ctx.get("speed", 1.0)

        try:
            start_time = time.time()
            audio_np, sample_rate = self._extract_audio_tensor(mm_output)

            encode_fmt = "wav" if output_format is None else output_format
            assert encode_fmt is not None
            audio_bytes, media_type = await asyncio.to_thread(
                self._encode_audio, audio_np, sample_rate, encode_fmt, speed
            )

            logger.info(
                "Audio encoded for request %s: %d samples, sr=%d, %d bytes %s",
                request_id,
                len(audio_np),
                sample_rate,
                len(audio_bytes),
                encode_fmt,
            )

            if response_format == "url":
                ext = encode_fmt if encode_fmt != "opus" else "ogg"
                url = await upload_to_fs(
                    self._media_fs,
                    f"audios/{request_id}/{uuid.uuid4()}.{ext}",
                    audio_bytes,
                    self._media_http_url,
                )
                audio_data_obj = self._AudioData(output_format=encode_fmt, url=url)
            else:
                audio_data_obj = self._AudioData(
                    output_format=encode_fmt,
                    b64_json=base64.b64encode(audio_bytes).decode(),
                )

            return NvAudioSpeechResponse(
                id=request_id,
                object="audio.speech",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[audio_data_obj],
                inference_time_s=time.time() - start_time,
            ).model_dump()

        except Exception as e:
            logger.error("Failed to process audio for request %s: %s", request_id, e)
            return self._error_response(request_id, str(e))

    def _extract_audio_tensor(self, mm_output: Dict[str, Any]) -> tuple:
        audio_key = "audio" if "audio" in mm_output else "model_outputs"
        audio_val = mm_output.get(audio_key)
        if audio_val is None:
            raise ValueError(
                f"No audio data in multimodal_output. Keys: {list(mm_output.keys())}"
            )

        if isinstance(audio_val, list):
            audio_val = torch.cat(audio_val, dim=-1)

        if hasattr(audio_val, "float"):
            audio_np = audio_val.float().detach().cpu().numpy()
        elif isinstance(audio_val, np.ndarray):
            audio_np = audio_val.astype(np.float32)
        else:
            audio_np = np.array(audio_val, dtype=np.float32)

        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        sr_raw = mm_output.get("sr", 24000)
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else 24000
        sample_rate = sr_raw.item() if hasattr(sr_raw, "item") else int(sr_raw)

        return audio_np, sample_rate

    def _encode_audio(
        self, audio_np: Any, sample_rate: int, fmt: str = "wav", speed: float = 1.0
    ) -> tuple:
        if speed != 1.0:
            try:
                import librosa

                audio_np = librosa.effects.time_stretch(y=audio_np, rate=speed)
            except ImportError:
                logger.warning("librosa not installed, ignoring speed adjustment")

        fmt = (fmt or "wav").lower()
        format_map = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "aac": ("AAC", "audio/aac", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if fmt not in format_map:
            logger.warning("Unsupported format '%s', defaulting to wav", fmt)
            fmt = "wav"

        sf_format, media_type, kwargs = format_map[fmt]

        buf = BytesIO()
        sf.write(buf, audio_np, sample_rate, format=sf_format, **kwargs)
        return buf.getvalue(), media_type

    def _error_response(self, request_id: str, error: str) -> Dict[str, Any]:
        return NvAudioSpeechResponse(
            id=request_id,
            model=self._model_name,
            status="failed",
            created=int(time.time()),
            error=error,
        ).model_dump()


def _error_chunk(
    request_id: str, model_name: str, error_message: str
) -> Dict[str, Any]:
    """Error response in OpenAI chat.completion.chunk format."""
    return {
        "id": request_id,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"Error: {error_message}"},
                "finish_reason": "error",
            }
        ],
    }


def _build_completion_usage(request_output: Any) -> Dict[str, Any]:
    """Build completion usage stats from a vLLM RequestOutput."""
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    prompt_tokens = (
        len(prompt_token_ids)
        if prompt_token_ids is not None and not is_empty_payload(prompt_token_ids)
        else None
    )
    completion_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": build_prompt_tokens_details(
            getattr(request_output, "num_cached_tokens", None)
        ),
    }


class OutputFormatter:
    """Dispatches raw engine output to modality-specific formatters.

    Shared by OmniHandler (aggregated) and any future disaggregated router.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any = None,
        media_http_url: Optional[str] = None,
        default_fps: int = 16,
    ) -> None:
        self._formatters: Dict[str, Any] = {
            "text": TextFormatter(model_name),
            "image": DiffusionFormatter(
                model_name, media_fs, media_http_url, default_fps
            ),
            "audio": AudioFormatter(model_name, media_fs, media_http_url),
        }

    async def format(
        self,
        stage_output: Any,
        request_id: str,
        *,
        request_type: Any = None,
        **ctx: Any,
    ) -> Dict[str, Any] | None:
        fmt_type = getattr(stage_output, "final_output_type", None)
        formatter = self._formatters.get(fmt_type) if fmt_type else None
        if formatter is None:
            return None

        # TextFormatter is sync and takes request_output, not stage_output.
        if fmt_type == "text":
            ro = getattr(stage_output, "request_output", None)
            if not ro:
                return None
            return formatter.format(
                ro, request_id, previous_text=ctx.get("previous_text", "")
            )

        return await formatter.format(
            stage_output, request_id, request_type=request_type, **ctx
        )

    def stream_video_cmaf(
        self, stage_output: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Delegate binary-CMAF streaming to the diffusion (image/video) formatter."""
        return self._formatters["image"].stream_video_cmaf(
            stage_output, request_id, fps=fps
        )

    def stream_video_cmaf_persistent(
        self, stage_output: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Delegate persistent-ffmpeg binary-CMAF streaming to the diffusion formatter."""
        return self._formatters["image"].stream_video_cmaf_persistent(
            stage_output, request_id, fps=fps
        )

    def stream_video_cmaf_live(
        self, frame_chunks: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Delegate live binary-CMAF streaming to the diffusion formatter."""
        return self._formatters["image"].stream_video_cmaf_live(
            frame_chunks, request_id, fps=fps
        )

    def stream_av_cmaf_live(
        self,
        frame_chunks: Any,
        request_id: str,
        *,
        fps: int,
        audio: AudioTrack,
        video_duration_s: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Delegate live binary-CMAF A/V streaming to the diffusion formatter."""
        return self._formatters["image"].stream_av_cmaf_live(
            frame_chunks,
            request_id,
            fps=fps,
            audio=audio,
            video_duration_s=video_duration_s,
        )

    def merge_av_cmaf(
        self,
        video_pieces: Any,
        request_id: str,
        *,
        audio: AudioTrack,
        video_segment_seconds: float,
        video_duration_s: Optional[float] = None,
        created: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Merge caller-supplied video pieces with an audio track.

        For the router's persistent-encoder path, which owns its own video piece
        source and so cannot use :meth:`stream_av_cmaf_live`.
        """
        return self._formatters["image"]._merge_av_cmaf(
            video_pieces,
            request_id,
            audio=audio,
            video_segment_seconds=video_segment_seconds,
            video_duration_s=video_duration_s,
            created=created,
        )

    async def format_video_frames(
        self, frames: Any, request_id: str, *, fps: int
    ) -> Dict[str, Any] | None:
        """Format already-canonical frames as a single video response.

        :meth:`format` cannot serve this: it dispatches on the engine output's
        ``final_output_type``, and a pipelined stream never has an engine output --
        the router assembles its own frames from per-block decodes. This addresses
        the video formatter directly instead of inventing a fake wrapper object.
        """
        return await self._formatters["image"].format(
            frames, request_id, request_type=RequestType.VIDEO_GENERATION, fps=fps
        )
