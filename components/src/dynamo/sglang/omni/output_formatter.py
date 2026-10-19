# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging decoded frames as the frontend's video responses.

Video only, and smaller than its vLLM-Omni counterpart for that reason: this
entrypoint serves one pipeline, Causal-Forcing text-to-video, so there is no
modality dispatch to do and no image or audio leg to carry.

Two deliveries of the same frames:

* **Live CMAF** -- the default for this pipeline, and the one the browser demo
  uses. Frames arrive as the rollout decodes them, so a media segment ships while
  later frames do not yet exist. That is the whole point of the pipelined stream:
  time-to-first-frame stops being time-to-whole-clip.
* **Batch mp4** -- one finished clip, for ``POST /v1/videos`` without the CMAF
  opt-in. The pipelining still applies; only the delivery waits.

What is *not* here is the wire protocol. The tags, the metadata JSON and the
segment framing live in :mod:`dynamo.common.omni.cmaf_video`, shared with the
vLLM-Omni backend, because they are a contract with the browser client rather than
with an engine -- a backend that drifted from them would not be a local bug, it
would be a protocol break in a client nobody changed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any, AsyncGenerator, Optional

import numpy as np

from dynamo.common.omni.cf_pipeline import AsyncFragments
from dynamo.common.omni.cmaf_video import (
    CMAF_INIT_TAG,
    CMAF_METADATA_TAG,
    CMAF_SEGMENT_PREFIX,
    cmaf_emit_cadence_s,
    cmaf_segment_seconds,
    metadata_bytes,
    stream_frames_to_cmaf,
)
from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.video_utils import encode_video

logger = logging.getLogger(__name__)


class VideoFormatter:
    """Turns canonical ``(T, H, W, 3) uint8`` frames into frontend responses."""

    def __init__(
        self,
        model_name: str,
        media_fs: Any = None,
        media_http_url: Optional[str] = None,
        default_fps: int = 16,
    ) -> None:
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._default_fps = default_fps

    # -- live CMAF ----------------------------------------------------------

    async def stream_video_cmaf_live(
        self, frame_chunks: Any, request_id: str, *, fps: int
    ) -> AsyncGenerator[dict, None]:
        """Stream binary-CMAF pieces from frames that are still being generated.

        ``frame_chunks`` is a *blocking* iterable -- a
        :class:`~dynamo.common.omni.cf_pipeline.FrameQueue` the decode task fills
        as blocks come back. The encoder pulls from it on its own writer thread,
        and its finished fragments come back to this coroutine through
        :class:`~dynamo.common.omni.cf_pipeline.AsyncFragments`. So neither
        direction of the encoder's pipe is ever touched from the event loop.

        Progress is a monotonic count capped below 100, not a fraction: a live
        stream has no denominator, and a progress field that reached 100 mid-stream
        would be worse than one that lags. See :meth:`_live_progress`.
        """
        created = int(time.time())
        segment_seconds = cmaf_segment_seconds()
        cadence = cmaf_emit_cadence_s()

        pieces = AsyncFragments(
            stream_frames_to_cmaf(frame_chunks, fps, segment_seconds)
        )
        emitted = 0
        try:
            async for kind, payload, extra in pieces:
                if kind == "init":
                    # ``extra`` is the codec string parsed out of the init segment,
                    # so the metadata advertises the profile and level the encoder
                    # actually produced rather than the one we asked for.
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
        except Exception as e:  # noqa: BLE001 -- the client holds a partial asset
            # init and some segments are already with the client, so it has a
            # partial but valid video. Saying so beats stopping silently, which is
            # indistinguishable from a clip that simply ended.
            logger.error(
                "Live CMAF streaming failed for %s after %d segment(s): %s",
                request_id,
                emitted,
                e,
            )
            yield self._cmaf_failure(request_id, created, str(e))
            return
        finally:
            # Runs the generator's own cleanup, which reaps the encoder process. A
            # client disconnect leaves this coroutine closed mid-iteration, and
            # without it the ffmpeg child would outlive the request.
            await pieces.aclose()

        logger.info(
            "Live CMAF stream for %s complete: %d segment(s)", request_id, emitted
        )

    @staticmethod
    def _live_progress(emitted: int) -> int:
        """Progress for a stream whose length is not knowable.

        Monotonic and asymptotic rather than a percentage: there is no total to be
        a fraction of. It approaches but never reaches 100, so a client can show
        motion without ever being told the stream finished before it did.
        """
        return max(1, min(99, 100 - int(90 / (1 + emitted * 0.15))))

    def _cmaf_chunk(
        self, request_id: str, created: int, tag: str, payload: bytes, progress: int
    ) -> dict:
        """One binary-CMAF response item: the tag in ``url``, the payload in b64."""
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

    def _cmaf_failure(self, request_id: str, created: int, error: str) -> dict:
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

    # -- batch mp4 ----------------------------------------------------------

    async def format_video(
        self,
        frames: np.ndarray,
        request_id: str,
        *,
        fps: Optional[int] = None,
        response_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> dict:
        """Encode one finished clip and answer with a URL or inline base64.

        A failure becomes a ``status="failed"`` response rather than an exception:
        by the time we are encoding, the generation has already happened and the
        client is owed a verdict on it -- an exception out of the handler would
        reach it as a dropped connection.
        """
        fps = int(fps or self._default_fps)
        output_format = output_format or "mp4"
        response_format = response_format or "url"
        if response_format not in ("url", "b64_json"):
            raise ValueError(
                f"Unsupported response_format: {response_format!r}; expected 'url' "
                "or 'b64_json'"
            )
        if output_format != "mp4":
            raise ValueError(
                f"Unsupported output_format: {output_format!r}; only 'mp4' is supported"
            )

        started = time.time()
        try:
            video_bytes = await asyncio.to_thread(
                encode_video, frames, fps, container=output_format
            )
            if response_format == "b64_json":
                video_data = VideoData(
                    output_format=output_format,
                    b64_json=base64.b64encode(video_bytes).decode("utf-8"),
                )
            else:
                url = await upload_to_fs(
                    self._media_fs,
                    f"videos/{request_id}.{output_format}",
                    video_bytes,
                    self._media_http_url,
                )
                video_data = VideoData(output_format=output_format, url=url)
            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=[video_data],
                inference_time_s=time.time() - started,
            ).model_dump()
        except Exception as e:  # noqa: BLE001 -- reported to the client as a status
            logger.error("Failed to encode video for %s: %s", request_id, e)
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

    def failure(self, request_id: str, error: str) -> dict:
        """A terminal failure response for a request that produced no frames."""
        return self._cmaf_failure(request_id, int(time.time()), error)
