# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary CMAF packaging for the vLLM-Omni video path.

The Dynamo HTTP frontend exposes ``POST /v1/videos/stream/binary/cmaf``. That
route injects the ``experimental_binary_cmaf`` annotation into the request and
re-frames a stream of :class:`NvVideosResponse` items into a single binary CMAF
response body. Each response item carries a *tag* in ``VideoData.url`` and a
base64 payload in ``VideoData.b64_json``:

    cmaf:metadata   -> JSON describing the asset (protocol, codecs, segment_count)
    cmaf:init       -> the fMP4 initialization segment
    cmaf:segment:N  -> the Nth fMP4 media segment

Wan2.1-T2V (and diffusion video generation generally) produces the *whole* clip
in one call. The pipeline is therefore ``diffuser -> encode_video -> split ->
stream``: the shared unified encoder
(:func:`dynamo.common.utils.video_utils.encode_video`) does the hardware
encoding, asked for a fragmented/CMAF-compatible MP4, and the encoder-agnostic
:func:`split_fragmented_mp4` chops that single stream into an init segment plus
media segments. This module only owns the *protocol* (tags, metadata, request
opt-in); it contains no encoder- or hardware-specific logic.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from dynamo.common.utils.video_utils import (
    DEFAULT_AUDIO_FRAG_SECONDS,
    audio_codec_string,
    encode_audio_fragments,
    encode_video,
    encode_video_fragments,
    h264_codec_string_from_init,
    split_fragmented_mp4,
)

logger = logging.getLogger(__name__)

# Wire protocol constants -- must match the Rust frontend
# (lib/llm/src/http/service/openai.rs).
CMAF_ANNOTATION = "experimental_binary_cmaf"
CMAF_PROTOCOL = "dynamo-video-binary-cmaf-v1"
CMAF_METADATA_TAG = "cmaf:metadata"
CMAF_INIT_TAG = "cmaf:init"
CMAF_SEGMENT_PREFIX = "cmaf:segment:"

# Audio is a *second, independent* CMAF track, not muxed into the video one: in
# fMP4 each track is fragmented separately, so nothing has to align, and the
# muxer never has to hold a video fragment back waiting for audio.
CMAF_AUDIO_INIT_TAG = "cmaf:audio:init"
CMAF_AUDIO_SEGMENT_PREFIX = "cmaf:audio:segment:"
# A non-fatal audio failure. Distinct from a failed NvVideosResponse, which
# aborts the whole stream at the frontend: when only audio dies the video is
# still good, so this rides through as an ordinary payload item.
CMAF_AUDIO_ERROR_TAG = "cmaf:audio:error"

# Audio codec for the second track. AAC-LC because AAC-in-fMP4 is the one
# combination every MSE implementation supports; Opus-in-MP4 is spottier.
CMAF_AUDIO_CODEC = "aac"

# ``segment_count`` value advertised when the total is not yet known because
# segments are still being produced. 0 is used rather than null so the field
# stays a number for clients that predate open-ended streams; they will simply
# append nothing, which is safe. New clients branch on ``open_ended``.
CMAF_OPEN_ENDED = 0

# Fallback H.264 codec string (Main profile, level 3.1) advertised only when the
# real profile/level cannot be parsed from the encoded init segment.
CMAF_FALLBACK_VIDEO_CODEC = "avc1.4d401f"

_DEFAULT_SEGMENT_SECONDS = 2
_DEFAULT_EMIT_CADENCE_MS = 0
_DEFAULT_GOP_FRAMES = 4
# One audio fragment: the smallest lead that keeps audio from being the track
# that starves. See cmaf_audio_lead_seconds().
_DEFAULT_AUDIO_LEAD_SECONDS = DEFAULT_AUDIO_FRAG_SECONDS


def has_cmaf_annotation(nvext) -> bool:
    """Return True when the request opts in to binary CMAF streaming.

    ``nvext`` is a ``VideoNvExt`` (or None). The frontend's binary-CMAF route
    appends :data:`CMAF_ANNOTATION` to ``nvext.annotations`` before dispatch.
    """
    return bool(
        nvext is not None
        and nvext.annotations
        and CMAF_ANNOTATION in nvext.annotations
    )


def cmaf_segment_seconds() -> int:
    """Target CMAF fragment duration (env: ``DYN_CMAF_SEGMENT_SECONDS``)."""
    raw = os.environ.get("DYN_CMAF_SEGMENT_SECONDS")
    if not raw:
        return _DEFAULT_SEGMENT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid DYN_CMAF_SEGMENT_SECONDS=%r; using %d",
            raw,
            _DEFAULT_SEGMENT_SECONDS,
        )
        return _DEFAULT_SEGMENT_SECONDS
    return max(1, value)


def cmaf_emit_cadence_s() -> float:
    """Optional delay between emitted media segments (env: ``DYN_CMAF_EMIT_CADENCE_MS``).

    Defaults to 0 (emit as fast as the client can consume). A positive value
    paces the segment stream, which can smooth out client-side buffering during
    demos.
    """
    raw = os.environ.get("DYN_CMAF_EMIT_CADENCE_MS")
    if not raw:
        return _DEFAULT_EMIT_CADENCE_MS / 1000.0
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid DYN_CMAF_EMIT_CADENCE_MS=%r; using 0", raw)
        return 0.0
    return max(0, value) / 1000.0


def cmaf_gop_frames() -> int:
    """Encoder GOP / fragment length in frames (env: ``DYN_CMAF_GOP_FRAMES``).

    For the live persistent-ffmpeg path, keyframes (and therefore fragment
    boundaries) land every this many frames. Small values reduce latency; the
    one-fragment flush lag is intrinsic to fragmented MP4.
    """
    raw = os.environ.get("DYN_CMAF_GOP_FRAMES")
    if not raw:
        return _DEFAULT_GOP_FRAMES
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid DYN_CMAF_GOP_FRAMES=%r; using %d", raw, _DEFAULT_GOP_FRAMES)
        return _DEFAULT_GOP_FRAMES
    return max(1, value)


def cmaf_audio_frag_seconds() -> float:
    """Target audio fragment duration (env: ``DYN_CMAF_AUDIO_FRAG_SECONDS``).

    Approximate by nature -- the muxer cuts on codec frame boundaries -- so this
    is a request, not a guarantee. Nothing may depend on it being exact.
    """
    raw = os.environ.get("DYN_CMAF_AUDIO_FRAG_SECONDS")
    if not raw:
        return DEFAULT_AUDIO_FRAG_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid DYN_CMAF_AUDIO_FRAG_SECONDS=%r; using %.3f",
            raw,
            DEFAULT_AUDIO_FRAG_SECONDS,
        )
        return DEFAULT_AUDIO_FRAG_SECONDS
    return max(0.05, value)


def cmaf_audio_delay_s() -> float:
    """Artificial per-fragment audio delay (env: ``DYN_CMAF_AUDIO_DELAY_MS``).

    Stands in for an audio generator that is slower than realtime. A file source
    outruns realtime by orders of magnitude, so without this the "audio lags
    behind video" case -- the one that stalls MSE playback, and the one a real
    audio worker will actually produce -- is unreachable in testing.

    Defaults to 0: no delay, audio paced only by the caller's watermark.
    """
    raw = os.environ.get("DYN_CMAF_AUDIO_DELAY_MS")
    if not raw:
        return 0.0
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid DYN_CMAF_AUDIO_DELAY_MS=%r; using 0", raw)
        return 0.0
    return max(0, value) / 1000.0


def cmaf_audio_lead_seconds() -> float:
    """How far audio may run ahead of video on the wire (env: ``DYN_CMAF_AUDIO_LEAD_SECONDS``).

    A lead exists because of an asymmetry in MSE: playback stalls if *any* active
    ``SourceBuffer`` lacks data at ``currentTime``, so audio starving is as fatal
    as video starving while being far cheaper to prevent. Keeping audio slightly
    ahead makes video the only track that can ever be the bottleneck.
    """
    raw = os.environ.get("DYN_CMAF_AUDIO_LEAD_SECONDS")
    if not raw:
        return _DEFAULT_AUDIO_LEAD_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid DYN_CMAF_AUDIO_LEAD_SECONDS=%r; using %.3f",
            raw,
            _DEFAULT_AUDIO_LEAD_SECONDS,
        )
        return _DEFAULT_AUDIO_LEAD_SECONDS
    return max(0.0, value)


def resolve_cmaf_audio_file() -> str | None:
    """Locate the POC audio bed, or return None to stream video-only.

    Resolution order: ``CF_POC_AUDIO_FILE`` if set, else the example's bundled
    mp3 if it happens to be present. Returning None rather than raising is the
    whole enablement rule -- audio is added when a file can be found and silently
    skipped when it cannot, so no client-facing opt-in is needed for the POC.

    Note this is a read on the **router host**, unlike the example's client-side
    music bed which the proxy serves over HTTP.
    """
    override = os.environ.get("CF_POC_AUDIO_FILE")
    if override:
        if Path(override).is_file():
            return override
        # Explicitly asked for and missing is worth a complaint; the default
        # simply not being there is not.
        logger.warning(
            "CF_POC_AUDIO_FILE=%r is not a readable file; streaming video-only",
            override,
        )
        return None

    default = (
        Path(__file__).resolve().parents[5]
        / "examples"
        / "custom_backend"
        / "cmaf_binary_video_streaming"
        / "underwater_theme.mp3"
    )
    return str(default) if default.is_file() else None


def source_buffer_mime_type(video_codec: str) -> str:
    """MSE ``SourceBuffer`` mime type for the packaged video-only asset."""
    return f'video/mp4; codecs="{video_codec}"'


def audio_source_buffer_mime_type(audio_codec: str) -> str:
    """MSE ``SourceBuffer`` mime type for the separate audio track."""
    return f'audio/mp4; codecs="{audio_codec}"'


def metadata_bytes(
    segment_count: int | None,
    target_duration_seconds: int,
    video_codec: str,
    audio_codec: str | None = None,
) -> bytes:
    """Serialize the ``cmaf:metadata`` payload (JSON, UTF-8).

    ``segment_count`` may be ``None``, meaning *open-ended*: the segments are
    being produced as the video is generated, so how many there will be is not
    yet known. A client must then read until the stream ends rather than loop to
    a count -- see :data:`CMAF_OPEN_ENDED`. Both live paths pass ``None``: the
    persistent-ffmpeg encoder and the fragment generator alike learn the total
    only when the stream ends.

    ``audio_codec`` set means a second, independent audio track will follow on the
    same wire. It must be decided *before* this frame is sent and not revised
    afterwards: the client creates both ``SourceBuffer``s from this payload, and
    browsers are unreliable about adding one later, once the first holds data.
    """
    payload = {
        "protocol": CMAF_PROTOCOL,
        "mime_type": "video/mp4",
        "source_buffer_mime_type": source_buffer_mime_type(video_codec),
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "audio_source_buffer_mime_type": (
            audio_source_buffer_mime_type(audio_codec) if audio_codec else None
        ),
        "has_audio": audio_codec is not None,
        "target_duration_seconds": target_duration_seconds,
        # Kept for clients written against the fixed-count metadata. 0 rather
        # than null because the field is typed as a number there; a client that
        # understands open-ended streams should branch on `open_ended`.
        "segment_count": CMAF_OPEN_ENDED if segment_count is None else segment_count,
        "open_ended": segment_count is None,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def audio_error_bytes(message: str) -> bytes:
    """Serialize the ``cmaf:audio:error`` payload (JSON, UTF-8).

    Same ``{"error": ...}`` envelope the frontend uses for its own error frames,
    so a client needs one handler for both, plus a ``track`` field naming the
    casualty. The field is load-bearing: both errors arrive as wire kind
    ``0x04``, and a client that cannot tell them apart cannot know to drop the
    audio ``SourceBuffer`` -- which it must, or playback stalls at the point
    audio stopped even though the video track is complete.
    """
    return json.dumps(
        {"error": message, "track": "audio"}, separators=(",", ":")
    ).encode("utf-8")


def package_frames_to_cmaf(
    frames: np.ndarray, fps: int, segment_seconds: int
) -> tuple[bytes, list[bytes], int, str]:
    """Encode canonical frames and split them into CMAF init + media segments.

    Thin orchestration over the shared unified encoder: ask
    :func:`encode_video` for a fragmented (CMAF-compatible) H.264/MP4 stream,
    then split it with :func:`split_fragmented_mp4`. No encoder- or
    hardware-specific logic lives here.

    Args:
        frames: Canonical ``np.ndarray (T, H, W, 3)`` uint8 RGB frames.
        fps: Frames per second of the generated clip.
        segment_seconds: Target duration of each CMAF media fragment.

    Returns:
        ``(init_bytes, segment_bytes_list, target_duration_seconds, video_codec)``.
        ``video_codec`` is parsed from the encoded bitstream so the advertised
        codec matches reality.

    Raises:
        RuntimeError: If ffmpeg is missing or encoding/fragmentation fails.
        ValueError: If ``frames`` is not canonical ``(T, H, W, 3)`` uint8.
    """
    mp4_bytes = encode_video(
        frames,
        fps,
        container="mp4",
        codec="h264",
        gop_seconds=segment_seconds,
        fragmented=True,
    )
    init_bytes, segments = split_fragmented_mp4(mp4_bytes)

    video_codec = h264_codec_string_from_init(init_bytes) or CMAF_FALLBACK_VIDEO_CODEC

    # TODO: parse the real per-fragment duration from each moof (tfhd default
    # sample duration / trun sample durations, scaled by the mdhd timescale in
    # the init segment) instead of approximating from the requested segment
    # length. HW encoders may not honor the requested GOP exactly, so the actual
    # fragment durations can differ.
    target_duration = segment_seconds

    logger.info(
        "CMAF packaging produced init (%d bytes) + %d segments, codec=%s",
        len(init_bytes),
        len(segments),
        video_codec,
    )
    return init_bytes, segments, target_duration, video_codec


def stream_frames_to_cmaf(
    frame_chunks,
    fps: int,
    segment_seconds: int,
    *,
    width: int | None = None,
    height: int | None = None,
):
    """Incremental :func:`package_frames_to_cmaf`: yield CMAF pieces as they encode.

    Same protocol, different latency profile. :func:`package_frames_to_cmaf`
    cannot emit anything until the whole clip is encoded, because
    :func:`encode_video` muxes to a temp file. This drives
    :func:`encode_video_fragments` instead, which muxes to a pipe, so a media
    segment leaves for the client as soon as the encoder closes it.

    The codec string is parsed from the init segment, which arrives before any
    media -- so the metadata frame can still advertise the true profile/level
    rather than a guess, exactly as the batch path does.

    Args:
        frame_chunks: Iterable of canonical ``(T, H, W, 3) uint8`` RGB arrays.
            Chunk boundaries are independent of segment boundaries.
        fps: Frames per second of the generated clip.
        segment_seconds: Target duration of each media fragment.
        width / height: Passed through to the encoder; inferred when omitted.

    Yields:
        ``("init", bytes, codec_string)`` once, then ``("segment", bytes, index)``
        per fragment. The caller owns the tag/metadata framing.

    Raises:
        RuntimeError: If encoding fails or produces no init segment.
    """
    index = 0
    for kind, payload in encode_video_fragments(
        frame_chunks,
        fps,
        codec="h264",
        gop_seconds=segment_seconds,
        width=width,
        height=height,
    ):
        if kind == "init":
            codec = h264_codec_string_from_init(payload) or CMAF_FALLBACK_VIDEO_CODEC
            logger.info(
                "CMAF stream opened: init (%d bytes), codec=%s", len(payload), codec
            )
            yield ("init", payload, codec)
        else:
            yield ("segment", payload, index)
            index += 1


def stream_audio_file_to_cmaf(
    source: str,
    duration_s: float,
    *,
    frag_seconds: float | None = None,
    delay_s: float | None = None,
    start_s: float = 0.0,
):
    """:func:`stream_frames_to_cmaf` for the audio track: same tuple protocol.

    Deliberately the same ``(kind, payload, extra)`` shape as the video generator
    so a caller merging the two tracks needs no per-track special-casing beyond
    which tag it stamps on the result.

    Lazy, like its video sibling: ffmpeg does not start until the first ``next()``,
    which is what lets the consumer's pacing decide when audio is produced.

    Args:
        source: Path to an ffmpeg-readable audio file, on this host.
        duration_s: Output length; the track is trimmed or silence-padded to it.
        frag_seconds / delay_s: Default to the env-configured values.
        start_s: Offset into the source to start from. A chained-scene client gets
            a continuous bed across requests by passing the timeline position it
            has already consumed; the fragments still start at timestamp 0, so
            positioning them stays the consumer's job.

    Yields:
        ``("init", bytes, codec_string)`` once, then ``("segment", bytes, index)``
        per fragment.
    """
    index = 0
    for kind, payload in encode_audio_fragments(
        source,
        duration_s,
        codec=CMAF_AUDIO_CODEC,
        frag_seconds=(
            cmaf_audio_frag_seconds() if frag_seconds is None else frag_seconds
        ),
        delay_s=cmaf_audio_delay_s() if delay_s is None else delay_s,
        start_s=start_s,
    ):
        if kind == "init":
            codec = audio_codec_string(CMAF_AUDIO_CODEC)
            logger.info(
                "CMAF audio track opened: init (%d bytes), codec=%s",
                len(payload),
                codec,
            )
            yield ("init", payload, codec)
        else:
            yield ("segment", payload, index)
            index += 1
