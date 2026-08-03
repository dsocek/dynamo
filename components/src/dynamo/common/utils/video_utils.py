# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video utilities for video diffusion.

Provides helpers for parsing video request parameters and encoding numpy
video frames to MP4 format.
"""

import asyncio
import io
import logging
import os
import shutil
import subprocess
import tempfile
from typing import AsyncIterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_VIDEO_WIDTH = 832
DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 16
DEFAULT_VIDEO_NUM_FRAMES = 97


def parse_size(
    size: str | None,
    default_w: int = DEFAULT_VIDEO_WIDTH,
    default_h: int = DEFAULT_VIDEO_HEIGHT,
) -> Tuple[int, int]:
    """Parse a 'WxH' string into (width, height).

    Falls back to default_w x default_h when size is None or malformed.
    """
    if not size:
        return default_w, default_h
    try:
        w, h = size.split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        logger.warning("Invalid size format: %s, using defaults", size)
        return default_w, default_h


def compute_num_frames(
    num_frames: int | None = None,
    seconds: int | None = None,
    fps: int | None = None,
    default_fps: int = DEFAULT_VIDEO_FPS,
    default_num_frames: int = DEFAULT_VIDEO_NUM_FRAMES,
) -> int:
    """Compute the number of video frames.

    Priority: num_frames > seconds x fps > default_num_frames.
    """
    if num_frames is not None:
        return num_frames
    if seconds is not None or fps is not None:
        _seconds = seconds if seconds is not None else 4
        _fps = fps if fps is not None else default_fps
        return _seconds * _fps
    return default_num_frames


# ---------------------------------------------------------------------------
# Unified video encoding
#
# A single shared entry point (``encode_video``) that all backends call with a
# canonical frame array -- ``np.ndarray (T, H, W, 3) uint8`` RGB. Each backend
# owns a ``to_canonical()`` converter (next to its handler) that maps its native
# output into this format, composed from the canonical-domain primitives below
# (``ensure_uint8_rgb`` / ``pil_frames_to_array`` / ``drop_alpha``). The encoder
# validates the canonical contract on entry and dispatches to one of two paths:
#
#   * imageio -> ffmpeg (NVENC)  -- preserves the existing NVIDIA behavior.
#   * ffmpeg CLI (raw pipe)      -- new hardware (XPU) / codecs that imageio's
#                                   ffmpeg plugin cannot reach.
#
# Encoding controls are read from DYN_VIDEO_* environment variables for now.
# ---------------------------------------------------------------------------


# Logical codec name -> ffmpeg encoder, per hardware path.
_FFMPEG_ENCODERS = {
    "nvenc": {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "vp9": "libvpx-vp9"},
    "xpu": {"h264": "h264_vaapi", "hevc": "hevc_vaapi", "vp9": "vp9_vaapi"},
    "cpu": {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9"},
}

# Default logical codec per container.
_DEFAULT_CODEC = {"mp4": "h264", "webm": "vp9"}

def normalize_video_frames(images: list) -> list:
    """Normalize stage_output.images into a frame list for export_to_video.

    Args:
        images: stage_output.images -- a list that may contain a single
            torch.Tensor or np.ndarray representing the full video.

    Returns:
        List of frames suitable for diffusers export_to_video.
    """
    frames = images[0] if len(images) == 1 else images

    if isinstance(frames, np.ndarray):
        if frames.ndim == 5:
            frames = frames[0]
        return list(frames)

    return list(frames)


def frames_to_numpy(images: list) -> np.ndarray:
    """Convert a list of PIL Images to a numpy array suitable for video encoding.

    Args:
        images: List of PIL Image objects (video frames).

    Returns:
        Numpy array of shape ``(num_frames, height, width, 3)`` with dtype
        ``uint8`` and values in ``[0, 255]``.

    Raises:
        ValueError: If no images are provided or images have inconsistent sizes.
    """
    if not images:
        raise ValueError("No images provided for video encoding")

    frames = []
    for img in images:
        arr = np.array(img.convert("RGB"))
        frames.append(arr)

    # Validate consistent sizes
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(
            f"Inconsistent frame sizes detected: {shapes}. "
            "All frames must have the same dimensions."
        )

    return np.stack(frames, axis=0)


def encode_to_mp4(
    frames: np.ndarray,
    output_dir: str,
    request_id: str,
    fps: int = 16,
) -> str:
    """Encode numpy frames to MP4 file.

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        output_dir: Directory to save the output video.
        request_id: Unique identifier for the request (used in filename).
        fps: Frames per second for the output video.

    Returns:
        Path to the saved MP4 file.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{request_id}.mp4")

    logger.info(f"Encoding {len(frames)} frames to {output_path} at {fps} fps")

    try:
        # Use imageio to write MP4. We use h264_nvenc (NVIDIA HW encoder) instead
        # of libx264 because the in-tree ffmpeg build is LGPL-only and libx264
        # is GPL-licensed; see container/templates/wheel_builder.Dockerfile.
        # Requires a CUDA-capable GPU at runtime.
        if hasattr(iio, "imwrite"):
            iio.imwrite(output_path, frames, fps=fps, codec="h264_nvenc")
        else:
            # Fall back to v2 API
            writer = iio.get_writer(output_path, fps=fps, codec="h264_nvenc")  # type: ignore[attr-defined]
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        logger.info(f"Video saved to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to encode video: {e}")
        raise RuntimeError(f"Video encoding failed: {e}") from e


def encode_to_video_bytes(
    frames: np.ndarray,
    fps: int = 16,
    output_format: str = "mp4",
) -> bytes:
    """Encode numpy frames to video bytes (in-memory).

    Args:
        frames: Video frames as numpy array of shape (num_frames, height, width, 3)
            with uint8 values 0-255.
        fps: Frames per second for the output video.
        output_format: Container format — "mp4", "webm".

    Returns:
        Encoded video as bytes.

    Raises:
        ImportError: If imageio is not available.
        RuntimeError: If encoding fails.
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            )

    logger.info(f"Encoding {len(frames)} frames to {output_format} bytes at {fps} fps")

    try:
        buffer = io.BytesIO()

        kwargs: dict = {"fps": fps}
        if output_format == "webm":
            kwargs["codec"] = "libvpx-vp9"
        elif output_format == "mp4":
            kwargs["codec"] = "h264_nvenc"
        else:
            raise ValueError(f"No codec specified for response format: {output_format}")

        if hasattr(iio, "imwrite"):
            # v3 API
            iio.imwrite(buffer, frames, extension=f".{output_format}", **kwargs)
        else:
            # v2 API
            writer = iio.get_writer(  # type: ignore[attr-defined]
                buffer, format="FFMPEG", mode="I", **kwargs
            )
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()

        video_bytes = buffer.getvalue()
        logger.info(f"Encoded video to {len(video_bytes)} bytes")
        return video_bytes

    except Exception as e:
        logger.error(f"Failed to encode video to bytes: {e}")
        raise RuntimeError(f"Video encoding to bytes failed: {e}") from e


# Logical codecs each container can legally mux. Used to reject incompatible
# (container, codec) combinations before invoking the encoder.
_CONTAINER_CODECS = {
    "mp4": ("h264", "hevc"),
    "webm": ("vp9",),
}


def _video_codec() -> str | None:
    """Codec override from ``DYN_VIDEO_CODEC`` (e.g. ``h264`` / ``hevc`` / ``vp9``)."""
    val = os.environ.get("DYN_VIDEO_CODEC")
    return val.strip().lower() if val else None


def _video_container() -> str:
    """Container from ``DYN_VIDEO_CONTAINER`` (default ``mp4``)."""
    return os.environ.get("DYN_VIDEO_CONTAINER", "mp4").strip().lower()


def _video_hw_accel() -> str:
    """HW accelerator from ``DYN_VIDEO_HW_ACCEL`` (``auto``/``nvenc``/``xpu``/``cpu``)."""
    return os.environ.get("DYN_VIDEO_HW_ACCEL", "auto").strip().lower()


def _video_device() -> str:
    """DRM render node / device for HW encoding.

    Read from ``DYN_VIDEO_DEVICE``, falling back to the legacy
    ``DYNAMO_VAAPI_DEVICE`` and finally ``/dev/dri/renderD128``.
    """
    return (
        os.environ.get("DYN_VIDEO_DEVICE")
        or os.environ.get("DYNAMO_VAAPI_DEVICE")
        or "/dev/dri/renderD128"
    )


def _running_on_xpu() -> bool:
    """Return True when torch reports an available XPU backend."""
    try:
        import torch

        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False


def _resolve_ffmpeg_encoder(container: str, codec: str | None, hw_accel: str) -> str:
    """Map (container, logical codec, hw path) to an ffmpeg encoder name.

    ``container`` / ``codec`` / ``hw_accel`` are assumed already validated by
    ``encode_video``. Raises ``ValueError`` if no encoder entry exists for the
    resolved (hw_accel, codec) pair -- i.e. the lookup tables are inconsistent.
    """
    codec = codec or _DEFAULT_CODEC[container]
    table = _FFMPEG_ENCODERS[hw_accel]
    encoder = table.get(codec)
    if encoder is None:
        raise ValueError(
            f"No ffmpeg encoder for codec {codec!r} on {hw_accel!r}; "
            f"available: {sorted(table)}"
        )
    return encoder


def _validate_container_codec(container: str, codec: str | None) -> None:
    """Reject container/codec pairs that are not muxing-compatible.

    A ``codec`` of ``None`` means "use the container's default", which is
    compatible by construction. Raises ``ValueError`` for an unknown container
    or a codec the container cannot carry (e.g. H.264 in webm).
    """
    allowed = _CONTAINER_CODECS.get(container)
    if allowed is None:
        raise ValueError(
            f"Unsupported container {container!r}; "
            f"supported: {sorted(_CONTAINER_CODECS)}"
        )
    effective = codec or _DEFAULT_CODEC[container]
    if effective not in allowed:
        raise ValueError(
            f"Codec {effective!r} is not compatible with container "
            f"{container!r}; supported: {list(allowed)}"
        )


def drop_alpha(frames: np.ndarray) -> np.ndarray:
    """Drop a trailing alpha channel (RGBA -> RGB) when present."""
    if frames.shape[-1] == 4:
        return frames[..., :3]
    return frames


def ensure_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """Normalize an RGB frame array to contiguous ``(T, H, W, 3) uint8``.

    Drops a trailing alpha channel and scales floating-point values in
    ``[0, 1]`` up to ``[0, 255]``. Channel order and axis layout are assumed to
    be RGB / ``(T, H, W, C)`` already; this operates purely in the canonical
    domain and carries no backend-specific knowledge.
    """
    frames = drop_alpha(frames)
    if np.issubdtype(frames.dtype, np.floating):
        frames = np.clip(frames * 255.0, 0, 255).round()
    return np.ascontiguousarray(frames, dtype=np.uint8)


def pil_frames_to_array(frames) -> np.ndarray:
    """Stack a list of per-frame images into a single ``(T, H, W, C)`` array.

    Each element may be a ``PIL.Image`` or an ``np.ndarray``; PIL images are
    converted to RGB numpy arrays first.
    """
    per_frame = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            per_frame.append(frame)
        else:
            per_frame.append(np.array(frame.convert("RGB")))
    return np.stack(per_frame, axis=0)


def _validate_canonical_frames(frames) -> None:
    """Validate the canonical encoder input contract.

    Raises ``ValueError`` unless ``frames`` is an ``np.ndarray`` of shape
    ``(T, H, W, 3)`` with dtype ``uint8``.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError(
            "encode_video expects canonical frames as np.ndarray (T, H, W, 3) "
            f"uint8; got {type(frames).__name__}. Convert backend output with "
            "the backend's to_canonical() first."
        )
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"encode_video expects shape (T, H, W, 3); got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"encode_video expects dtype uint8; got {frames.dtype}")


def _encode_video_imageio(
    frames: np.ndarray, fps: int, container: str, codec: str | None
) -> bytes:
    """Encode via imageio -> ffmpeg (NVENC path). Preserves existing behavior."""
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError as err:
            raise ImportError(
                "imageio is required for video encoding. "
                "Install with: pip install imageio[ffmpeg]"
            ) from err

    encoder = _resolve_ffmpeg_encoder(container, codec, "nvenc")
    buffer = io.BytesIO()
    if hasattr(iio, "imwrite"):
        iio.imwrite(buffer, frames, extension=f".{container}", fps=fps, codec=encoder)
    else:
        writer = iio.get_writer(  # type: ignore[attr-defined]
            buffer, format="FFMPEG", mode="I", fps=fps, codec=encoder
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
    return buffer.getvalue()


def _encode_video_ffmpeg_cli(
    frames: np.ndarray,
    fps: int,
    container: str,
    codec: str | None,
    hw_accel: str,
    device: str,
    gop_seconds: int | None = None,
    fragmented: bool = False,
) -> bytes:
    """Encode by piping raw RGB frames to the ffmpeg CLI (XPU / new codecs).

    mp4 needs a seekable output for the moov atom, so we encode to a temp file
    and read the bytes back.

    When ``fragmented`` is set, emit a fragmented (CMAF-compatible) MP4 -- an
    ``empty_moov`` initialization plus one ``moof``+``mdat`` per keyframe -- and
    force keyframes every ``gop_seconds`` so the stream can be split into
    independently decodable media segments. This is the single controlled
    encoder path; callers fragment the result with :func:`split_fragmented_mp4`.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH; required for video encoding")

    num_frames, height, width, _ = frames.shape
    encoder = _resolve_ffmpeg_encoder(container, codec, hw_accel)

    with tempfile.NamedTemporaryFile(suffix=f".{container}", delete=False) as tmp:
        output_path = tmp.name
    try:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if hw_accel == "xpu":
            cmd += ["-vaapi_device", device]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if hw_accel == "xpu":
            cmd += ["-vf", "format=nv12,hwupload"]
        cmd += ["-c:v", encoder]
        if hw_accel == "cpu":
            # Software encoders default to yuv444p from rgb24 input, which most
            # players (browsers, mobile) cannot decode; force widely-compatible
            # 4:2:0. The XPU path already gets 4:2:0 via the nv12 hwupload above.
            cmd += ["-pix_fmt", "yuv420p"]
        if fragmented:
            # Force IDR keyframes at segment boundaries so each fragment is
            # independently decodable. The HW encoder may not honor this
            # exactly (driver GOP constraints), which is why callers must parse
            # the produced bitstream rather than assume the requested layout.
            gop = max(1, (gop_seconds or 1) * fps)
            cmd += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
            if gop_seconds:
                cmd += [
                    "-force_key_frames",
                    f"expr:gte(t,n_forced*{gop_seconds})",
                ]
        if container == "mp4":
            if fragmented:
                # Fragmented CMAF MP4: empty moov init + one moof+mdat per
                # keyframe. Mutually exclusive with +faststart.
                cmd += [
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                ]
            else:
                cmd += ["-movflags", "+faststart"]
        cmd += ["-f", container, output_path]

        logger.info(
            "Encoding %d frames (%dx%d @ %d fps) via %s on %s (fragmented=%s)",
            num_frames,
            width,
            height,
            fps,
            encoder,
            device if hw_accel == "xpu" else hw_accel,
            fragmented,
        )
        proc = subprocess.run(
            cmd,
            input=frames.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg {encoder} encode failed (exit {proc.returncode}): {stderr}"
            )
        with open(output_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def encode_video(
    frames: np.ndarray,
    fps: int = DEFAULT_VIDEO_FPS,
    *,
    container: str | None = None,
    codec: str | None = None,
    hw_accel: str | None = None,
    device: str | None = None,
    gop_seconds: int | None = None,
    fragmented: bool = False,
) -> bytes:
    """Unified video encoder: encode canonical frames to video bytes.

    ``frames`` must already be in the canonical format -- an ``np.ndarray`` of
    shape ``(T, H, W, 3)``, dtype ``uint8``, RGB. Each backend converts its
    native output with its own ``to_canonical()`` before calling this.

    Any explicit argument overrides its ``DYN_VIDEO_*`` environment variable,
    which in turn overrides platform auto-detection. Dispatches to the imageio
    NVENC path on NVIDIA, or the ffmpeg-CLI path for XPU / CPU.

    Args:
        frames: Canonical ``np.ndarray (T, H, W, 3) uint8`` RGB frames.
        fps: Frames per second for the output video.
        container: ``mp4`` / ``webm`` (env: ``DYN_VIDEO_CONTAINER``).
        codec: Logical codec ``h264`` / ``hevc`` / ``vp9`` (env: ``DYN_VIDEO_CODEC``).
        hw_accel: ``auto`` / ``nvenc`` / ``xpu`` / ``cpu`` (env: ``DYN_VIDEO_HW_ACCEL``).
            ``auto`` picks a hardware encoder (XPU if present, else NVENC) and
            never selects CPU -- request ``cpu`` explicitly for software encoding.
        device: HW device / DRM render node (env: ``DYN_VIDEO_DEVICE``).
        gop_seconds: When ``fragmented``, force a keyframe every this many
            seconds so the stream splits into ~this-long media segments.
        fragmented: Emit a fragmented (CMAF-compatible) MP4 instead of a
            monolithic one. The returned bytes are still a single valid MP4;
            split them into init + media segments with
            :func:`split_fragmented_mp4`. Forces the ffmpeg-CLI path for every
            accelerator (the imageio/NVENC path cannot emit fragmented MP4).

    Returns:
        Encoded video as bytes.

    Raises:
        ValueError: If ``frames`` is not the canonical ``(T, H, W, 3) uint8`` array.
    """
    _validate_canonical_frames(frames)

    container = (container or _video_container()).lower()
    codec = codec.lower() if codec else _video_codec()
    _validate_container_codec(container, codec)

    hw_accel = (hw_accel or _video_hw_accel()).lower()
    if hw_accel == "auto":
        # "auto" selects a hardware encoder only: XPU when present, otherwise
        # NVENC. It never resolves to CPU -- software encoding is slow and must
        # be requested explicitly (hw_accel="cpu" / DYN_VIDEO_HW_ACCEL=cpu). A
        # deployment with neither NVIDIA nor XPU is unsupported by design.
        hw_accel = "xpu" if _running_on_xpu() else "nvenc"
    if hw_accel not in _FFMPEG_ENCODERS:
        raise ValueError(
            f"Unsupported hw_accel {hw_accel!r}; "
            f"supported: {sorted(_FFMPEG_ENCODERS)} (or 'auto')"
        )

    logger.info(
        "Encoding %d frames -> %s (hw=%s codec=%s fragmented=%s) at %d fps",
        len(frames),
        container,
        hw_accel,
        codec or "default",
        fragmented,
        fps,
    )

    # The imageio/NVENC path cannot emit fragmented MP4; fragmented requests go
    # through the ffmpeg CLI for every accelerator (NVENC via h264_nvenc).
    if hw_accel == "nvenc" and not fragmented:
        return _encode_video_imageio(frames, fps, container, codec)

    device = device or _video_device()
    return _encode_video_ffmpeg_cli(
        frames,
        fps,
        container,
        codec,
        hw_accel,
        device,
        gop_seconds=gop_seconds,
        fragmented=fragmented,
    )


# ---------------------------------------------------------------------------
# Fragmented-MP4 (CMAF) helpers
#
# These are pure ISO-BMFF container operations -- independent of which HW
# encoder produced the stream -- so they live next to the encoder rather than
# being duplicated per backend.
# ---------------------------------------------------------------------------


def _iter_top_level_boxes(data: bytes):
    """Yield ``(offset, size, type)`` for each top-level ISO-BMFF box."""
    off = 0
    n = len(data)
    while off + 8 <= n:
        size = int.from_bytes(data[off : off + 4], "big")
        btype = data[off + 4 : off + 8]
        header = 8
        if size == 1:
            if off + 16 > n:
                break
            size = int.from_bytes(data[off + 8 : off + 16], "big")
            header = 16
        elif size == 0:
            size = n - off
        if size < header or off + size > n:
            break
        yield off, size, btype
        off += size


def split_fragmented_mp4(data: bytes) -> Tuple[bytes, list[bytes]]:
    """Split a fragmented (fMP4/CMAF) MP4 into ``(init, [media_segments])``.

    The initialization segment is everything up to and including the ``moov``
    box (typically ``ftyp`` + ``moov``). Each media segment is a ``moof`` box
    with its following boxes (``mdat``) up to the next ``moof``. Pure container
    parsing -- encoder-agnostic. A stream with a single ``moof`` yields one
    segment, which CMAF handles fine.

    Raises:
        RuntimeError: If the stream has no ``moov`` or no ``moof`` box.
    """
    moov_end: int | None = None
    moof_starts: list[int] = []
    for off, size, btype in _iter_top_level_boxes(data):
        if btype == b"moov":
            moov_end = off + size
        elif btype == b"moof":
            moof_starts.append(off)

    if moov_end is None:
        raise RuntimeError("Fragmented MP4 has no moov box; cannot build init segment")
    if not moof_starts:
        raise RuntimeError("Fragmented MP4 has no moof boxes; no media segments")

    init = data[:moov_end]
    boundaries = [start for start in moof_starts if start >= moov_end]
    boundaries.append(len(data))
    segments = [
        data[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)
    ]
    return init, segments


def h264_codec_string_from_init(init: bytes) -> str | None:
    """Derive the RFC 6381 ``avc1.PPCCLL`` codec string from an fMP4 init segment.

    Reads the profile / compatibility / level bytes from the
    AVCDecoderConfigurationRecord (``avcC`` box) so the codec string advertised
    to Media Source Extensions matches the encoded bitstream exactly (HW
    encoders may not honor a requested profile). Returns ``None`` if no ``avcC``
    box is present (e.g. a non-H.264 codec).
    """
    idx = init.find(b"avcC")
    if idx == -1:
        return None
    # avcC payload immediately follows the 4-byte box type:
    #   configurationVersion(1) AVCProfileIndication(1)
    #   profile_compatibility(1) AVCLevelIndication(1) ...
    payload = init[idx + 4 : idx + 8]
    if len(payload) < 4:
        return None
    profile, compat, level = payload[1], payload[2], payload[3]
    return f"avc1.{profile:02x}{compat:02x}{level:02x}"


def _next_complete_box(buf: bytearray) -> Optional[Tuple[int, bytes]]:
    """Peek the first top-level ISO-BMFF box in ``buf`` if it is fully present.

    Returns ``(size, btype)`` when at least ``size`` bytes are buffered, else
    ``None`` (need more data). The incremental counterpart to
    :func:`_iter_top_level_boxes`, which requires the whole stream up front.
    """
    if len(buf) < 8:
        return None
    size = int.from_bytes(buf[0:4], "big")
    btype = bytes(buf[4:8])
    header = 8
    if size == 1:
        if len(buf) < 16:
            return None
        size = int.from_bytes(buf[8:16], "big")
        header = 16
    elif size == 0:
        # "to end of stream" -- never valid for the moof/mdat/ftyp/moov boxes a
        # fragmented muxer emits over a pipe; treat as not-yet-cuttable.
        return None
    if size < header or len(buf) < size:
        return None
    return size, btype


# ---------------------------------------------------------------------------
# Live per-request CMAF encoding (persistent ffmpeg session)
#
# ``StreamingCmafEncoder`` owns ONE long-lived ffmpeg process for the whole
# presentation, so the muxer itself owns ``mfhd.sequence_number`` /
# ``tfdt.baseMediaDecodeTime`` and there is exactly one init segment -- no box
# surgery. Frames are pushed incrementally; completed CMAF boxes are yielded as
# they flush. Protocol-neutral: yields ``("init"|"segment", bytes)`` so this
# layer carries no wire-tag knowledge (that stays in ``cmaf_video``).
# ---------------------------------------------------------------------------


class StreamingCmafEncoder:
    """Persistent-ffmpeg fragmented-MP4 encoder for live CMAF streaming.

    Lifecycle: :meth:`start` (spawn ffmpeg) -> :meth:`push` per pixel chunk
    (yields any boxes that flushed) -> :meth:`finish` (close stdin, drain the
    tail). A background reader task pumps ffmpeg stdout through an incremental
    box cutter onto a queue; ``push``/``finish`` yield whatever the cutter has
    completed so far.

    Yields ``(kind, payload)`` where ``kind`` is ``"init"`` (once) or
    ``"segment"`` (per media fragment). Callers map those to wire tags.
    """

    def __init__(
        self,
        fps: int,
        width: int,
        height: int,
        *,
        gop_frames: int,
        hw_accel: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self.gop_frames = max(1, gop_frames)
        hw = (hw_accel or _video_hw_accel()).lower()
        if hw == "auto":
            hw = "xpu" if _running_on_xpu() else "nvenc"
        self.hw_accel = hw
        self.device = device or _video_device()

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._buf = bytearray()
        # Init is everything up to and including ``moov``; accumulate until then.
        self._init_boxes: list[bytes] = []
        self._init_emitted = False
        self._init_bytes: Optional[bytes] = None
        self._codec_string: Optional[str] = None
        # Current media fragment being assembled (a ``moof`` + following boxes).
        self._seg_accum = bytearray()
        self._stderr = bytearray()

    def _build_argv(self, ffmpeg: str) -> list[str]:
        encoder = _resolve_ffmpeg_encoder("mp4", "h264", self.hw_accel)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if self.hw_accel == "xpu":
            cmd += ["-vaapi_device", self.device]
        cmd += [
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
        ]
        if self.hw_accel == "xpu":
            cmd += ["-vf", "format=nv12,hwupload"]
        cmd += ["-c:v", encoder]
        if self.hw_accel == "cpu":
            # rgb24 -> yuv420p for broad decoder support; closed GOP so each
            # fragment is independently decodable (VA-API is closed-GOP already).
            cmd += ["-pix_fmt", "yuv420p", "-flags", "+cgop"]
        # Deterministic keyframe every gop_frames, no B-frames (no reordering /
        # cross-fragment prediction), fragment cut at each keyframe -> every
        # emitted moof begins with an IDR by construction.
        cmd += [
            "-bf", "0",
            "-g", str(self.gop_frames),
            "-keyint_min", str(self.gop_frames),
            "-sc_threshold", "0",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-flush_packets", "1",
            "-f", "mp4", "pipe:1",
        ]
        return cmd

    async def start(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH; required for CMAF streaming")
        argv = self._build_argv(ffmpeg)
        logger.info(
            "StreamingCmafEncoder: %dx%d @ %d fps, gop=%d, hw=%s",
            self.width, self.height, self.fps, self.gop_frames, self.hw_accel,
        )
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    async def _read_stdout(self) -> None:
        """Pump ffmpeg stdout -> incremental cutter -> queue until EOF."""
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            chunk = await self._proc.stdout.read(65536)
            if not chunk:
                break
            self._buf.extend(chunk)
            self._cut_ready_boxes()
        # EOF: the last fragment has no following moof to close it -- flush it.
        self._flush_pending_segment()
        await self._queue.put(None)  # sentinel: no more boxes

    async def _read_stderr(self) -> None:
        """Drain stderr so a chatty ffmpeg can never block on a full pipe."""
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            chunk = await self._proc.stderr.read(65536)
            if not chunk:
                break
            self._stderr.extend(chunk)

    def _cut_ready_boxes(self) -> None:
        while True:
            parsed = _next_complete_box(self._buf)
            if parsed is None:
                return
            size, btype = parsed
            box = bytes(self._buf[:size])
            del self._buf[:size]
            self._handle_box(btype, box)

    def _handle_box(self, btype: bytes, box: bytes) -> None:
        if not self._init_emitted:
            self._init_boxes.append(box)
            if btype == b"moov":
                init = b"".join(self._init_boxes)
                self._init_bytes = init
                self._codec_string = h264_codec_string_from_init(init)
                self._init_boxes = []
                self._init_emitted = True
                self._queue.put_nowait(("init", init))
            return
        # Post-init boxes: a ``moof`` opens a new fragment (flush the previous),
        # everything else (``mdat``, ...) belongs to the fragment in progress.
        if btype == b"moof":
            self._flush_pending_segment()
            self._seg_accum = bytearray(box)
        else:
            self._seg_accum.extend(box)

    def _flush_pending_segment(self) -> None:
        if self._seg_accum:
            self._queue.put_nowait(("segment", bytes(self._seg_accum)))
            self._seg_accum = bytearray()

    def _drain_ready(self) -> list[Tuple[str, bytes]]:
        """Pop every box the cutter has completed so far (non-blocking)."""
        items: list[Tuple[str, bytes]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:  # sentinel only expected in finish()
                break
            items.append(item)
        return items

    async def push(self, frames: np.ndarray) -> AsyncIterator[Tuple[str, bytes]]:
        """Feed one pixel chunk; yield any CMAF boxes that have since flushed.

        ``frames`` is canonical ``(t, H, W, 3)`` uint8 RGB. Because a fragmented
        muxer finalizes fragment *k* only when fragment *k+1*'s first frame
        arrives, boxes returned here typically lag the frames just written by
        ~one fragment; the tail comes out in :meth:`finish`.
        """
        assert self._proc is not None and self._proc.stdin is not None
        _validate_canonical_frames(frames)
        self._proc.stdin.write(frames.tobytes())
        await self._proc.stdin.drain()
        for item in self._drain_ready():
            yield item

    async def finish(self) -> AsyncIterator[Tuple[str, bytes]]:
        """Close stdin and drain the remaining fragment(s) to end-of-stream."""
        assert self._proc is not None
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        # Consume boxes until the reader signals EOF with its sentinel.
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
        await self._proc.wait()
        if self._stderr_task is not None:
            await self._stderr_task
        if self._proc.returncode:
            err = bytes(self._stderr).decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg CMAF session failed (exit {self._proc.returncode}): {err}"
            )

    def codec_string(self) -> Optional[str]:
        """RFC 6381 ``avc1.PPCCLL`` parsed from the init segment, else ``None``.

        Available only after the init box has flushed (i.e. after the first
        ``push`` that produces it). Callers apply their own fallback when
        ``None`` (the protocol layer owns the fallback constant).
        """
        return self._codec_string
