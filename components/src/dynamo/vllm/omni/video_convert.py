# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-Omni video frame conversion to the canonical encoder format."""

from typing import Any

import numpy as np
import torch

from dynamo.common.utils.video_utils import ensure_uint8_rgb


def to_canonical(images: list) -> np.ndarray:
    """Convert ``stage_output.images`` to canonical ``(T, H, W, 3) uint8``.

    ``stage_output.images`` is a list holding a single full-video array of shape
    ``(1, T, H, W, C)`` or ``(T, H, W, C)`` (``np.ndarray`` or ``torch.Tensor``).
    """
    array = images[0] if len(images) == 1 else images
    if isinstance(array, torch.Tensor):
        array = array.cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 5:
        array = array[0]
    return ensure_uint8_rgb(array)


def decoded_chunk_to_canonical(chunk: Any) -> np.ndarray:
    """Convert one raw VAE decode chunk to canonical ``(T, H, W, 3) uint8``.

    A pipelined stream reaches the router straight off ``session_decode_step``,
    which returns the decoder's own output: ``[B, C, T, H, W]`` in ``[-1, 1]``. The
    batch path never sees this shape because the engine's post-process func runs
    diffusers' ``VideoProcessor.postprocess_video`` first; bypassing the engine
    means doing that conversion here instead.

    Getting it wrong is quiet rather than loud, which is why it is its own
    function: treating ``[-1, 1]`` as ``[0, 1]`` clips every dark pixel to black
    and halves the contrast of the rest, producing a video that plays fine and
    looks wrong.
    """
    if isinstance(chunk, list):
        chunk = chunk[0] if len(chunk) == 1 else chunk
    if isinstance(chunk, torch.Tensor):
        video = chunk.detach().float().cpu()
        if video.ndim == 5:
            video = video[0]
        if video.ndim != 4:
            raise ValueError(
                f"expected a decoded chunk of [B, C, T, H, W] or [C, T, H, W]; got {tuple(chunk.shape)}"
            )
        # [-1, 1] -> [0, 1], then [C, T, H, W] -> [T, H, W, C].
        video = (video / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
        return ensure_uint8_rgb(video.numpy())

    # Already post-processed (np in [0, 1]) -- e.g. an aggregated pipeline whose
    # post-process func ran before the handoff.
    return to_canonical(chunk if isinstance(chunk, list) else [chunk])
