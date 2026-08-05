# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pipelined-stream handoff primitives.

The properties worth pinning are the ones whose absence is silent. A handoff that
loses a terminal error looks exactly like a stream that ended early, and for video
"ended early" is a plausible outcome -- so the failure surfaces as a short clip
rather than as an error, days later, in someone's player.
"""

import asyncio

import numpy as np
import pytest
import torch
from dynamo.vllm.omni.cf_pipeline import FrameQueue, Relay, cf_put_key
from dynamo.vllm.omni.video_convert import decoded_chunk_to_canonical

pytestmark = pytest.mark.unit


# -- connector keys --------------------------------------------------------


def test_put_key_is_unique_per_block():
    """The bug this prevents: every block of a stream overwriting one SHM segment."""
    keys = {cf_put_key("req-1", b) for b in range(8)}
    assert len(keys) == 8


def test_put_key_keeps_cleanup_matchable():
    """SharedMemoryConnector.cleanup(request_id) matches an ``_``-delimited suffix, so
    the key has to stay in that shape or per-block segments leak."""
    key = cf_put_key("req-1", 3)
    assert key.startswith("req-1_")
    assert key.rsplit("_", 1)[0] == "req-1"


# -- Relay -----------------------------------------------------------------


async def test_relay_delivers_items_in_order():
    relay = Relay()
    for i in range(3):
        await relay.put(i)
    relay.close()
    assert [x async for x in relay] == [0, 1, 2]


async def test_relay_close_ends_iteration_after_queued_items():
    """The end marker rides in the queue, so it cannot overtake queued data."""
    relay = Relay()
    await relay.put("a")
    relay.close()
    assert [x async for x in relay] == ["a"]


async def test_relay_raises_the_producer_error():
    relay = Relay()
    await relay.put("a")
    relay.close(RuntimeError("DiT stage died"))

    seen = []
    with pytest.raises(RuntimeError, match="DiT stage died"):
        async for item in relay:
            seen.append(item)
    assert seen == ["a"], "the error must arrive after the data already produced"


async def test_relay_close_is_idempotent_and_keeps_the_first_outcome():
    """A failure while unwinding an already-finished producer must not be reported as
    the stream's outcome."""
    relay = Relay()
    relay.close()
    relay.close(RuntimeError("late unrelated error"))
    assert [x async for x in relay] == []


async def test_relay_bounds_blocks_in_flight():
    """The DiT stage is faster than the VAE stage, so without back-pressure it would
    run the whole rollout ahead and pin an SHM segment per queued block."""
    relay = Relay(maxsize=2)
    await relay.put(0)
    await relay.put(1)

    third = asyncio.create_task(relay.put(2))
    await asyncio.sleep(0)
    assert not third.done(), "put must block once maxsize items are in flight"

    consumed = []
    async for item in relay:
        consumed.append(item)
        if len(consumed) == 1:
            await asyncio.sleep(0)  # let the blocked put through
            relay.close()
    await third
    assert consumed[0] == 0


async def test_relay_close_never_blocks_on_a_full_queue():
    """Why the bound lives in a semaphore and not the queue: close() is called from a
    ``finally``, and a close that could block would deadlock exactly when a terminal
    error most needs to get through."""
    relay = Relay(maxsize=1)
    await relay.put("a")
    relay.close(RuntimeError("boom"))  # would block on a bounded queue
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in relay:
            pass


# -- FrameQueue ------------------------------------------------------------


def test_frame_queue_iterates_puts_then_ends():
    fq = FrameQueue()
    fq.put("chunk0")
    fq.put("chunk1")
    fq.close()
    assert list(fq) == ["chunk0", "chunk1"]


def test_frame_queue_raises_the_producer_error_after_its_frames():
    """The encoder must not mistake a crashed rollout for a finished one -- both would
    otherwise produce a valid, shorter video."""
    fq = FrameQueue()
    fq.put("chunk0")
    fq.close(RuntimeError("VAE stage died"))

    seen = []
    with pytest.raises(RuntimeError, match="VAE stage died"):
        for item in fq:
            seen.append(item)
    assert seen == ["chunk0"]


def test_frame_queue_times_out_rather_than_hanging_forever():
    """A stage worker that dies without closing the stream would otherwise leave the
    encoder's writer thread -- and the request -- blocked with nothing to report."""
    fq = FrameQueue(timeout_s=0.05)
    with pytest.raises(TimeoutError, match="stopped producing"):
        list(fq)


async def test_frame_queue_crosses_the_thread_boundary():
    """Its whole purpose: filled from the event loop, drained by the encoder thread."""
    fq = FrameQueue(timeout_s=5.0)

    async def produce():
        for i in range(3):
            await asyncio.sleep(0)
            fq.put(i)
        fq.close()

    task = asyncio.create_task(produce())
    assert await asyncio.to_thread(list, fq) == [0, 1, 2]
    await task


# -- decoded chunk conversion ---------------------------------------------


def test_decoded_chunk_maps_minus_one_to_black_and_one_to_white():
    """The silent trap: session_decode_step returns [-1, 1], not [0, 1]. Reading it as
    [0, 1] clips every dark pixel to black and halves the contrast of the rest --
    a video that plays fine and looks wrong."""
    chunk = torch.stack(
        [
            torch.full((3, 4, 4), -1.0),  # black
            torch.zeros((3, 4, 4)),  # mid grey
            torch.full((3, 4, 4), 1.0),  # white
        ],
        dim=1,
    ).unsqueeze(0)  # [B=1, C=3, T=3, H=4, W=4]

    frames = decoded_chunk_to_canonical(chunk)

    assert frames.shape == (3, 4, 4, 3)
    assert frames.dtype == np.uint8
    assert frames[0].max() == 0
    assert 126 <= frames[1].min() <= 129
    assert frames[2].min() == 255


def test_decoded_chunk_accepts_an_unbatched_tensor():
    chunk = torch.zeros((3, 2, 4, 4))  # [C, T, H, W]
    assert decoded_chunk_to_canonical(chunk).shape == (2, 4, 4, 3)


def test_decoded_chunk_unwraps_a_single_element_list():
    chunk = [torch.full((1, 3, 2, 4, 4), 1.0)]
    assert decoded_chunk_to_canonical(chunk).shape == (2, 4, 4, 3)


def test_decoded_chunk_rejects_an_unusable_shape():
    with pytest.raises(ValueError, match=r"\[B, C, T, H, W\]"):
        decoded_chunk_to_canonical(torch.zeros((4, 4)))


def test_decoded_chunk_passes_through_post_processed_frames():
    """An aggregated pipeline may have run the post-process func before the handoff,
    in which case the frames are already canonical."""
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    assert decoded_chunk_to_canonical(frames).shape == (2, 4, 4, 3)
