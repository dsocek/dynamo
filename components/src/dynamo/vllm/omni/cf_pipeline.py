# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Handoff primitives for a pipelined Causal-Forcing stream.

A one-shot disaggregated request runs DiT to completion, then VAE, then encodes:
three phases, each idle while the others work. A rollout is naturally
block-at-a-time, so those phases can overlap instead -- block N decodes while
block N+1 rolls out and block N-1 is being packaged. That is the whole gain, and
it is entirely a scheduling change: the same tensors go through the same code.

What makes it fiddly is that the three consumers do not share a concurrency
model. The stage hops are ``async``; the video encoder is a *blocking* generator
driving an ffmpeg pipe from a worker thread (see
:func:`~dynamo.common.utils.video_utils.encode_video_fragments`). So the chain
needs two different handoffs, and this module holds both:

``Relay``
    async -> async, bounded. Carries rollout blocks from the DiT stage to the VAE
    stage. Bounded because the DiT stage is the faster of the two (measured 0.957
    s/latent against the VAE's 1.290), so an unbounded queue would let it run
    away and turn a flat-memory rollout into a growing one -- each queued block
    also pins a shared-memory segment until the VAE stage reads it.

``FrameQueue``
    async -> blocking-iterator. Carries decoded frames to the encoder's writer
    thread, which pulls with an ordinary ``for``. Off the event loop by
    construction: the thread blocks on ``get`` while the loop keeps filling it.

Both carry a *terminal error* alongside the terminal marker, which is the part
worth being deliberate about. A consumer that only knows "the producer stopped"
cannot tell a finished stream from a crashed one -- and for video the two look
identical downstream: a short clip. So a failed producer delivers its exception
to the consumer, and the consumer raises it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)

# Rollout blocks allowed between the DiT and VAE stages. Enough to keep the VAE
# stage from ever waiting on the DiT stage (one in flight plus slack), small
# enough that queued blocks and their SHM segments stay a rounding error against
# the KV window.
CF_MAX_BLOCKS_IN_FLIGHT = 4

# How long the encoder's writer thread waits for the next decoded chunk before
# giving up. Generous next to a block's decode time (~1.3 s), but finite: without
# it, a stage worker that dies mid-stream would leave that thread -- and the
# request -- blocked forever with nothing to report.
CF_FRAME_TIMEOUT_S = 300.0


def cf_put_key(request_id: str, block: int) -> str:
    """Connector key for one block of a pipelined stream.

    A one-shot request puts exactly once and can key by ``request_id``. A
    pipelined one puts per block, and the shared-memory connector names both the
    segment *and* its ``/dev/shm/shm_<key>_lockfile.lock`` after the key -- so
    reusing the request id would have every block of a stream overwrite the
    previous one's segment while contending on one lock.

    The ``_``-delimited suffix keeps ``SharedMemoryConnector.cleanup(request_id)``
    working, which matches pending keys on exactly that shape.
    """
    return f"{request_id}_b{block}"


class Relay:
    """Bounded async handoff that delivers a terminal error as well as an end.

    One producer, one consumer. ``put`` blocks once ``maxsize`` items are in
    flight, so the producer is paced by the consumer rather than by memory.

    The end marker rides in the queue instead of a flag so that it is ordered
    with respect to the data: a consumer can never see the stream end while an
    item it has not taken is still queued behind it.
    """

    _ITEM = "item"
    _END = "end"

    def __init__(self, maxsize: int = CF_MAX_BLOCKS_IN_FLIGHT) -> None:
        # The queue itself is unbounded and back-pressure lives in the semaphore.
        # A bounded queue would make ``close`` able to block on a full queue,
        # which is precisely when the terminal error most needs to get through.
        self._q: asyncio.Queue = asyncio.Queue()
        self._slots = asyncio.Semaphore(maxsize)
        self._closed = False

    async def put(self, item: Any) -> None:
        await self._slots.acquire()
        self._q.put_nowait((self._ITEM, item))

    def close(self, error: BaseException | None = None) -> None:
        """End the stream, optionally with the exception that ended it.

        Idempotent and never blocking, so it is safe from a ``finally``. Only the
        first close is delivered: an error raised while unwinding a producer that
        already ended normally would be reported as the stream's outcome.
        """
        if self._closed:
            return
        self._closed = True
        self._q.put_nowait((self._END, error))

    async def __aiter__(self) -> AsyncIterator[Any]:
        while True:
            kind, payload = await self._q.get()
            if kind is self._END:
                if payload is not None:
                    raise payload
                return
            try:
                yield payload
            finally:
                self._slots.release()


class FrameQueue:
    """Blocking-iterable handoff from the event loop to the encoder's writer thread.

    ``put`` is called from async code; iteration happens on the thread ffmpeg's
    stdin writer runs on, so the blocking ``get`` never touches the event loop.

    ``timeout_s`` bounds that block. It is not a tuning knob for throughput -- it
    is the only thing standing between a dead upstream stage and a request that
    hangs until the client gives up.
    """

    _DONE = object()

    def __init__(self, timeout_s: float = CF_FRAME_TIMEOUT_S) -> None:
        self._q: queue.Queue = queue.Queue()
        self._timeout_s = timeout_s
        self._error: BaseException | None = None
        self._closed = False

    def put(self, frames: Any) -> None:
        self._q.put(frames)

    def close(self, error: BaseException | None = None) -> None:
        """Signal the end of the frames, optionally with the failure that caused it."""
        if self._closed:
            return
        self._closed = True
        self._error = error
        self._q.put(self._DONE)

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                item = self._q.get(timeout=self._timeout_s)
            except queue.Empty as e:
                raise TimeoutError(
                    f"no decoded frames for {self._timeout_s:.0f}s; the upstream "
                    "stage stopped producing without closing the stream"
                ) from e
            if item is self._DONE:
                if self._error is not None:
                    raise self._error
                return
            yield item
