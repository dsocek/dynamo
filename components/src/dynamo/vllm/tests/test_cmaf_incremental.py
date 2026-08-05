# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for incremental (streaming) CMAF packaging.

No ffmpeg, no device: the box parser and the formatter's framing are pure logic, and a
fake fragment generator stands in for the encoder. What is worth pinning here is not
that the happy path emits pieces -- it is the properties that are invisible when broken:

* a media segment is never emitted before the init segment that configures the decoder,
* a partially-arrived box is never emitted as if it were complete,
* the trailing index box is never handed to a SourceBuffer,
* a mid-stream failure is reported rather than looking like a short clip,
* and the encoder process is reaped when the consumer walks away.

The ffmpeg-backed behaviour (that fMP4 to a pipe really does yield fragments early) is
verified on hardware; see PHASE2_DESIGN.md. These tests deliberately do not shell out.
"""

import asyncio
import base64
import json

import numpy as np
import pytest
from dynamo.common.utils.video_utils import iter_fragmented_mp4_boxes
from dynamo.vllm.omni.cmaf_video import (
    CMAF_FALLBACK_VIDEO_CODEC,
    CMAF_OPEN_ENDED,
    metadata_bytes,
)
from dynamo.vllm.omni.output_formatter import DiffusionFormatter, _AsyncFragments

pytestmark = pytest.mark.unit


def box(btype: bytes, payload: bytes = b"") -> bytes:
    """Build a minimal top-level ISO-BMFF box."""
    return (len(payload) + 8).to_bytes(4, "big") + btype + payload


# -- the incremental box parser --------------------------------------------


def test_complete_boxes_are_split_with_no_remainder():
    data = box(b"ftyp", b"iso5") + box(b"moov", b"x" * 16)
    boxes, rest = iter_fragmented_mp4_boxes(data)
    assert [b[0] for b in boxes] == [b"ftyp", b"moov"]
    assert rest == b""


def test_partial_box_is_held_back_not_emitted():
    """The failure this prevents: half an mdat handed to a client as a segment."""
    full = box(b"mdat", b"y" * 64)
    boxes, rest = iter_fragmented_mp4_boxes(full[:40])
    assert boxes == []
    assert rest == full[:40]
    # ... and completes once the rest arrives.
    boxes, rest = iter_fragmented_mp4_boxes(rest + full[40:])
    assert [b[0] for b in boxes] == [b"mdat"]
    assert rest == b""


def test_a_header_split_across_reads_is_held():
    """Fewer than 8 bytes cannot even name the box, let alone size it."""
    boxes, rest = iter_fragmented_mp4_boxes(b"\x00\x00\x00")
    assert boxes == [] and rest == b"\x00\x00\x00"


def test_64_bit_extended_size_box():
    payload = b"z" * 8
    raw = (1).to_bytes(4, "big") + b"mdat" + (16 + len(payload)).to_bytes(8, "big") + payload
    boxes, rest = iter_fragmented_mp4_boxes(raw)
    assert [b[0] for b in boxes] == [b"mdat"] and rest == b""


def test_size_zero_box_is_not_consumed_mid_stream():
    """size==0 means "to end of stream", which cannot be trusted while more is coming."""
    boxes, rest = iter_fragmented_mp4_boxes((0).to_bytes(4, "big") + b"mdat" + b"abc")
    assert boxes == []
    assert rest.startswith((0).to_bytes(4, "big"))


def test_garbage_size_terminates_rather_than_looping():
    """A size below the header length would not advance the cursor."""
    boxes, rest = iter_fragmented_mp4_boxes((2).to_bytes(4, "big") + b"mdat")
    assert boxes == []


# -- metadata / open-ended protocol ----------------------------------------


def test_open_ended_metadata_declares_itself():
    meta = json.loads(metadata_bytes(None, 2, "avc1.64081e"))
    assert meta["open_ended"] is True
    assert meta["segment_count"] == CMAF_OPEN_ENDED
    # A count-driven client appends nothing rather than misreading a real count.
    assert meta["segment_count"] == 0


def test_known_count_metadata_is_unchanged_for_old_clients():
    meta = json.loads(metadata_bytes(4, 2, "avc1.64081e"))
    assert meta["segment_count"] == 4
    assert meta["open_ended"] is False


def test_metadata_advertises_the_codec_it_was_given():
    meta = json.loads(metadata_bytes(None, 2, CMAF_FALLBACK_VIDEO_CODEC))
    assert CMAF_FALLBACK_VIDEO_CODEC in meta["source_buffer_mime_type"]


# -- the async adapter -----------------------------------------------------


async def test_async_fragments_iterates_and_closes():
    closed = []

    def gen():
        try:
            yield ("init", b"i", "avc1.42c00d")
            yield ("segment", b"s", 0)
        finally:
            closed.append(True)

    frags = _AsyncFragments(gen())
    got = [item async for item in frags]
    assert [g[0] for g in got] == ["init", "segment"]
    await frags.aclose()
    assert closed == [True], "generator cleanup must run -- it reaps the encoder"


async def test_aclose_is_idempotent_and_safe_after_exhaustion():
    def gen():
        yield ("init", b"i", "c")

    frags = _AsyncFragments(gen())
    _ = [x async for x in frags]
    await frags.aclose()
    await frags.aclose()  # stream_video_cmaf's finally may hit an exhausted generator


async def test_early_abandonment_runs_generator_cleanup():
    """A client disconnect must not leave an encoder subprocess behind."""
    closed = []

    def gen():
        try:
            yield ("init", b"i", "c")
            yield ("segment", b"s", 0)
        finally:
            closed.append(True)

    frags = _AsyncFragments(gen())
    await frags.__anext__()  # take only the init, then walk away
    await frags.aclose()
    assert closed == [True]


# -- the formatter's framing -----------------------------------------------


def _fake_stream(pieces):
    """Patch-in replacement for stream_frames_to_cmaf."""

    def factory(frame_chunks, fps, segment_seconds, **kwargs):
        # Drain the chunk generator so its own laziness is exercised.
        for _ in frame_chunks:
            pass
        yield from pieces

    return factory


async def _collect(formatter, frames, monkeypatch, pieces):
    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.stream_frames_to_cmaf",
        _fake_stream(pieces),
    )
    return [c async for c in formatter.stream_video_cmaf(frames, "req1", fps=16)]


@pytest.fixture
def formatter():
    return DiffusionFormatter("test-model", None, None, 16)


@pytest.fixture
def frames():
    return np.zeros((32, 16, 16, 3), np.uint8)


async def test_metadata_and_init_precede_any_segment(formatter, frames, monkeypatch):
    """Ordering is the whole contract: a SourceBuffer that gets media before its
    init segment throws, and the video never plays."""
    chunks = await _collect(
        formatter,
        frames,
        monkeypatch,
        [("init", b"INIT", "avc1.64081e"), ("segment", b"S0", 0), ("segment", b"S1", 1)],
    )
    tags = [c["data"][0]["url"] for c in chunks]
    assert tags == ["cmaf:metadata", "cmaf:init", "cmaf:segment:0", "cmaf:segment:1"]


async def test_payloads_survive_base64_round_trip(formatter, frames, monkeypatch):
    chunks = await _collect(
        formatter, frames, monkeypatch, [("init", b"\x00\x01INIT", "c"), ("segment", b"\xff\xfeS", 0)]
    )
    assert base64.b64decode(chunks[1]["data"][0]["b64_json"]) == b"\x00\x01INIT"
    assert base64.b64decode(chunks[2]["data"][0]["b64_json"]) == b"\xff\xfeS"


async def test_metadata_is_open_ended(formatter, frames, monkeypatch):
    """The count is unknown while encoding, so it must not be asserted."""
    chunks = await _collect(
        formatter, frames, monkeypatch, [("init", b"I", "avc1.64081e"), ("segment", b"S", 0)]
    )
    meta = json.loads(base64.b64decode(chunks[0]["data"][0]["b64_json"]))
    assert meta["open_ended"] is True
    assert meta["video_codec"] == "avc1.64081e", "codec comes from the real init segment"


async def test_progress_never_reaches_100_mid_stream(formatter, monkeypatch):
    """HW encoders may emit more segments than predicted, so progress is an estimate;
    reaching 100 before the end would be worse than lagging."""
    frames = np.zeros((32, 16, 16, 3), np.uint8)
    pieces = [("init", b"I", "c")] + [("segment", b"S", i) for i in range(12)]
    chunks = await _collect(formatter, frames, monkeypatch, pieces)
    progress = [c["progress"] for c in chunks if c["data"][0]["url"].startswith("cmaf:segment")]
    assert max(progress) <= 99
    assert all(p >= 1 for p in progress)


async def test_midstream_failure_is_reported_not_silent(formatter, frames, monkeypatch):
    """A truncated stream is indistinguishable from a short clip, so the client has to
    be told explicitly that what it holds is incomplete."""

    def exploding(frame_chunks, fps, segment_seconds, **kwargs):
        yield ("init", b"I", "c")
        yield ("segment", b"S0", 0)
        raise RuntimeError("encoder died")

    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.stream_frames_to_cmaf", exploding
    )
    chunks = [c async for c in formatter.stream_video_cmaf(frames, "req1", fps=16)]
    assert chunks[-1]["status"] == "failed"
    assert "encoder died" in chunks[-1]["error"]
    # The pieces already sent are still valid and were not retracted.
    assert [c["data"][0]["url"] for c in chunks[:3]] == [
        "cmaf:metadata",
        "cmaf:init",
        "cmaf:segment:0",
    ]


async def test_empty_payload_yields_nothing(formatter, monkeypatch):
    assert await _collect(formatter, [], monkeypatch, [("init", b"I", "c")]) == []


# -- frame chunking --------------------------------------------------------


def test_frame_chunks_cover_every_frame_exactly_once():
    frames = np.arange(97 * 2 * 2 * 3, dtype=np.uint8).reshape(97, 2, 2, 3)
    chunks = list(DiffusionFormatter._frame_chunks(frames, fps=16, segment_seconds=2))
    assert sum(len(c) for c in chunks) == 97
    assert np.array_equal(np.concatenate(chunks), frames)


def test_frame_chunks_are_segment_sized():
    frames = np.zeros((97, 2, 2, 3), np.uint8)
    chunks = list(DiffusionFormatter._frame_chunks(frames, fps=16, segment_seconds=2))
    assert [len(c) for c in chunks] == [32, 32, 32, 1]


def test_frame_chunks_never_yields_an_empty_chunk():
    """An empty write would look like end-of-input to the encoder."""
    for n in (1, 5, 32, 33, 97):
        frames = np.zeros((n, 2, 2, 3), np.uint8)
        chunks = list(DiffusionFormatter._frame_chunks(frames, 16, 2))
        assert all(len(c) > 0 for c in chunks)


def test_expected_segments_rounds_up():
    assert DiffusionFormatter._expected_segments(97, 16, 2) == 4
    assert DiffusionFormatter._expected_segments(32, 16, 2) == 1
    assert DiffusionFormatter._expected_segments(33, 16, 2) == 2
    assert DiffusionFormatter._expected_segments(0, 16, 2) == 1  # never zero-divides
