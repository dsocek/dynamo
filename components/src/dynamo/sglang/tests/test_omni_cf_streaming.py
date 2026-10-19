# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SGLang-Omni Causal-Forcing streaming pipeline.

No engine, no device, no ffmpeg. Stubs stand in for the SGLang instance and for the
Dynamo stage clients, which leaves the parts that are genuinely ours under test: the
block numbering, the payload transport, the two workers' response contracts, and the
router's pipelining and session pinning.

What is worth pinning here is the set of failures that are *invisible* when broken --
each of these would render a video rather than raise:

* a ``continue`` on a session nobody opened restarts the shot from noise,
* a scene's blocks are numbered from something other than 0 after a cut,
* two payloads collide on one transport key and one block decodes the other's bytes,
* a block is read twice, or left resident in ``/dev/shm`` after being read,
* multi-frame payloads are split on the wrong stride, so the clip is scrambled,
* the router loses a block or reorders two, so the seam moves,
* a session's chunks land on different instances, so the KV window resets mid-shot,
* a stage failure ends the stream quietly and the client keeps a short clip.

The end-to-end behaviour (that these blocks really do come back as seam-free video)
is verified on hardware, where the DiT and VAE stages run on separate XPU cards.
"""

from __future__ import annotations

import asyncio
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dynamo.common.omni.cf_session import CFSceneRequest, build_cf_annotations
from dynamo.common.omni.cmaf_video import CMAF_ANNOTATION
from dynamo.sglang.omni import blocks
from dynamo.sglang.omni.args import default_stage_names
from dynamo.sglang.omni.cf_dispatch import (
    CFChunkDispatcher,
    CFDispatchError,
    CFSessionBook,
    SceneGeometry,
    frames_from_output,
    handoff_payload,
    latent_frames,
    scene_chunk_count,
    scene_geometry,
)
from dynamo.sglang.omni.output_formatter import VideoFormatter
from dynamo.sglang.omni.stage_router import OmniSglangStageRouter
from dynamo.sglang.omni.stage_worker import OmniSglangStageWorker

pytestmark = pytest.mark.unit

HEIGHT, WIDTH = 4, 6
FPS = 16


@pytest.fixture(autouse=True)
def block_dir(tmp_path, monkeypatch):
    """Keep every test's block payloads out of the real ``/dev/shm`` directory."""
    monkeypatch.setenv("DYN_CF_BLOCK_DIR", str(tmp_path / "cf_blocks"))
    return tmp_path / "cf_blocks"


# -- stubs -----------------------------------------------------------------


class StubHandoff:
    """Stands in for SGLang's ``StageHandoff``: pickled, shipped, rebuilt as a Req."""

    def __init__(self, session_id: str, block: int) -> None:
        self.session_id = session_id
        self.block = block

    def to_req(self) -> SimpleNamespace:
        return SimpleNamespace(
            realtime_session_id=self.session_id, block_idx=self.block
        )


def stub_output(
    *, handoff: StubHandoff | None = None, frames: np.ndarray | None = None
) -> SimpleNamespace:
    """An ``OutputBatch``-shaped answer carrying a handoff or raw frame bytes."""
    if frames is None:
        return SimpleNamespace(
            stage_handoff=handoff, raw_frame_batches=[], raw_frame_metadata={}, error=None
        )
    return SimpleNamespace(
        stage_handoff=None,
        # One payload holding every frame, as SGLang delivers a multi-frame chunk.
        raw_frame_batches=[[frames.tobytes()]],
        raw_frame_metadata={
            "height": frames.shape[1],
            "width": frames.shape[2],
            "channels": frames.shape[3],
        },
        error=None,
    )


def block_frames(block: int, count: int = 1) -> np.ndarray:
    """Frames whose pixel value identifies the block they came from."""
    return np.full((count, HEIGHT, WIDTH, 3), block + 1, dtype=np.uint8)


class StubDispatcher(CFChunkDispatcher):
    """Real book and real release path; the two engine calls are stubbed.

    Subclassed rather than replaced so the parts under test stay real: block
    numbering comes from the actual :class:`CFSessionBook`, and ``release`` still
    goes through the dispatcher's own best-effort handling.
    """

    def __init__(
        self,
        stage_id: int,
        *,
        decodes: bool,
        frames_per_block: int = 1,
        chunk_size: int = 1,
        fail_on_block: int | None = None,
    ) -> None:
        super().__init__(SimpleNamespace(), stage_id)
        self._decodes = decodes
        self._frames_per_block = frames_per_block
        self._chunk_size = chunk_size
        self._fail_on_block = fail_on_block
        self.rollouts: list[tuple[str, int]] = []
        self.seeds: list[int | None] = []
        self.decoded: list[tuple[str, int]] = []
        self.released: list[str] = []

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    async def rollout_chunk(
        self, session_id: str, block_idx: int, prompt: str, geometry: SceneGeometry
    ) -> Any:
        if block_idx == self._fail_on_block:
            raise CFDispatchError(f"stub rollout failure at block {block_idx}")
        self.rollouts.append((session_id, block_idx))
        self.seeds.append(geometry.seed)
        if self._decodes:
            return stub_output(frames=block_frames(block_idx, self._frames_per_block))
        return stub_output(handoff=StubHandoff(session_id, block_idx))

    async def decode_chunk(self, handoff_payload_bytes: bytes) -> Any:
        handoff = pickle.loads(handoff_payload_bytes)
        req = handoff.to_req()
        self.decoded.append((req.realtime_session_id, req.block_idx))
        return stub_output(
            frames=block_frames(req.block_idx, self._frames_per_block)
        )

    async def release(self, session_id: str) -> None:
        self.released.append(session_id)
        self.book.forget(session_id)


def stage_worker(stage_id: int, dispatcher: StubDispatcher) -> OmniSglangStageWorker:
    config = SimpleNamespace(default_video_fps=FPS)
    worker = OmniSglangStageWorker.__new__(OmniSglangStageWorker)
    worker.config = config
    worker.stage_id = stage_id
    worker.dispatcher = dispatcher
    return worker


class StubChunk:
    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        return self._data


class StubClient:
    """A Dynamo client whose instances are in-process worker handlers."""

    def __init__(self, handlers: dict[int, Any]) -> None:
        self._handlers = handlers
        self.calls: list[tuple[int, dict]] = []

    def instance_ids(self) -> list[int]:
        return list(self._handlers)

    async def direct(self, request: dict, instance_id: int):
        self.calls.append((instance_id, request))
        return self._stream(self._handlers[instance_id], request)

    @staticmethod
    async def _stream(handler: Any, request: dict):
        async for chunk in handler(request, None):
            yield StubChunk(chunk)


def router(
    clients: dict[str, StubClient], *, num_stages: int = 2
) -> OmniSglangStageRouter:
    config = SimpleNamespace(
        resolved_stage_names=default_stage_names(num_stages),
        served_model_name="cf-test",
        model="cf-test",
        media_output_fs_url=None,
        media_output_http_url=None,
        default_video_fps=FPS,
    )
    r = OmniSglangStageRouter(config)
    for name, client in clients.items():
        r.set_stage_client(name, client)
    return r


def video_request(
    session_id: str | None,
    *,
    transition: str = "cut",
    latents: int | None = None,
    close: bool = False,
    cmaf: bool = False,
    seed: int | None = None,
    nvext_seed: int | None = None,
) -> dict:
    nvext: dict[str, Any] = {"fps": FPS, "height": HEIGHT, "width": WIDTH}
    if nvext_seed is not None:
        nvext["seed"] = nvext_seed
    annotations: list[str] = [CMAF_ANNOTATION] if cmaf else []
    if session_id is not None:
        annotations += build_cf_annotations(
            session_id, transition=transition, latents=latents, seed=seed, close=close
        )
    if annotations:
        nvext["annotations"] = annotations
    return {"prompt": "a wave breaking", "size": f"{WIDTH}x{HEIGHT}", "nvext": nvext}


async def drain(gen) -> list[dict]:
    return [chunk async for chunk in gen]


async def _completed(frames, request_id, *, fps=None, **_):
    """A stand-in for the batch formatter, so no test shells out to ffmpeg."""
    return {"status": "completed"}


# -- block numbering -------------------------------------------------------


def test_cut_numbers_from_zero_and_continue_keeps_counting():
    """Block 0 *is* the cut, so the numbering is the protocol, not a detail."""
    book = CFSessionBook()
    book.begin_scene("s", "cut")
    assert [book.take("s") for _ in range(3)] == [0, 1, 2]

    book.begin_scene("s", "continue")
    # Not 0: a continue that renumbered would clear the attention window and the
    # shot would restart from noise, which renders rather than fails.
    assert book.take("s") == 3

    book.begin_scene("s", "cut")
    assert book.take("s") == 0


def test_continue_without_a_shot_is_refused():
    book = CFSessionBook()
    with pytest.raises(CFDispatchError, match="no shot in progress"):
        book.begin_scene("ghost", "continue")


def test_closing_a_session_forgets_its_numbering():
    book = CFSessionBook()
    book.begin_scene("s", "cut")
    book.take("s")
    book.forget("s")
    with pytest.raises(CFDispatchError):
        book.begin_scene("s", "continue")


def test_scene_length_in_chunks():
    geometry = SceneGeometry(height=480, width=832, num_frames=81, fps=16)
    assert latent_frames(81) == 21
    assert geometry.latents == 21
    # No explicit length: the clip's own latent count.
    assert scene_chunk_count(CFSceneRequest(), geometry, 1) == 21
    # A scene that is not a whole number of chunks still runs its tail.
    assert scene_chunk_count(CFSceneRequest(latents=5), geometry, 2) == 3


def test_geometry_prefers_nvext_over_the_portable_fields():
    geometry = scene_geometry(
        {"size": "832x480", "nvext": {"height": 320, "width": 512, "fps": 24}}, 16
    )
    assert (geometry.height, geometry.width, geometry.fps) == (320, 512, 24)


# -- seeds -----------------------------------------------------------------
#
# Every failure below renders video rather than raising. SGLang keeps one noise
# generator per realtime session and reuses it only while ``batch.seed`` is
# unchanged, so a seed that moves between chunks silently rebuilds the generator
# and every block draws the same noise -- a shot that sits still instead of moving.


def test_a_scene_is_given_one_seed_even_when_the_client_states_none():
    """An unstated seed is drawn once per scene, not once per chunk."""
    book = CFSessionBook()
    book.begin_scene("s", "cut")
    drawn = book.seed_for_scene("s", None, "cut")
    assert isinstance(drawn, int) and drawn >= 0
    # A `continue` inherits it: this is the same shot, so it is the same generator.
    assert book.seed_for_scene("s", None, "continue") == drawn
    # A cut is a new shot, so it does not reuse the previous shot's noise.
    assert book.seed_for_scene("s", None, "cut") != drawn


def test_a_stated_seed_wins_and_is_remembered_for_the_scenes_that_follow():
    book = CFSessionBook()
    book.begin_scene("s", "cut")
    assert book.seed_for_scene("s", 7, "cut") == 7
    assert book.seed_for_scene("s", None, "continue") == 7
    assert book.seed_for_scene("s", 9, "continue") == 9
    # Forgetting the session forgets the seed, so a reused id is a fresh shot.
    book.forget("s")
    assert book.seed_for_scene("s", None, "continue") != 9


async def test_every_chunk_of_a_scene_carries_the_same_resolved_seed():
    dispatcher = StubDispatcher(0, decodes=False, chunk_size=1)
    worker = stage_worker(0, dispatcher)
    await drain(worker.generate(video_request("s", latents=3), None))
    assert len(dispatcher.seeds) == 3
    assert len(set(dispatcher.seeds)) == 1
    assert isinstance(dispatcher.seeds[0], int) and dispatcher.seeds[0] >= 0


async def test_the_scenes_seed_outranks_a_request_level_one():
    """``cf_scene`` is the per-scene channel, so it is the one that decides."""
    dispatcher = StubDispatcher(0, decodes=False, chunk_size=1)
    worker = stage_worker(0, dispatcher)
    await drain(
        worker.generate(video_request("s", latents=2, seed=11, nvext_seed=22), None)
    )
    assert dispatcher.seeds == [11, 11]


async def test_a_request_level_seed_is_used_when_the_scene_states_none():
    dispatcher = StubDispatcher(0, decodes=False, chunk_size=1)
    worker = stage_worker(0, dispatcher)
    await drain(worker.generate(video_request("s", latents=2, nvext_seed=22), None))
    assert dispatcher.seeds == [22, 22]


# -- transport -------------------------------------------------------------


def test_reading_a_block_consumes_it():
    ref = blocks.put_handoff(blocks.block_key("r", 0, blocks.KIND_HANDOFF), b"payload")
    assert blocks.take_handoff(ref) == b"payload"
    # A second consumer must not get stale bytes, and the segment must not linger:
    # /dev/shm is RAM, so a leak grows with the stream rather than with its load.
    with pytest.raises(RuntimeError, match="already taken"):
        blocks.take_handoff(ref)


def test_the_two_stages_do_not_collide_on_one_block():
    """Both stages write block N; only the kind keeps them apart."""
    handoff = blocks.put_handoff(
        blocks.block_key("r", 7, blocks.KIND_HANDOFF), b"latents"
    )
    frames = blocks.put_frames(
        blocks.block_key("r", 7, blocks.KIND_FRAMES), block_frames(7)
    )
    assert handoff["path"] != frames["path"]
    assert blocks.take_handoff(handoff) == b"latents"
    assert blocks.take_frames(frames).shape == (1, HEIGHT, WIDTH, 3)


def test_a_truncated_block_is_refused_rather_than_decoded():
    ref = blocks.put_frames(
        blocks.block_key("r", 1, blocks.KIND_FRAMES), block_frames(1)
    )
    ref = {**ref, "nbytes": ref["nbytes"] + 1}
    with pytest.raises(RuntimeError, match="ref says"):
        blocks.take_frames(ref)


def test_frames_survive_the_round_trip_writable():
    original = block_frames(3, count=4)
    ref = blocks.put_frames(blocks.block_key("r", 3, blocks.KIND_FRAMES), original)
    got = blocks.take_frames(ref)
    assert np.array_equal(original, got)
    # The encoder downstream is entitled to write into what it is handed.
    got[0, 0, 0, 0] = 99


def test_multi_frame_payloads_split_on_the_frame_stride():
    """Chunk 0 emits one pixel frame and later chunks emit four."""
    frames = block_frames(0, count=4)
    got = frames_from_output(stub_output(frames=frames))
    assert got.shape == (4, HEIGHT, WIDTH, 3)
    assert np.array_equal(got, frames)


def test_a_payload_that_is_not_a_whole_number_of_frames_is_refused():
    output = stub_output(frames=block_frames(0))
    output.raw_frame_batches = [[output.raw_frame_batches[0][0][:-3]]]
    with pytest.raises(CFDispatchError, match="frame stride"):
        frames_from_output(output)


def test_handoff_and_frames_are_told_apart_by_the_answer():
    assert handoff_payload(stub_output(frames=block_frames(0))) is None
    assert handoff_payload(stub_output(handoff=StubHandoff("s", 0))) is not None


# -- rollout worker --------------------------------------------------------


async def test_rollout_emits_one_block_per_chunk_then_a_terminal_marker():
    dispatcher = StubDispatcher(0, decodes=False)
    worker = stage_worker(0, dispatcher)
    request = video_request("s1", latents=3)

    chunks = await drain(worker.generate({"cf_request_id": "r1", **request}))

    assert [c.get("cf_block") for c in chunks] == [0, 1, 2, 3]
    assert [c["cf_ref"]["kind"] for c in chunks[:3]] == [blocks.KIND_HANDOFF] * 3
    terminal = chunks[-1]
    assert terminal["cf_last"] is True and terminal["cf_block"] == 3
    assert dispatcher.rollouts == [("s1", 0), ("s1", 1), ("s1", 2)]
    # Not closed: the shot stays live so the next scene can continue it.
    assert dispatcher.released == []
    for chunk in chunks[:3]:
        blocks.take_handoff(chunk["cf_ref"])


async def test_a_closing_scene_releases_the_session():
    dispatcher = StubDispatcher(0, decodes=False)
    worker = stage_worker(0, dispatcher)

    chunks = await drain(
        worker.generate(video_request("s2", latents=1, close=True))
    )

    assert dispatcher.released == ["s2"]
    assert chunks[-1]["cf_close"] is True
    blocks.take_handoff(chunks[0]["cf_ref"])


async def test_a_failed_block_is_reported_and_the_session_still_released():
    dispatcher = StubDispatcher(0, decodes=False, fail_on_block=1)
    worker = stage_worker(0, dispatcher)

    chunks = await drain(
        worker.generate(video_request("s3", latents=3, close=True))
    )

    assert chunks[0]["cf_block"] == 0
    # A terminal error chunk, not a silent stop: the router has to be able to fail
    # the request rather than hand the client a short clip.
    assert "stub rollout failure" in chunks[-1]["error"]
    assert chunks[-1]["cf_last"] is True
    assert dispatcher.released == ["s3"]
    blocks.take_handoff(chunks[0]["cf_ref"])


async def test_a_continue_with_no_shot_is_refused_before_any_work():
    dispatcher = StubDispatcher(0, decodes=False)
    worker = stage_worker(0, dispatcher)

    chunks = await drain(
        worker.generate(video_request("ghost", transition="continue", latents=2))
    )

    assert len(chunks) == 1 and "no shot in progress" in chunks[0]["error"]
    assert dispatcher.rollouts == []


async def test_an_aggregated_worker_publishes_pixels_not_a_handoff():
    """One worker holding every role answers with frames, and says so in the ref."""
    dispatcher = StubDispatcher(0, decodes=True, frames_per_block=4)
    worker = stage_worker(0, dispatcher)

    chunks = await drain(worker.generate(video_request("s4", latents=2)))

    assert [c["cf_ref"]["kind"] for c in chunks[:2]] == [blocks.KIND_FRAMES] * 2
    assert chunks[0]["cf_ref"]["shape"] == [4, HEIGHT, WIDTH, 3]
    for chunk in chunks[:2]:
        blocks.take_frames(chunk["cf_ref"])


# -- decode worker ---------------------------------------------------------


async def test_decode_turns_a_handoff_into_published_pixels():
    dispatcher = StubDispatcher(1, decodes=True, frames_per_block=4)
    worker = stage_worker(1, dispatcher)
    ref = blocks.put_handoff(
        blocks.block_key("r5", 2, blocks.KIND_HANDOFF),
        pickle.dumps(StubHandoff("s5", 2)),
    )

    chunks = await drain(
        worker.generate(
            {"cf_request_id": "r5", "cf_session": "s5", "cf_block": 2, "cf_ref": ref}
        )
    )

    assert len(chunks) == 1
    assert dispatcher.decoded == [("s5", 2)]
    frames = blocks.take_frames(chunks[0]["cf_ref"])
    assert np.array_equal(frames, block_frames(2, 4))


async def test_decode_releases_only_when_the_scene_asked_to_close():
    dispatcher = StubDispatcher(1, decodes=True)
    worker = stage_worker(1, dispatcher)

    await drain(
        worker.generate({"cf_session": "s6", "cf_block": 3, "cf_last": True})
    )
    # A shot that will be continued needs its conv cache to still be there, or the
    # next block's first frame becomes a seam.
    assert dispatcher.released == []

    await drain(
        worker.generate(
            {"cf_session": "s6", "cf_block": 3, "cf_last": True, "cf_close": True}
        )
    )
    assert dispatcher.released == ["s6"]


async def test_decode_refuses_a_reference_that_is_already_pixels():
    dispatcher = StubDispatcher(1, decodes=True)
    worker = stage_worker(1, dispatcher)
    ref = blocks.put_frames(
        blocks.block_key("r7", 0, blocks.KIND_FRAMES), block_frames(0)
    )

    chunks = await drain(
        worker.generate({"cf_session": "s7", "cf_block": 0, "cf_ref": ref})
    )

    assert "stage-roles" in chunks[0]["error"]
    assert dispatcher.decoded == []
    blocks.take_frames(ref)


# -- router ----------------------------------------------------------------


def two_stage_router(
    *, frames_per_block: int = 4, fail_on_block: int | None = None
) -> tuple[OmniSglangStageRouter, StubDispatcher, StubDispatcher]:
    rollout = StubDispatcher(0, decodes=False, fail_on_block=fail_on_block)
    decode = StubDispatcher(1, decodes=True, frames_per_block=frames_per_block)
    names = default_stage_names(2)
    clients = {
        names[0]: StubClient({11: stage_worker(0, rollout).generate}),
        names[1]: StubClient({21: stage_worker(1, decode).generate}),
    }
    return router(clients), rollout, decode


async def test_the_router_returns_every_block_in_order():
    r, rollout, decode = two_stage_router()
    captured: dict[str, Any] = {}

    async def fake_format(frames, request_id, *, fps=None, **_):
        captured["frames"] = frames
        captured["fps"] = fps
        return {"status": "completed"}

    r._formatter.format_video = fake_format  # type: ignore[method-assign]

    chunks = await drain(r.generate(video_request("s8", latents=3, close=True)))

    assert chunks == [{"status": "completed"}]
    assert rollout.rollouts == [("s8", 0), ("s8", 1), ("s8", 2)]
    assert decode.decoded == [("s8", 0), ("s8", 1), ("s8", 2)]
    # 3 blocks x 4 frames, and the pixel values say which block each came from --
    # a reordering or a lost block shows up here rather than as a moved seam.
    assert captured["frames"].shape == (12, HEIGHT, WIDTH, 3)
    assert [int(f[0, 0, 0]) for f in captured["frames"]] == [1] * 4 + [2] * 4 + [3] * 4
    assert captured["fps"] == FPS


async def test_a_plain_request_gets_an_implicit_single_scene_session():
    """``POST /v1/videos`` keeps working, and does not leak a session."""
    r, rollout, decode = two_stage_router()
    r._formatter.format_video = _completed  # type: ignore[method-assign]

    await drain(r.generate({"prompt": "surf", "size": f"{WIDTH}x{HEIGHT}"}))

    sessions = {sid for sid, _ in rollout.rollouts}
    assert len(sessions) == 1
    session_id = sessions.pop()
    assert session_id.startswith("auto-")
    # Started from a cut and closed at the end: a one-shot caller cannot leave a
    # card holding a window it will never continue.
    assert rollout.rollouts[0] == (session_id, 0)
    assert rollout.released == [session_id]
    assert decode.released == [session_id]


async def test_a_rollout_failure_is_reported_not_truncated():
    r, _, _ = two_stage_router(fail_on_block=1)
    r._formatter.format_video = _completed  # type: ignore[method-assign]

    chunks = await drain(r.generate(video_request("s9", latents=3, close=True)))

    assert chunks[-1]["status"] == "failed"
    assert "stub rollout failure" in chunks[-1]["error"]


async def test_a_session_stays_on_one_instance():
    rollout = StubDispatcher(0, decodes=False)
    decode = StubDispatcher(1, decodes=True)
    names = default_stage_names(2)
    rollout_client = StubClient(
        {11: stage_worker(0, rollout).generate, 12: stage_worker(0, rollout).generate}
    )
    clients = {
        names[0]: rollout_client,
        names[1]: StubClient({21: stage_worker(1, decode).generate}),
    }
    r = router(clients)
    r._formatter.format_video = _completed  # type: ignore[method-assign]

    await drain(r.generate(video_request("s10", latents=2)))
    await drain(r.generate(video_request("s10", transition="continue", latents=2)))

    # The KV window lives on whichever instance took the first chunk; scattering
    # the rest of the session would reset the shot mid-stream.
    assert {instance for instance, _ in rollout_client.calls} == {11}


async def test_a_second_session_prefers_an_idle_instance():
    """Each SGLang instance holds one realtime session, so a busy one is avoided."""
    rollout = StubDispatcher(0, decodes=False)
    client = StubClient(
        {11: stage_worker(0, rollout).generate, 12: stage_worker(0, rollout).generate}
    )
    r = router({default_stage_names(2)[0]: client})

    first = r._instance_for(default_stage_names(2)[0], client, "a")
    second = r._instance_for(default_stage_names(2)[0], client, "b")
    assert first != second

    # Closing the first frees its instance for the next session.
    r._unpin("a")
    assert r._instance_for(default_stage_names(2)[0], client, "c") == first


async def test_cmaf_requests_take_the_live_lane():
    r, _, _ = two_stage_router()
    seen: dict[str, Any] = {}

    async def fake_live(frame_chunks, request_id, *, fps):
        seen["fps"] = fps
        seen["blocks"] = await asyncio.to_thread(list, frame_chunks)
        yield {"status": "in_progress"}

    r._formatter.stream_video_cmaf_live = fake_live  # type: ignore[method-assign]

    chunks = await drain(
        r.generate(video_request("s11", latents=2, close=True, cmaf=True))
    )

    assert chunks == [{"status": "in_progress"}]
    # Fed block by block rather than as one array: that is what lets a media
    # segment ship before the later frames exist.
    assert [b.shape[0] for b in seen["blocks"]] == [4, 4]
    assert seen["fps"] == FPS


async def test_an_aggregated_deploy_skips_the_decode_hop():
    aggregated = StubDispatcher(0, decodes=True, frames_per_block=4)
    names = default_stage_names(1)
    clients = {names[0]: StubClient({11: stage_worker(0, aggregated).generate})}
    r = router(clients, num_stages=1)
    captured: dict[str, Any] = {}

    async def fake_format(frames, request_id, *, fps=None, **_):
        captured["frames"] = frames
        return {"status": "completed"}

    r._formatter.format_video = fake_format  # type: ignore[method-assign]

    await drain(r.generate(video_request("s12", latents=2, close=True)))

    assert captured["frames"].shape == (8, HEIGHT, WIDTH, 3)
    assert aggregated.decoded == []


# -- formatter -------------------------------------------------------------


def test_live_progress_is_monotonic_and_never_complete():
    formatter = VideoFormatter("cf-test")
    values = [formatter._live_progress(n) for n in range(1, 200)]
    assert values == sorted(values)
    assert values[0] >= 1 and max(values) <= 99
