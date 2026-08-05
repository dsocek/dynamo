# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the router's pipelined Causal-Forcing path.

The central test here is ``test_the_stages_actually_overlap``. Everything else about
this path -- the frames, the ordering, the wire protocol -- is identical whether the
stages overlap or not, so a serial implementation would pass every other test in this
file while giving back the whole latency win. That test is the only thing standing
between "pipelined" and "spelled pipelined".
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from dynamo.common.utils.output_modalities import RequestType
from dynamo.vllm.omni import stage_router
from dynamo.vllm.omni.cf_pipeline import cf_put_key

pytestmark = pytest.mark.unit


class _Chunk:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class _StageClient:
    """A stage that may emit several chunks per request, one at a time."""

    def __init__(self, handler):
        self._handler = handler
        self.requests: list[dict] = []

    async def round_robin(self, request):
        self.requests.append(request)
        payloads = self._handler(request)

        async def _gen():
            for payload in payloads:
                if asyncio.iscoroutine(payload):
                    payload = await payload
                yield _Chunk(payload)

        return _gen()


def _stage_cfg(stage_id):
    return SimpleNamespace(
        stage_id=stage_id,
        engine_args=SimpleNamespace(model_stage=f"stage{stage_id}"),
    )


def _router(dit_client, vae_client, formatter=None, connector=None):
    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(
        output_modalities=["video"],
        model="cf",
        served_model_name=None,
        default_video_fps=16,
    )
    router.stage_configs = [_stage_cfg(0), _stage_cfg(1)]
    router.stage_clients = {"stage0": dit_client, "stage1": vae_client}
    router._formatter = formatter or MagicMock()
    router.connectors = (
        {stage_router._connector_key(1, "router"): connector} if connector else {}
    )
    return router


def _rollout(blocks=3):
    """A DiT stage that emits ``blocks`` chunks then the terminal marker."""

    def handler(request):
        payloads = [
            {
                "original_prompt": {"prompt": "reef"},
                "stage_connector_refs": {"0": {"name": f"lat{b}"}},
                "cf_session": "s1",
                "cf_block": b,
                "finished": True,
            }
            for b in range(blocks)
        ]
        payloads.append(
            {"cf_session": "s1", "cf_block": blocks, "cf_last": True, "finished": True}
        )
        return payloads

    return _StageClient(handler)


def _decoder():
    """A VAE stage that answers each per-block request with a router ref."""

    def handler(request):
        return [
            {
                "stage_connector_refs": {
                    "1": {"name": f"frames{request.get('cf_block')}"}
                },
                "cf_session": request.get("cf_session"),
                "cf_block": request.get("cf_block"),
                "finished": True,
            }
        ]

    return _StageClient(handler)


def _frame_connector(frames_per_block=2):
    """A connector returning a decoder-shaped chunk: [B, C, T, H, W] in [-1, 1]."""
    c = MagicMock()
    c.get.side_effect = lambda *a, **k: (
        torch.zeros((1, 3, frames_per_block, 8, 8)),
        1,
    )
    return c


def _cf_request(session_id="s1", cmaf=False):
    annotations = [f"cf_session={session_id}", 'cf_scene={"transition":"cut"}']
    if cmaf:
        annotations.append(stage_router.CMAF_ANNOTATION)
    return {
        "prompt": "a coral reef",
        "size": "832x480",
        "nvext": {"annotations": annotations},
    }


# -- dispatch --------------------------------------------------------------


async def test_a_session_request_takes_the_streaming_path():
    dit, vae = _rollout(2), _decoder()
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": [{"url": "x"}]})
    router = _router(dit, vae, formatter, _frame_connector())

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert chunks == [{"data": [{"url": "x"}]}]
    # Every block reached the VAE stage as its own request -- not one merged call.
    assert [r.get("cf_block") for r in vae.requests] == [0, 1, 2]


async def test_a_one_shot_request_still_takes_the_serial_path():
    """The property that keeps the tested batch chain untouched."""
    dit = _StageClient(lambda r: [{"shm_meta": {"x": 1}, "finished": True}])
    formatter = MagicMock()
    formatter.format = _async_return({"data": []})
    router = _router(dit, _decoder(), formatter)
    router.stage_configs = [_stage_cfg(0)]
    router.stage_clients = {"stage0": dit}

    with (
        patch.object(stage_router, "shm_deserialize", return_value=SimpleNamespace()),
        patch.object(
            stage_router,
            "parse_request_type",
            return_value=(None, RequestType.VIDEO_GENERATION),
        ),
    ):
        chunks = [c async for c in router.generate({"prompt": "a dog"}, None)]

    assert chunks == [{"data": []}]


async def test_a_single_stage_deployment_refuses_a_session_request():
    """Streaming needs the DiT/VAE split; saying so beats a failure deeper in."""
    dit = _rollout(1)
    router = _router(dit, _decoder())
    router.stage_configs = [_stage_cfg(0)]

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert len(chunks) == 1
    assert "2-stage" in chunks[0]["error"]


# -- the pipelining --------------------------------------------------------


async def test_the_stages_actually_overlap():
    """The whole point. A serial implementation produces identical frames in identical
    order, so this is the only test that can tell the two apart: block 1 must reach the
    VAE stage before the DiT stage has finished emitting block 2.
    """
    timeline: list[str] = []

    def dit_handler(request):
        async def block(b):
            timeline.append(f"dit{b}")
            await asyncio.sleep(0.02)  # the DiT stage keeps rolling out
            return {
                "stage_connector_refs": {"0": {"name": f"lat{b}"}},
                "cf_session": "s1",
                "cf_block": b,
                "finished": True,
            }

        async def last():
            return {
                "cf_session": "s1",
                "cf_block": 3,
                "cf_last": True,
                "finished": True,
            }

        return [block(0), block(1), block(2), last()]

    def vae_handler(request):
        async def decode():
            b = request.get("cf_block")
            timeline.append(f"vae{b}")
            await asyncio.sleep(0.03)  # slower than the DiT stage, as measured
            return {
                "stage_connector_refs": {"1": {"name": f"frames{b}"}},
                "cf_session": "s1",
                "cf_block": b,
                "finished": True,
            }

        return [decode()]

    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(
        _StageClient(dit_handler),
        _StageClient(vae_handler),
        formatter,
        _frame_connector(),
    )

    _ = [c async for c in router.generate(_cf_request(), None)]

    # A serial chain would give dit0 dit1 dit2 vae0 vae1 vae2. Overlapped, the first
    # decode starts while the rollout is still producing.
    assert timeline.index("vae0") < timeline.index("dit2"), timeline


async def test_decodes_stay_in_stream_order():
    """The one thing that must not be parallelised: the VAE's temporal feat_cache makes
    block N's first frame depend on block N-1's last, so out-of-order decoding would not
    fail -- it would produce visible seams at every block boundary."""
    order: list[int] = []

    def vae_handler(request):
        async def decode():
            b = request.get("cf_block")
            # Later blocks return faster; if the router decoded concurrently, the
            # recorded order would invert.
            await asyncio.sleep(0.03 - 0.01 * b)
            if not request.get("cf_last"):
                order.append(b)
            return {
                "stage_connector_refs": {"1": {"name": "f"}},
                "cf_session": "s1",
                "cf_block": b,
                "finished": True,
            }

        return [decode()]

    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(
        _rollout(3), _StageClient(vae_handler), formatter, _frame_connector()
    )

    _ = [c async for c in router.generate(_cf_request(), None)]

    assert order == [0, 1, 2]


async def test_each_block_is_fetched_under_its_own_key():
    connector = _frame_connector()
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(_rollout(3), _decoder(), formatter, connector)

    _ = [c async for c in router.generate(_cf_request(), None)]

    keys = [call.args[2] for call in connector.get.call_args_list]
    assert keys == [cf_put_key(_request_id(router), b) for b in range(3)]


async def test_the_terminal_marker_is_forwarded_to_the_vae_stage():
    """It carries no frames, but the VAE worker needs it to release its decode cursor --
    otherwise the next stream inherits a warm feat_cache from the last one."""
    vae = _decoder()
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(_rollout(2), vae, formatter, _frame_connector())

    _ = [c async for c in router.generate(_cf_request(), None)]

    assert [r.get("cf_last") for r in vae.requests] == [None, None, True]


# -- frames and formatting -------------------------------------------------


async def test_frames_are_converted_from_the_decoder_range():
    """The decoder hands back [-1, 1]; the batch path's post-process func is bypassed,
    so the router has to do that conversion or every dark pixel clips to black."""
    captured = {}

    async def fake_format(frames, request_id, *, fps):
        captured["frames"] = frames
        captured["fps"] = fps
        return {"data": []}

    formatter = MagicMock()
    formatter.format_video_frames = fake_format
    router = _router(
        _rollout(2), _decoder(), formatter, _frame_connector(frames_per_block=2)
    )

    _ = [c async for c in router.generate(_cf_request(), None)]

    video = captured["frames"][0]
    assert video.shape == (4, 8, 8, 3), "2 blocks x 2 frames, concatenated in order"
    assert video.dtype == np.uint8
    # zeros in [-1, 1] are mid grey, not black.
    assert 126 <= int(video.min()) <= 129


async def test_fps_comes_from_the_request():
    captured = {}

    async def fake_format(frames, request_id, *, fps):
        captured["fps"] = fps
        return {"data": []}

    formatter = MagicMock()
    formatter.format_video_frames = fake_format
    router = _router(_rollout(1), _decoder(), formatter, _frame_connector())

    request = _cf_request()
    request["nvext"]["fps"] = 24
    _ = [c async for c in router.generate(request, None)]

    assert captured["fps"] == 24


async def test_cmaf_opt_in_streams_live_segments():
    """With the CMAF annotation the frames go out as they decode, which is what the
    pipelining is for -- segment 0 ships while later frames do not exist yet."""
    pieces = [{"cmaf": "metadata"}, {"cmaf": "init"}, {"cmaf": "segment"}]
    seen = {}

    def fake_live(frame_chunks, request_id, *, fps):
        seen["chunks"] = frame_chunks
        seen["fps"] = fps

        async def _gen():
            for p in pieces:
                yield p

        return _gen()

    formatter = MagicMock()
    formatter.stream_video_cmaf_live = fake_live
    router = _router(_rollout(2), _decoder(), formatter, _frame_connector())

    chunks = [c async for c in router.generate(_cf_request(cmaf=True), None)]

    assert chunks == pieces
    assert seen["fps"] == 16


async def test_an_empty_stream_is_reported_rather_than_returned_as_a_clip():
    """A rollout that produced nothing must not format as a zero-frame video."""
    dit = _StageClient(
        lambda r: [
            {"cf_session": "s1", "cf_block": 0, "cf_last": True, "finished": True}
        ]
    )
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(dit, _decoder(), formatter, _frame_connector())

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert "no frames" in chunks[-1]["error"]


# -- failures --------------------------------------------------------------


async def test_a_dit_stage_error_ends_the_stream():
    dit = _StageClient(lambda r: [{"error": "rollout blew up", "finished": True}])
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(dit, _decoder(), formatter, _frame_connector())

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert any("rollout blew up" in str(c) for c in chunks)


async def test_a_vae_stage_error_ends_the_stream():
    vae = _StageClient(lambda r: [{"error": "decode blew up", "finished": True}])
    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(_rollout(2), vae, formatter, _frame_connector())

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert any("decode blew up" in str(c) for c in chunks)


async def test_a_missing_router_connector_ref_is_an_error():
    """A pipelined stream cannot use the SHM-by-request-id fallback: every block would
    collide on one key."""

    def vae_handler(request):
        return [{"shm_meta": {"x": 1}, "cf_session": "s1", "finished": True}]

    formatter = MagicMock()
    formatter.format_video_frames = _async_return({"data": []})
    router = _router(
        _rollout(2), _StageClient(vae_handler), formatter, _frame_connector()
    )

    chunks = [c async for c in router.generate(_cf_request(), None)]

    assert any("connector ref" in str(c) for c in chunks)


async def test_the_producers_are_cancelled_when_the_consumer_stops_early():
    """A client disconnect leaves both producers blocked on a handoff nobody will drain
    again -- a leaked task per abandoned request."""
    def fake_live(frame_chunks, request_id, *, fps):
        async def _gen():
            yield {"cmaf": "init"}
            yield {"cmaf": "segment"}

        return _gen()

    formatter = MagicMock()
    formatter.stream_video_cmaf_live = fake_live
    router = _router(_rollout(50), _decoder(), formatter, _frame_connector())

    gen = router.generate(_cf_request(cmaf=True), None)
    assert await gen.__anext__() == {"cmaf": "init"}
    await gen.aclose()

    # Nothing left running: aclose() ran the finally, cancelling and awaiting both.
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []


# -- helpers ---------------------------------------------------------------


def _async_return(value):
    async def _f(*args, **kwargs):
        return value

    return _f


def _request_id(router):
    """The router mints its own uuid; recover it from the calls it made."""
    connector = router.connectors[stage_router._connector_key(1, "router")]
    return connector.get.call_args_list[0].args[2].rsplit("_b", 1)[0]
