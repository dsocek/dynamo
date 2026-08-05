# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the stage worker's Causal-Forcing streaming branches.

Fake engines and fake connectors, so these run without a model or a device. What is
worth pinning is the pipelining itself -- one chunk per block, drained one block at a
time -- because a version that drained everything first would pass any test that only
checked the frames, while giving back exactly the serial latency this work exists to
remove.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import as_transport_returns
from dynamo.vllm.omni.cf_pipeline import cf_put_key
from dynamo.vllm.omni.stage_worker import (
    OmniStageWorker,
    _cf_open_kwargs,
    cf_streaming_stage,
)
from dynamo.vllm.omni.types import StageRequest

pytestmark = pytest.mark.unit


class _SessionEngine:
    """Records collective_rpc traffic and replays a scripted rollout."""

    engine = None

    def __init__(self, blocks=3):
        self.calls = []
        self.open = []
        self._remaining = blocks

    def generate(self, prompt, request_id="", *, sampling_params_list=None):
        raise AssertionError("a session request must not go through engine.generate")

    async def get_tokenizer(self):
        return None

    async def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, stage_ids=None
    ):
        # as_transport_returns() applies the real transport's two wrapper levels; see
        # its docstring for why a fake that wraps only once is worse than no fake.
        self.calls.append((method, args, kwargs or {}))
        if method == "session_info":
            return as_transport_returns(
                {"sessions": [{"session_id": s} for s in self.open]}
            )
        if method == "session_open":
            self.open.append(args[0])
            return as_transport_returns({"session_id": args[0]})
        if method == "session_close":
            if args[0] in self.open:
                self.open.remove(args[0])
            return as_transport_returns({"closed": True})
        if method == "session_drain":
            max_blocks = (kwargs or {}).get("max_blocks")
            assert max_blocks == 1, "the pipeline must drain one block at a time"
            if self._remaining <= 0:
                return as_transport_returns([])
            self._remaining -= 1
            return as_transport_returns([f"latents-{self._remaining}"])
        if method == "session_decode_step":
            return as_transport_returns(f"frames-for-{args[1]}")
        return as_transport_returns({"ok": True})


class _Context:
    def id(self):
        return "ctx-req"


def _stage_config(**overrides):
    defaults = dict(
        stage_type="diffusion",
        final_output=False,
        final_output_type="image",
        engine_input_source=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _connector(put_metadata=None, get_value=None):
    c = MagicMock()
    c.put.return_value = (True, 0, put_metadata or {"name": "seg", "size": 1})
    c.get.return_value = get_value
    return c


def _dit_worker(engine, out_connector):
    return OmniStageWorker(
        engine=engine,
        stage_config=_stage_config(),
        connectors={("0", "1"): out_connector},
        stage_id=0,
    )


def _vae_worker(engine, in_connector, router_connector):
    return OmniStageWorker(
        engine=engine,
        stage_config=_stage_config(engine_input_source=[0]),
        connectors={("0", "1"): in_connector, ("1", "router"): router_connector},
        stage_id=1,
    )


def _session_request(session_id="s1", transition="cut", **extra):
    """A session request. ``cut`` by default because a stream's first scene has to be
    one -- ``transition`` defaults to ``continue`` on the wire, and continuing a
    session that is not open yet is refused (see the test below)."""
    annotations = [f"cf_session={session_id}"]
    if transition is not None:
        annotations.append(f'cf_scene={{"transition":"{transition}"}}')
    request = {
        "request_id": "req-1",
        "prompt": "a coral reef",
        "size": "832x480",
        "nvext": {"annotations": annotations},
    }
    request.update(extra)
    return request


# -- routing: which half of the stream is this ------------------------------


def test_stage_0_is_recognized_from_the_annotation():
    req = StageRequest.model_validate({"request_id": "r"})
    request = {"nvext": {"annotations": ["cf_session=s1"]}}
    assert cf_streaming_stage(req, request) == "dit"


def test_stage_n_is_recognized_from_the_hop_field():
    """nvext never reaches stage N>0, so the session id has to travel as a field."""
    req = StageRequest.model_validate({"request_id": "r", "cf_session": "s1"})
    assert cf_streaming_stage(req, {}) == "vae"


def test_an_ordinary_request_is_not_a_stream():
    """The property that keeps the batch path untouched."""
    req = StageRequest.model_validate({"request_id": "r"})
    assert cf_streaming_stage(req, {"prompt": "x", "nvext": {}}) is None
    assert cf_streaming_stage(req, {}) is None


# -- session open kwargs ---------------------------------------------------


def test_open_kwargs_come_from_the_request_size():
    kwargs = _cf_open_kwargs({"size": "832x480", "nvext": {}})
    assert kwargs == {"width": 832, "height": 480}


def test_open_kwargs_prefer_explicit_nvext_geometry():
    kwargs = _cf_open_kwargs(
        {"size": "832x480", "nvext": {"height": 720, "width": 1280}}
    )
    assert kwargs == {"width": 1280, "height": 720}


def test_open_kwargs_forward_the_seed_only_when_set():
    assert "seed" not in _cf_open_kwargs({"size": "832x480", "nvext": {}})
    assert _cf_open_kwargs({"nvext": {"seed": 7}})["seed"] == 7


def test_open_kwargs_fall_back_to_the_default_size():
    kwargs = _cf_open_kwargs({})
    assert kwargs["width"] > 0 and kwargs["height"] > 0


# -- stage 0: the rollout --------------------------------------------------


async def test_rollout_emits_one_chunk_per_block_then_a_terminal_marker():
    engine = _SessionEngine(blocks=3)
    connector = _connector()
    worker = _dit_worker(engine, connector)

    chunks = [c async for c in worker.generate(_session_request(), _Context())]

    blocks = [c for c in chunks if not c.get("cf_last")]
    assert [c["cf_block"] for c in blocks] == [0, 1, 2]
    assert all(c["cf_session"] == "s1" for c in chunks)
    assert chunks[-1] == {
        "cf_session": "s1",
        "cf_block": 3,
        "cf_last": True,
        "finished": True,
    }
    assert not any("error" in c for c in chunks)


async def test_rollout_drains_one_block_at_a_time():
    """This is the pipelining. Draining everything first would produce the same
    tensors and the same video, at the serial latency."""
    engine = _SessionEngine(blocks=2)
    worker = _dit_worker(engine, _connector())

    _ = [c async for c in worker.generate(_session_request(), _Context())]

    drains = [c for c in engine.calls if c[0] == "session_drain"]
    assert len(drains) == 3, "one drain per block, plus the one that finds it dry"
    assert all(c[2] == {"max_blocks": 1} for c in drains)


async def test_rollout_keys_each_block_separately():
    """Reusing the request id would have every block overwrite its predecessor's SHM
    segment while contending on one lockfile."""
    engine = _SessionEngine(blocks=3)
    connector = _connector()
    worker = _dit_worker(engine, connector)

    _ = [c async for c in worker.generate(_session_request(), _Context())]

    keys = [call.args[2] for call in connector.put.call_args_list]
    assert keys == [cf_put_key("req-1", b) for b in range(3)]


async def test_rollout_leaves_the_session_open_without_cf_close():
    """The whole point of a session: the next request continues the shot."""
    engine = _SessionEngine(blocks=1)
    worker = _dit_worker(engine, _connector())

    _ = [c async for c in worker.generate(_session_request(), _Context())]

    assert engine.open == ["s1"]
    assert "session_close" not in [c[0] for c in engine.calls]


async def test_rollout_closes_the_session_on_cf_close():
    engine = _SessionEngine(blocks=1)
    worker = _dit_worker(engine, _connector())
    request = _session_request()
    request["nvext"]["annotations"].append("cf_close")

    _ = [c async for c in worker.generate(request, _Context())]

    assert engine.open == []
    assert "session_close" in [c[0] for c in engine.calls]


async def test_rollout_rejects_an_empty_prompt():
    engine = _SessionEngine()
    worker = _dit_worker(engine, _connector())

    chunks = [
        c async for c in worker.generate(_session_request(prompt="  "), _Context())
    ]

    assert len(chunks) == 1
    assert "non-empty prompt" in chunks[0]["error"]
    assert chunks[0]["cf_last"] is True, "the consumer must learn the stream ended"


async def test_rollout_reports_a_malformed_scene_as_a_terminal_error():
    engine = _SessionEngine()
    worker = _dit_worker(engine, _connector())
    request = _session_request(transition=None)
    request["nvext"]["annotations"].append("cf_scene={not json}")

    chunks = [c async for c in worker.generate(request, _Context())]

    assert chunks[-1]["cf_last"] is True
    assert "error" in chunks[-1]


async def test_a_streams_first_scene_cannot_be_a_continue():
    """Carried through to the streaming path: continuing a session whose KV window does
    not exist would render from noise with no error anywhere."""
    engine = _SessionEngine(blocks=2)
    worker = _dit_worker(engine, _connector())

    chunks = [
        c
        async for c in worker.generate(
            _session_request(transition="continue"), _Context()
        )
    ]

    assert len(chunks) == 1
    assert "not open" in chunks[0]["error"]
    assert chunks[0]["cf_last"] is True
    assert "session_push" not in [c[0] for c in engine.calls]


async def test_rollout_without_a_downstream_connector_is_an_error_not_a_shm_fallback():
    """The batch path falls back to SHM keyed by request id. A stream cannot: every
    block would reuse the one key."""
    engine = _SessionEngine(blocks=2)
    worker = OmniStageWorker(
        engine=engine, stage_config=_stage_config(), connectors={}, stage_id=0
    )

    chunks = [c async for c in worker.generate(_session_request(), _Context())]

    assert "no connector" in chunks[-1]["error"]
    assert chunks[-1]["cf_last"] is True


async def test_a_failed_put_ends_the_stream_with_an_error():
    engine = _SessionEngine(blocks=2)
    connector = MagicMock()
    connector.put.return_value = (False, 0, None)
    worker = _dit_worker(engine, connector)

    chunks = [c async for c in worker.generate(_session_request(), _Context())]

    assert "connector.put() failed" in chunks[-1]["error"]


# -- stage 1: the decode ---------------------------------------------------


async def test_decode_reads_the_block_and_puts_frames_to_the_router():
    engine = _SessionEngine()
    in_connector = _connector(get_value="latents-0")
    router = _connector(put_metadata={"name": "frames", "size": 9})
    worker = _vae_worker(engine, in_connector, router)

    request = {
        "request_id": "req-1",
        "cf_session": "s1",
        "cf_block": 2,
        "stage_connector_refs": {"0": {"name": "ref0"}},
    }
    chunks = [c async for c in worker.generate(request, _Context())]

    in_connector.get.assert_called_once_with(
        "0", "1", cf_put_key("req-1", 2), metadata={"name": "ref0"}
    )
    decode = next(c for c in engine.calls if c[0] == "session_decode_step")
    assert decode[1] == ("s1", "latents-0")
    assert router.put.call_args.args[2] == cf_put_key("req-1", 2)
    assert chunks == [
        {
            "stage_connector_refs": {"1": {"name": "frames", "size": 9}},
            "cf_session": "s1",
            "cf_block": 2,
            "finished": True,
        }
    ]


async def test_decode_uses_the_session_cursor_not_the_batch_path():
    """session_decode_step keeps the decoder's feat_cache alive between blocks; decode
    through the batch path instead and every block boundary is a visible seam."""
    engine = _SessionEngine()
    worker = _vae_worker(engine, _connector(get_value="lat"), _connector())

    _ = [
        c
        async for c in worker.generate(
            {
                "request_id": "r",
                "cf_session": "s1",
                "cf_block": 0,
                "stage_connector_refs": {"0": {}},
            },
            _Context(),
        )
    ]

    assert "session_decode_step" in [c[0] for c in engine.calls]


async def test_decode_opens_a_cursor_on_any_block():
    """Unlike a rollout, a decode cursor has no history to lose -- which is what lets
    the VAE stage recover from having been restarted mid-stream."""
    engine = _SessionEngine()
    worker = _vae_worker(engine, _connector(get_value="lat"), _connector())

    _ = [
        c
        async for c in worker.generate(
            {
                "request_id": "r",
                "cf_session": "s1",
                "cf_block": 5,
                "stage_connector_refs": {"0": {}},
            },
            _Context(),
        )
    ]

    assert engine.open == ["s1"]


async def test_the_terminal_marker_releases_the_cursor_and_is_forwarded():
    engine = _SessionEngine()
    engine.open.append("s1")
    worker = _vae_worker(engine, _connector(), _connector())

    chunks = [
        c
        async for c in worker.generate(
            {"request_id": "r", "cf_session": "s1", "cf_block": 4, "cf_last": True},
            _Context(),
        )
    ]

    assert engine.open == []
    assert chunks == [
        {"cf_session": "s1", "cf_block": 4, "cf_last": True, "finished": True}
    ]
    assert "session_decode_step" not in [c[0] for c in engine.calls]


async def test_decode_reports_an_empty_upstream_payload():
    """An empty payload would otherwise decode to nothing and shorten the clip."""
    engine = _SessionEngine()
    worker = _vae_worker(engine, _connector(get_value=None), _connector())

    chunks = [
        c
        async for c in worker.generate(
            {
                "request_id": "r",
                "cf_session": "s1",
                "cf_block": 0,
                "stage_connector_refs": {"0": {}},
            },
            _Context(),
        )
    ]

    assert "empty payload" in chunks[-1]["error"]
    assert chunks[-1]["cf_last"] is True


async def test_decode_without_a_connector_ref_is_an_error():
    engine = _SessionEngine()
    worker = _vae_worker(engine, _connector(), _connector())

    chunks = [
        c
        async for c in worker.generate(
            {"request_id": "r", "cf_session": "s1", "cf_block": 0},
            _Context(),
        )
    ]

    assert "no connector ref" in chunks[-1]["error"]
