# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CFSessionDispatcher.

Fake engines stand in for AsyncOmni, so these run without a model or a device. What is
worth pinning here is not that the happy path forwards arguments -- it is that a
best-effort transport cannot report success when nothing ran, because that failure is
invisible in the output: the client would get a video that never changed.
"""

import pytest
from conftest import as_transport_returns
from dynamo.vllm.omni.cf_dispatch import CFDispatchError, CFSessionDispatcher
from dynamo.vllm.omni.cf_session import CFSceneRequest, CFSessionRequest

pytestmark = pytest.mark.unit


class FakeEngine:
    """Records collective_rpc calls and replays scripted results."""

    def __init__(self, results=None, raises=None):
        self.calls = []
        self._results = results or {}
        self._raises = raises
        self.open = []

    def expire(self, session_id):
        """Drop a session the way an idle timeout does on the worker: behind our back."""
        self.open.remove(session_id)

    async def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, stage_ids=None
    ):
        self.calls.append((method, args, kwargs))
        if self._raises is not None:
            raise self._raises
        if method in self._results:
            value = self._results[method]
        elif method == "session_info" and not args:
            value = {"sessions": [{"session_id": s} for s in self.open]}
        elif method == "session_open":
            self.open.append(args[0])
            value = {"session_id": args[0], "role": "dit"}
        elif method == "session_close":
            if args[0] in self.open:
                self.open.remove(args[0])
            value = {"session_id": args[0], "closed": True}
        elif method == "session_drain":
            value = ["block0", "block1"]
        else:
            value = {"ok": True}
        return as_transport_returns(value)


def _dispatcher(**kwargs):
    engine = FakeEngine(**kwargs)
    return CFSessionDispatcher(engine, stage_id=0), engine


# -- the best-effort transport ---------------------------------------------


async def test_todo_result_is_an_error_not_a_silent_noop():
    """The failure this exists for: the RPC returns a dict, nothing ran, and the
    client is told its scene was queued."""
    d, _ = _dispatcher(results={"session_info": {"todo": "not implemented"}})
    with pytest.raises(CFDispatchError, match="unsupported"):
        await d.open_sessions()


async def test_error_message_names_the_likely_cause():
    """A missing worker_extension_cls is the overwhelmingly common cause, so the
    message should say so rather than leaving the reader to guess."""
    d, _ = _dispatcher(results={"session_info": {"todo": 1}})
    with pytest.raises(CFDispatchError, match="worker_extension_cls"):
        await d.open_sessions()


def _returning(value):
    """A dispatcher whose transport returns ``value`` verbatim, unwrapped."""
    engine = FakeEngine()

    async def verbatim(method, timeout=None, args=(), kwargs=None, stage_ids=None):
        return value

    engine.collective_rpc = verbatim
    return CFSessionDispatcher(engine, stage_id=0)


@pytest.mark.parametrize("empty_at", [[], [[]]])
async def test_an_empty_result_at_either_wrapper_level_is_an_error(empty_at):
    with pytest.raises(CFDispatchError, match="no result"):
        await _returning(empty_at).open_sessions()


async def test_transport_exception_is_wrapped_with_the_stage():
    d, _ = _dispatcher(raises=ConnectionError("engine gone"))
    d.stage_id = 3
    with pytest.raises(CFDispatchError, match="stage 3"):
        await d.open_sessions()


# -- the nesting itself ----------------------------------------------------
#
# The bug these exist for: the transport wraps the worker's value twice, the dispatcher
# peeled once, and every caller silently received one list too many. Nothing raised at
# the seam -- ``drain()`` returned a truthy list and ``blocks[0]`` was a list of blocks
# rather than a block, so the first symptom was ``'list' object has no attribute 'to'``
# raised deep inside the VAE's denormalization, with no transport context left in the
# traceback to point back here.


async def test_both_wrapper_levels_are_peeled():
    """A list return value is the dangerous case: peeling "while it is a list" would
    keep going and eat the payload, so the depth is fixed rather than inferred."""
    assert await _returning([[["block0", "block1"]]]).drain("s1") == [
        "block0",
        "block1",
    ]


async def test_a_single_block_drain_keeps_its_list():
    """The case a depth-guessing unwrap gets wrong: one block in, a one-element list
    out, and the block itself must not be mistaken for another wrapper."""
    assert await _returning([[["only-block"]]]).drain("s1") == ["only-block"]


async def test_an_empty_drain_is_an_empty_list_not_an_error():
    """A dry queue ends the request's rollout. It is the normal way a stream finishes,
    so it must survive the unwrap as a falsy value rather than tripping the guards."""
    assert await _returning([[[]]]).drain("s1") == []


@pytest.mark.parametrize(
    "shallow", [{"sessions": []}, [{"sessions": []}]], ids=["depth0", "depth1"]
)
async def test_too_few_wrapper_levels_is_a_loud_error(shallow):
    """Tolerating a shallower return would hand back a value of the wrong shape and
    fail somewhere with no transport context left to explain it. If the transport's
    nesting changes, this is where it should be noticed."""
    with pytest.raises(CFDispatchError, match="wrapper level"):
        await _returning(shallow).open_sessions()


# -- lifecycle -------------------------------------------------------------


async def test_ensure_open_opens_once_then_reuses():
    d, engine = _dispatcher()
    assert await d.ensure_open("s1") is True
    assert await d.ensure_open("s1") is False
    assert [c[0] for c in engine.calls].count("session_open") == 1


async def test_ensure_open_asks_the_worker_rather_than_caching():
    """Idle sessions expire on the worker on their own. A local cache would claim a
    session is live after its KV window is gone, so state comes from the worker every
    time."""
    d, engine = _dispatcher()
    await d.ensure_open("s1")
    engine.open.clear()  # it expired behind our back
    assert await d.ensure_open("s1") is True, "must re-open, not trust a stale cache"


async def test_continue_onto_an_expired_session_is_an_error():
    """The bug this exists for: the window expired, ensure_open saw absence and opened a
    fresh stream, and the scene rendered from noise -- no error anywhere, just a video
    that cut back to the start mid-shot."""
    d, engine = _dispatcher()
    await d.ensure_open("s1")
    engine.expire("s1")

    with pytest.raises(CFDispatchError, match="expired after sitting idle"):
        await d.ensure_open("s1", continues=True)
    # Crucially, it did not open one anyway on the way out.
    assert [c[0] for c in engine.calls].count("session_open") == 1


async def test_cut_may_reuse_an_expired_id():
    """A cut zeroes the attention history, so there is nothing to lose -- refusing here
    would strand the id and force the client to invent a new one to recover."""
    d, engine = _dispatcher()
    await d.ensure_open("s1")
    engine.expire("s1")
    assert await d.ensure_open("s1", continues=False) is True
    assert engine.open == ["s1"]


async def test_a_storyboard_must_open_with_a_cut():
    """The cost of the guard, stated as a test so it is not a surprise: a session's
    first scene cannot be a "continue", because there is nothing yet to continue."""
    d, engine = _dispatcher()
    with pytest.raises(CFDispatchError, match="not open"):
        await d.ensure_open("s1", continues=True)
    assert engine.open == []
    assert await d.ensure_open("s1") is True  # a cut opens it


async def test_run_scene_refuses_a_continue_after_an_expiry():
    d, engine = _dispatcher()
    await d.run_scene(CFSessionRequest("s1", CFSceneRequest(transition="cut")), "shot")
    engine.expire("s1")
    with pytest.raises(CFDispatchError, match="not open"):
        await d.run_scene(
            CFSessionRequest("s1", CFSceneRequest(transition="continue")), "next"
        )
    # It refused before pushing, so no scene was queued against an empty window.
    assert [c[0] for c in engine.calls].count("session_push") == 1


async def test_open_kwargs_are_forwarded():
    d, engine = _dispatcher()
    await d.ensure_open("s1", height=480, width=832, seed=1234)
    call = next(c for c in engine.calls if c[0] == "session_open")
    assert call[1] == ("s1",)
    assert call[2] == {"height": 480, "width": 832, "seed": 1234}


async def test_push_forwards_prompt_and_scene_kwargs():
    d, engine = _dispatcher()
    await d.push("s1", "a coral reef", transition="cut", latents=21, seed=7)
    call = next(c for c in engine.calls if c[0] == "session_push")
    assert call[1] == ("s1", "a coral reef")
    assert call[2] == {"transition": "cut", "latents": 21, "seed": 7}


async def test_drain_returns_the_blocks():
    d, _ = _dispatcher()
    assert await d.drain("s1") == ["block0", "block1"]


async def test_drain_forwards_max_blocks():
    d, engine = _dispatcher()
    await d.drain("s1", max_blocks=2)
    assert next(c for c in engine.calls if c[0] == "session_drain")[2] == {
        "max_blocks": 2
    }


async def test_decode_step_forwards_latents():
    d, engine = _dispatcher()
    await d.decode_step("s1", "latents-tensor")
    assert next(c for c in engine.calls if c[0] == "session_decode_step")[1] == (
        "s1",
        "latents-tensor",
    )


# -- run_scene, end to end -------------------------------------------------


async def test_run_scene_opens_pushes_drains_in_order():
    d, engine = _dispatcher()
    req = CFSessionRequest("s1", CFSceneRequest(transition="cut", seed=5))
    assert await d.run_scene(req, "a reef wall") == ["block0", "block1"]
    methods = [c[0] for c in engine.calls]
    assert methods.index("session_open") < methods.index("session_push")
    assert methods.index("session_push") < methods.index("session_drain")
    assert "session_close" not in methods  # close=False


async def test_run_scene_honours_close():
    d, engine = _dispatcher()
    req = CFSessionRequest("s1", CFSceneRequest(transition="cut"), close=True)
    await d.run_scene(req, "last shot")
    assert [c[0] for c in engine.calls][-1] == "session_close"
    assert engine.open == []


async def test_run_scene_closes_even_when_the_rollout_fails():
    """A leaked session pins a KV window, and the VAE's temporal cache is per-module,
    so one stranded session blocks the whole card until it expires."""
    engine = FakeEngine()
    d = CFSessionDispatcher(engine, stage_id=0)
    real = engine.collective_rpc

    async def fail_on_drain(method, timeout=None, args=(), kwargs=None, stage_ids=None):
        if method == "session_drain":
            raise RuntimeError("rollout blew up")
        return await real(method, timeout, args, kwargs, stage_ids)

    engine.collective_rpc = fail_on_drain
    with pytest.raises(CFDispatchError):
        await d.run_scene(
            CFSessionRequest("s1", CFSceneRequest(transition="cut"), close=True), "shot"
        )
    assert "session_close" in [c[0] for c in engine.calls]


async def test_run_scene_leaves_the_session_open_without_close():
    """The whole point of a session: the next request continues this shot."""
    d, engine = _dispatcher()
    await d.run_scene(CFSessionRequest("s1", CFSceneRequest(transition="cut")), "shot one")
    await d.run_scene(CFSessionRequest("s1", CFSceneRequest()), "shot two")
    assert [c[0] for c in engine.calls].count("session_open") == 1
    assert engine.open == ["s1"]
