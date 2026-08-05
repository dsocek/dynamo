# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-scoped Causal-Forcing streaming: worker-side dispatch.

Maps a parsed :class:`CFSessionRequest` onto the registry that lives in the engine's
worker process. :mod:`cf_session` decides *what* a request asked for; this decides
*which RPCs* deliver it.

The registry cannot live here. ``CausalForcingStream`` owns a device-resident KV
window, so it exists one process further down, inside ``WorkerProc``, reachable only
through ``collective_rpc``. That indirection buys a useful property: the engine drains
control RPCs between scheduler steps, so a session mutation can never interleave with
a rollout block that is mid-flight.

Two things about that transport are worth stating, because both shape the code below:

* It is **best-effort**. ``AsyncOmni.collective_rpc`` returns ``{"todo": ...}`` for a
  stage that does not support the call rather than raising, so a result that looks
  successful may mean nothing ran. Every call here is checked.
* Results are **pickled across a process boundary**, so ``session_drain``'s latents
  take a host round-trip (measured: ~585 KB per block at 480x832, against seconds of
  compute per block -- not free, but not the bottleneck either).

A session is opened by the first scene that declares a ``cut``, so a client never has to
send an explicit open and the frontend needs no place to put one. What it must not do is
*guess*: a ``continue`` whose session is missing is refused rather than opened, because
the window may have been released after sitting idle and rendering anyway would restart
the shot from noise. For the same reason the dispatcher asks the worker which sessions
are open instead of tracking them locally -- a local cache goes stale the moment one
expires, and that staleness is invisible until the video cuts back to the start.
"""

from __future__ import annotations

import logging
from typing import Any

from .cf_session import CFSessionRequest

logger = logging.getLogger(__name__)


class CFDispatchError(RuntimeError):
    """Raised when a session RPC could not be delivered or was refused.

    Separate from ``CFSessionRequestError`` (a malformed request, the client's fault)
    because this is a delivery or routing failure, which is ours.
    """


# Two structural levels sit between this dispatcher and the worker's return value:
#
#   worker      session_drain(...)                      -> [block, block]
#   executor    MultiprocDiffusionExecutor: only rank 0 has a result_mq, so
#               num_responses == 1, and with unique_reply_rank unset it returns
#               the response *list* rather than responses[0]
#                                                       -> [[block, block]]
#   stage pool  passes the stage result through unchanged
#   orchestrator  appends one result per live replica    -> [[[block, block]]]
#
# Both wrapper levels are plain lists, and so is what several of these methods
# legitimately return, so they cannot be told apart by inspecting them -- peeling
# "while it is a list" would eat a one-block drain's only block. They are peeled by
# count instead. ``unique_reply_rank`` would collapse the executor level at the
# source, but nothing on the async control-plane path sets it, so it is always there.
_RPC_WRAPPER_DEPTH = 2


class CFSessionDispatcher:
    """Issues session RPCs against one stage's engine.

    One instance per stage worker. Holds no session state of its own -- the worker
    process is the single source of truth, and asking it is cheap next to a rollout.
    """

    def __init__(self, engine: Any, stage_id: int) -> None:
        self.engine = engine
        self.stage_id = stage_id

    # -- transport ----------------------------------------------------------

    async def _rpc(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call one registry method, and fail loudly if it did not actually run.

        Unwraps the two transport levels described at :data:`_RPC_WRAPPER_DEPTH` and
        returns what the worker method itself returned. Only rank 0 replies, and an
        exception on any rank has already been raised by the executor, so there is one
        value to take rather than a quorum to check.
        """
        try:
            results = await self.engine.collective_rpc(
                method, args=args, kwargs=kwargs or None
            )
        except Exception as e:
            raise CFDispatchError(
                f"stage {self.stage_id}: collective_rpc({method!r}) failed: {e}"
            ) from e

        result = results
        for level in range(_RPC_WRAPPER_DEPTH):
            if not isinstance(result, list):
                # Fewer levels than expected. Returning the value anyway would hand
                # back something of the wrong shape and fail later, somewhere with no
                # transport context left to explain it -- as a list of latent blocks
                # arriving where one block tensor was expected once did.
                raise CFDispatchError(
                    f"stage {self.stage_id}: collective_rpc({method!r}) returned "
                    f"{type(result).__name__} at wrapper level {level}, expected a "
                    "list. The transport's nesting has changed; see "
                    "_RPC_WRAPPER_DEPTH."
                )
            if not result:
                raise CFDispatchError(
                    f"stage {self.stage_id}: collective_rpc({method!r}) returned no "
                    f"result at wrapper level {level}; the stage may not have any "
                    "worker registered."
                )
            result = result[0]

        # A {"todo": ...} result means the stage silently declined the call. Treating
        # it as success is the dangerous reading: the caller would believe a scene was
        # queued and then wonder why the video never changed.
        if isinstance(result, dict) and result.get("todo"):
            raise CFDispatchError(
                f"stage {self.stage_id}: collective_rpc({method!r}) is unsupported "
                f"here (returned {result!r}). The stage worker likely has no "
                "worker_extension_cls declaring CausalForcingSessionExtension."
            )
        return result

    # -- session lifecycle --------------------------------------------------

    async def open_sessions(self) -> list[str]:
        """Session ids currently open on this stage's worker."""
        info = await self._rpc("session_info")
        sessions = info.get("sessions", []) if isinstance(info, dict) else []
        return [s["session_id"] for s in sessions if isinstance(s, dict)]

    async def ensure_open(
        self, session_id: str, *, continues: bool = False, **open_kwargs: Any
    ) -> bool:
        """Open ``session_id`` unless the worker already has it. Returns True if opened.

        Asks rather than remembers: idle sessions expire on the worker on their own, so a
        cache here could claim a session is live when its KV window is already gone.

        ``continues`` is the one guard. A missing session may be missing because it sat
        idle and its KV window was released; opening a fresh one would render the scene
        against empty history, restarting the shot from noise with nothing in the output
        to say so. So a scene that means to continue refuses instead, and only a ``cut``
        -- which declares a new shot and has no history to lose -- may open.
        """
        if session_id in await self.open_sessions():
            return False
        if continues:
            raise CFDispatchError(
                f"Session {session_id!r} is not open on stage {self.stage_id}, so this scene "
                "cannot continue the shot: its KV window is gone (most likely expired after "
                "sitting idle) and rendering now would silently restart from noise. Send a scene "
                'with transition "cut" to start a new shot under this id.'
            )
        await self._rpc("session_open", session_id, **open_kwargs)
        logger.info(
            "[CF_DISPATCH] stage %d opened session %s", self.stage_id, session_id
        )
        return True

    async def push(self, session_id: str, prompt: str, **scene_kwargs: Any) -> dict:
        """Queue one scene. Cheap -- text encode only, no rollout."""
        return await self._rpc("session_push", session_id, prompt, **scene_kwargs)

    async def drain(self, session_id: str, *, max_blocks: int | None = None) -> list:
        """Roll out queued blocks. Yields latents on the DiT stage, pixels when aggregated."""
        return await self._rpc("session_drain", session_id, max_blocks=max_blocks)

    async def decode_step(self, session_id: str, latents: Any) -> Any:
        """Decode one chunk on a VAE-role session, keeping the temporal cache alive."""
        return await self._rpc("session_decode_step", session_id, latents)

    async def close(self, session_id: str, *, missing_ok: bool = False) -> dict:
        """Release a session and its caches."""
        result = await self._rpc("session_close", session_id, missing_ok=missing_ok)
        logger.info(
            "[CF_DISPATCH] stage %d closed session %s", self.stage_id, session_id
        )
        return result

    # -- the whole request, end to end --------------------------------------

    async def run_scene(
        self,
        req: CFSessionRequest,
        prompt: str,
        *,
        open_kwargs: dict[str, Any] | None = None,
        max_blocks: int | None = None,
    ) -> list:
        """Open-if-needed, push one scene, roll it out, and close if asked.

        Returns the blocks the rollout produced. Only a ``cut`` may open a session --
        see :meth:`ensure_open` for why a ``continue`` must fail instead. A storyboard's
        first scene therefore has to be a ``cut``, which is what ``cut`` means anyway.

        ``cf_close`` is honoured even when the rollout raises: a session that outlives
        its request holds a KV window and, because the VAE's temporal cache is
        per-module rather than per-session, blocks the whole card until it expires.
        """
        await self.ensure_open(
            req.session_id,
            continues=req.scene.transition == "continue",
            **(open_kwargs or {}),
        )
        try:
            await self.push(req.session_id, prompt, **req.scene.push_kwargs())
            return await self.drain(req.session_id, max_blocks=max_blocks)
        finally:
            if req.close:
                await self.close(req.session_id, missing_ok=True)
