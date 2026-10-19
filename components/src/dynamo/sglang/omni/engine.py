# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One SGLang ``multimodal_gen`` instance, driven a chunk at a time.

A Dynamo stage worker holds exactly one SGLang instance, and the roles that
instance plays are decided entirely by its own ``--stage-roles``: a
``encoder,denoiser`` process answers with an ``OutputBatch.stage_handoff``
instead of pixels, and a ``decoder`` process turns that handoff back into frames.
Nothing in this module knows which of those it is talking to -- that is the point
of the split, and it is what lets one worker implementation serve both halves.

Two things are worth stating about the transport, because they shape the code:

*Requests go over the scheduler protocol, not through ``DiffGenerator.generate``.*
The generator is a batch API: it builds a ``Req`` from ``SamplingParams`` and
blocks until a whole clip is done. A Causal-Forcing stream is the opposite shape
-- one ``Req`` per chunk, each carrying the session id and the scene-relative
block index that tell the pipeline where in the shot it is -- so the worker
constructs those ``Req``s itself and posts them with
:class:`AsyncSchedulerClient`. ``DiffGenerator`` is still used, but only for its
*other* job: bringing the scheduler up (or attaching to one) and warming it.

*Sessions are pinned by construction.* SGLang's ``RealtimeSessionCache`` lives
inside the instance and is keyed by ``req.realtime_session_id``; the DiT KV window
and the VAE conv cache both live there. So a session's chunks must all reach the
same instance -- which they do, because a Dynamo stage worker is one instance, and
the router pins a session to one worker (see
:meth:`~dynamo.sglang.omni.stage_router.OmniSglangStageRouter._pin`). There is no
worker-side registry to keep in sync and no RPC to route: the state is wherever
the first chunk of the session went.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SglangOmniEngineError(RuntimeError):
    """Raised when the SGLang instance refused or failed a chunk.

    Distinct from a malformed client request: this is a failure of ours, on our
    side of the scheduler socket, and retrying the same request would not fix it.
    """


class SglangOmniEngine:
    """Async facade over one SGLang ``multimodal_gen`` scheduler."""

    def __init__(self, server_args: Any, generator: Any, client: Any) -> None:
        self.server_args = server_args
        self._generator = generator
        self._client = client

    @classmethod
    def launch(cls, server_args: Any, *, remote: bool = False) -> "SglangOmniEngine":
        """Bring up an instance for ``server_args``, or attach to a running one.

        ``remote=False`` is the deployed arrangement: ``DiffGenerator`` forks the
        scheduler (and its GPU worker) as children of this process, so the stage
        worker and its engine share a lifetime and Dynamo's graceful shutdown
        reaps both. ``remote=True`` attaches to a scheduler someone else started,
        which is how a single stage gets debugged with `sglang serve` in one
        terminal and the Dynamo worker in another.

        Either way the *client* is ours and asynchronous. ``DiffGenerator``
        installs a synchronous singleton for its own batch API; using that from a
        handler would block the event loop for the whole chunk -- seconds -- and
        with it every other request this worker is serving.
        """
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
        from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient

        generator = DiffGenerator.from_server_args(server_args, local_mode=not remote)
        client = AsyncSchedulerClient()
        client.initialize(server_args)
        logger.info(
            "SGLang instance ready at %s (roles=%s, %s)",
            server_args.scheduler_endpoint,
            getattr(server_args, "stage_roles", None) or "all",
            "attached" if remote else "launched",
        )
        return cls(server_args, generator, client)

    @property
    def chunk_size(self) -> int:
        """Latent frames per block, as the pipeline config declares it.

        Read from ``pipeline_config`` rather than from the DiT's ``arch_config``
        for the same reason the realtime adapter does: this process never loads
        ``transformer/config.json``, so the arch lookup would answer with the
        generic Wan default of 3 latent frames per block instead of
        Causal-Forcing's 1 -- and a stream driven at the wrong chunk size does
        not fail, it renders the wrong number of frames per block.
        """
        pipeline_config = self.server_args.pipeline_config
        chunk_size = getattr(pipeline_config, "realtime_chunk_size", None)
        if chunk_size is None:
            # Every realtime-capable config declares this. A config that does not
            # is not a Causal-Forcing pipeline -- usually a model path that resolved
            # to the generic Wan/T2V config -- and it would not stream at all.
            raise SglangOmniEngineError(
                f"{type(pipeline_config).__name__} declares no realtime_chunk_size, "
                "so this model does not support realtime streaming. Check "
                "--model-path: a Causal-Forcing checkpoint resolves to a "
                "CausalForcing* pipeline config."
            )
        return int(chunk_size)

    async def forward(self, payload: Any) -> Any:
        """Post one request to the scheduler and return the ``OutputBatch``.

        ``payload`` is either a ``list[Req]`` -- one generation batch, which for a
        Causal-Forcing stream is a single chunk -- or one of the scheduler's
        control requests (see :meth:`release_session`). The scheduler dispatches on
        the object's type, so this is one socket and one protocol either way.

        Raises rather than returning a batch that carries ``error``: an errored
        ``OutputBatch`` has no payload, so every caller would have to check, and
        the one that forgot would put an empty block on the wire and report a
        short video instead of a failure.
        """
        try:
            output = await self._client.forward(payload)
        except Exception as e:
            raise SglangOmniEngineError(
                f"scheduler at {self.server_args.scheduler_endpoint} failed the "
                f"request: {e}"
            ) from e
        if getattr(output, "error", None):
            raise SglangOmniEngineError(f"SGLang pipeline error: {output.error}")
        return output

    async def release_session(self, session_id: str) -> bool:
        """Drop this instance's state for ``session_id``; True if there was any.

        The engine holds a *single* realtime session per instance
        (``RealtimeSessionCache(max_sessions=1)`` in ``gpu_worker.py``), so the
        cache would evict the previous session anyway the moment a new one opened.
        Releasing explicitly still matters for two reasons: it frees the KV window
        (and, on the decode stage, the VAE conv cache) at the end of the shot
        rather than at the start of the next one, and it empties the allocator
        cache -- which is the difference between a card that has room for the next
        session and one that only finds it after an OOM.
        """
        from sglang.multimodal_gen.runtime.entrypoints.utils import (
            ReleaseRealtimeSessionReq,
        )

        output = await self.forward(ReleaseRealtimeSessionReq(session_id=session_id))
        result = getattr(output, "output", None)
        return bool(result.get("released")) if isinstance(result, dict) else False

    def shutdown(self) -> None:
        """Tear the instance down, tolerating a partially-built one.

        Called from a ``finally``, which may run after ``launch`` itself failed,
        so nothing here may assume it has anything to close. A scheduler child
        left behind would hold the GPU and the scheduler port, which is what
        makes the *next* launch fail with something unrelated-looking.
        """
        for name, closer in (
            ("scheduler client", getattr(self._client, "close", None)),
            ("SGLang instance", getattr(self._generator, "shutdown", None)),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception as e:  # noqa: BLE001 -- shutdown is best-effort
                logger.warning("Failed to close %s cleanly: %s", name, e)
