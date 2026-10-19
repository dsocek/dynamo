# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One stage of the disaggregated Causal-Forcing pipeline, as a Dynamo worker.

A stage worker is a thin thing on purpose: it owns one SGLang instance and serves
one internal endpoint, ``{namespace}.{stage_name}.generate``. It registers no
model, so the frontend never offers it to a client -- only the stage router
addresses it, and it does so by name.

The same class serves both halves of the split, and which half it is is not
configuration. It is read off the *shape of the request*, exactly as the
vLLM-Omni worker reads it, because the session id reaches the two halves by
different routes and neither route exists at both:

* The **rollout** half is handed the client's scene, ``nvext`` annotations and
  all, and answers with one response per block for the whole scene.
* The **decode** half is handed one block at a time -- a reference to the
  rollout's handoff plus the session and block it belongs to -- because ``nvext``
  is not forwarded past the first hop, and because one request per block is what
  lets block N decode while block N+1 rolls out.

What comes back from the SGLang instance is likewise not something this worker is
told in advance. An instance whose ``--stage-roles`` stop before the decoder
answers with a ``StageHandoff``; one that decodes answers with pixels. So the
worker publishes whichever it got and labels the reference with its kind, and the
router routes on that label. That single fact is what makes the 2-stage split, the
aggregated 1-stage deploy and the decode hop all one code path -- and it is why
adding a third role later would not need a change here.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import replace
from typing import Any, AsyncGenerator

from dynamo.common.omni.cf_session import (
    CFSessionRequestError,
    has_cf_session,
    parse_cf_session,
)
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.omni import blocks
from dynamo.sglang.omni.args import OmniSglangConfig
from dynamo.sglang.omni.cf_dispatch import (
    CFChunkDispatcher,
    CFDispatchError,
    SceneGeometry,
    frames_from_output,
    handoff_payload,
    scene_chunk_count,
    scene_geometry,
)
from dynamo.sglang.omni.engine import SglangOmniEngine, SglangOmniEngineError

logger = logging.getLogger(__name__)


class OmniSglangStageWorker:
    """Serves Causal-Forcing blocks from one SGLang instance."""

    def __init__(
        self,
        engine: SglangOmniEngine,
        config: OmniSglangConfig,
        stage_id: int,
    ) -> None:
        self.config = config
        self.stage_id = stage_id
        self.dispatcher = CFChunkDispatcher(engine, stage_id)

    async def generate(
        self, request: dict, context: Any = None
    ) -> AsyncGenerator[dict, None]:
        """Serve one stage request, yielding one response per block.

        Errors are *yielded*, not raised. A raised exception aborts the stream
        with a transport-level failure, which reaches the router as "the stage
        stopped" -- indistinguishable from a scene that simply ended, and for
        video the two look identical downstream: a short clip. A terminal error
        chunk lets the router fail the request with the reason attached.
        """
        request_id = str(request.get("cf_request_id") or uuid.uuid4().hex[:12])
        gen = (
            self._decode(request, request_id)
            if self._is_decode_request(request)
            else self._rollout(request, request_id)
        )
        try:
            async for chunk in gen:
                yield chunk
        except CFSessionRequestError as e:
            # The client's request does not describe a runnable scene.
            logger.warning("Stage %d: malformed CF request: %s", self.stage_id, e)
            yield self._failure(str(e))
        except (CFDispatchError, SglangOmniEngineError) as e:
            # Ours: the chunk was not delivered, or the instance refused it.
            logger.error("Stage %d: CF dispatch failed: %s", self.stage_id, e)
            yield self._failure(str(e))
        except Exception as e:  # noqa: BLE001 -- the router needs a reason, always
            logger.error(
                "Stage %d: CF stream failed for %s: %s",
                self.stage_id,
                request_id,
                e,
                exc_info=True,
            )
            yield self._failure(f"{type(e).__name__}: {e}")

    @staticmethod
    def _is_decode_request(request: dict) -> bool:
        """Whether this request is a block to decode rather than a scene to roll out.

        A decode request carries a block reference (or the terminal marker that
        follows the last one) and no annotations; a rollout request is the client's
        own request and carries the session annotation. Reading the shape rather
        than trusting ``--stage-id`` means a misconfigured deployment fails on the
        first request with something legible instead of silently decoding noise.
        """
        if request.get("cf_ref") is not None or request.get("cf_last"):
            return True
        return not has_cf_session(request.get("nvext"))

    def _failure(self, message: str) -> dict:
        return {
            "error": message,
            "cf_last": True,
            "finished": True,
            "cf_stage": self.stage_id,
        }

    # -- rollout half -------------------------------------------------------

    async def _rollout(
        self, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Roll one scene out block by block, publishing each block as it lands.

        The block count is known up front here, unlike on the vLLM-Omni path where
        the rollout is drained until dry: an SGLang realtime stream is driven by
        *sending* chunks, so how many chunks a scene is worth is the caller's
        decision rather than the pipeline's. That is a better fit for a storyboard
        -- a scene is a stated number of latent frames -- and it means the router
        knows the length of the shot before the first block arrives.
        """
        cf = parse_cf_session(request.get("nvext"))
        if cf is None:  # pragma: no cover -- _is_decode_request established this
            raise CFSessionRequestError(
                "a rollout request must carry a cf_session annotation"
            )
        prompt = (request.get("prompt") or "").strip()
        if not prompt:
            raise CFSessionRequestError("a session scene needs a non-empty prompt.")

        geometry = scene_geometry(request, self.config.default_video_fps)
        chunk_size = self.dispatcher.chunk_size
        # Raises for a `continue` with no shot in progress, before any work is
        # scheduled -- so a client that lost its session is told, rather than
        # served a new shot under the old id.
        self.dispatcher.book.begin_scene(cf.session_id, cf.scene.transition)
        # ``cf_scene``'s seed is the per-scene field of the streaming protocol, so
        # it outranks a request-level ``nvext.seed``; with neither, the session's
        # own seed is used or drawn. Substituted back into the geometry because
        # that is what every chunk of the scene is built from.
        geometry = replace(
            geometry,
            seed=self.dispatcher.book.seed_for_scene(
                cf.session_id,
                cf.scene.seed if cf.scene.seed is not None else geometry.seed,
                cf.scene.transition,
            ),
        )
        chunks = scene_chunk_count(cf.scene, geometry, chunk_size)

        logger.info(
            "[CF_STREAM] stage %d session %s: %s scene, %d chunk(s) of %d latent(s) "
            "at %dx%d, seed %d",
            self.stage_id,
            cf.session_id,
            cf.scene.transition,
            chunks,
            chunk_size,
            geometry.width,
            geometry.height,
            geometry.seed,
        )

        emitted = 0
        try:
            for _ in range(chunks):
                block = self.dispatcher.book.take(cf.session_id)
                yield await self._run_block(
                    cf.session_id, block, prompt, geometry, request_id
                )
                emitted += 1
        finally:
            if cf.close:
                # Honoured even on failure. A session that outlives its request
                # holds the DiT's KV window on the card, and on this backend the
                # cache holds exactly one session -- so a stale one is not merely
                # untidy, it is the memory the next stream needs.
                await self.dispatcher.release(cf.session_id)

        # A terminal marker of its own rather than a flag on the last block, so
        # the router can tell "the scene ended" from "the stage died" without
        # having to count blocks it was never told the total of.
        yield {
            "cf_session": cf.session_id,
            "cf_block": emitted,
            "cf_last": True,
            "cf_close": cf.close,
            "finished": True,
            "cf_stage": self.stage_id,
        }

    async def _run_block(
        self,
        session_id: str,
        block: int,
        prompt: str,
        geometry: SceneGeometry,
        request_id: str,
    ) -> dict:
        """Run one chunk and publish whatever this instance's roles produced."""
        t0 = time.monotonic()
        output = await self.dispatcher.rollout_chunk(
            session_id, block, prompt, geometry
        )
        t_run = time.monotonic()
        ref = await asyncio.to_thread(self._publish, output, request_id, block)
        t_put = time.monotonic()

        logger.info(
            "[CF_BLOCK_TIME] stage %d session %s block %d: run=%.3fs put=%.3fs "
            "kind=%s bytes=%d",
            self.stage_id,
            session_id,
            block,
            t_run - t0,
            t_put - t_run,
            ref["kind"],
            ref["nbytes"],
        )
        return {
            "cf_session": session_id,
            "cf_block": block,
            "cf_ref": ref,
            "cf_stage": self.stage_id,
            "cf_fps": geometry.fps,
            "finished": True,
        }

    def _publish(self, output: Any, request_id: str, block: int) -> dict[str, Any]:
        """Publish a block's payload, labelled with what it turned out to be.

        Runs on a worker thread: assembling frames copies a few megabytes and
        writing them to ``/dev/shm`` is a syscall against tmpfs, and neither
        belongs on the event loop of a process that is simultaneously the endpoint
        for the next block.
        """
        payload = handoff_payload(output)
        if payload is not None:
            key = blocks.block_key(request_id, block, blocks.KIND_HANDOFF)
            return blocks.put_handoff(key, payload)
        key = blocks.block_key(request_id, block, blocks.KIND_FRAMES)
        return blocks.put_frames(key, frames_from_output(output))

    # -- decode half --------------------------------------------------------

    async def _decode(
        self, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Decode one block through the session's live conv cache.

        One request per block, so this is a single decode and a single publish --
        the pipelining belongs to the router. What makes it a *session* decode
        rather than an ordinary one is that the handoff carries the session id and
        the block index, so SGLang continues the same ``RealtimeSession`` instead
        of opening a new one. Decode each block independently and every block
        boundary would be a visible seam.
        """
        session_id = str(request.get("cf_session") or "")
        block = int(request.get("cf_block") or 0)

        if request.get("cf_last"):
            # The rollout's terminal marker. Nothing to decode; release the
            # session's conv cache only if the scene asked to be closed, since a
            # session that will be continued needs it to still be there.
            if request.get("cf_close"):
                await self.dispatcher.release(session_id)
            yield {
                "cf_session": session_id,
                "cf_block": block,
                "cf_last": True,
                "finished": True,
                "cf_stage": self.stage_id,
            }
            return

        ref = request.get("cf_ref")
        if not isinstance(ref, dict) or ref.get("kind") != blocks.KIND_HANDOFF:
            raise CFDispatchError(
                f"decode stage expected a {blocks.KIND_HANDOFF!r} block reference, "
                f"got {ref!r}. A reference of kind {blocks.KIND_FRAMES!r} means the "
                "upstream instance already decoded this block, so it should have "
                "gone to the encoder rather than here -- check --stage-roles."
            )

        t0 = time.monotonic()
        payload = await asyncio.to_thread(blocks.take_handoff, ref)
        t_get = time.monotonic()
        output = await self.dispatcher.decode_chunk(payload)
        t_decode = time.monotonic()
        out_ref = await asyncio.to_thread(self._publish, output, request_id, block)
        t_put = time.monotonic()

        logger.info(
            "[CF_BLOCK_TIME] stage %d session %s block %d: get=%.3fs decode=%.3fs "
            "put=%.3fs frames=%s",
            self.stage_id,
            session_id,
            block,
            t_get - t0,
            t_decode - t_get,
            t_put - t_decode,
            out_ref.get("shape"),
        )
        yield {
            "cf_session": session_id,
            "cf_block": block,
            "cf_ref": out_ref,
            "cf_stage": self.stage_id,
            "finished": True,
        }


async def init_omni_stage(
    runtime: DistributedRuntime,
    config: OmniSglangConfig,
    shutdown_endpoints: list,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Bring up one stage worker: an SGLang instance behind an internal endpoint."""
    if config.stage_id is None:
        raise ValueError("--stage-id is required for a stage worker")
    stage_id = config.stage_id
    stage_name = config.stage_name

    # A run that died between publishing a block and its consumer taking it leaves
    # the payload resident in /dev/shm. Swept once at startup rather than
    # periodically: the only thing that produces residue is a crash, and startup is
    # when the next run can still be spared the memory.
    blocks.sweep()

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{stage_name}.{config.endpoint}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    # Synchronous on purpose: SGLang forks the scheduler and its GPU worker here,
    # and process setup wants the main thread. Nothing else is being served yet, so
    # blocking the loop for the load and warmup costs nothing -- the router waits
    # for this endpoint to appear before it addresses it.
    engine = SglangOmniEngine.launch(config.server_args, remote=config.sglang_remote)
    worker = OmniSglangStageWorker(engine, config, stage_id)

    logger.info(
        "Stage %d (%s): serving internal endpoint '%s.%s.%s'; no model registered",
        stage_id,
        "rollout" if stage_id == 0 else "decode",
        config.namespace,
        stage_name,
        config.endpoint,
    )
    try:
        await generate_endpoint.serve_endpoint(
            worker.generate,
            graceful_shutdown=True,
            metrics_labels=[],
        )
    except Exception as e:
        logger.error("Stage %d: endpoint failed: %s", stage_id, e)
        raise
    finally:
        # The scheduler is a child of this process and holds a GPU. Left behind, it
        # makes the *next* launch fail with an unrelated-looking OOM or a bound port.
        engine.shutdown()
