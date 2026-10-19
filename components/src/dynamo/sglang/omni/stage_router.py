# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The user-visible front of the disaggregated Causal-Forcing pipeline.

The router is the only model the frontend knows about. It owns no engine and no
weights: it addresses the stage workers by name, pipelines them, and packages what
comes back. Everything expensive happens elsewhere, which is what lets it stay a
single process while the stages scale with the cards.

Its one real job is to turn three *phases* into three overlapping *stages*. Run
serially, a block costs a rollout plus a decode plus an encode; pipelined, it costs
the slowest of the three, and the encode disappears under the other two. On the
vLLM-Omni path the measured per-latent cost was 0.957 s of DiT against 1.290 s of
VAE, so the overlap is worth roughly the DiT's whole contribution. Three tasks,
chained by the handoffs in :mod:`dynamo.common.omni.cf_pipeline`::

    rollout --Relay--> decode --FrameQueue--> encoder thread --> client

The decode leg is the one that must stay strictly ordered: the VAE's temporal
cache makes block N's first frame depend on block N-1's last, so blocks are awaited
in turn. Reordering there would not fail -- it would produce visible seams, which
is the kind of bug that ships.

An **aggregated** deploy (``--num-stages 1``) uses the same lane with the decode
hop absent rather than replaced: a worker holding all three roles answers with
pixels, so the reference it publishes is already frames and the second RPC is
skipped. Encoder overlap remains, DiT/VAE overlap does not -- which is exactly what
makes the two configurations comparable over one wire protocol.

Two things about session affinity, both consequences of where SGLang keeps its
realtime state. The KV window and the conv cache live *inside* an instance, keyed
by ``realtime_session_id``, so every chunk of a session must reach the same
instance -- hence :meth:`OmniSglangStageRouter._instance_for` pinning a session to
one instance with ``client.direct`` rather than letting round-robin scatter it. And
that instance holds exactly one session at a time
(``RealtimeSessionCache(max_sessions=1)``), so concurrency across sessions is
bounded by the number of stage workers, not by memory. The router therefore prefers
an idle instance and says so in the log when there is none.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import zlib
from typing import Any, AsyncGenerator, Optional

import numpy as np

from dynamo.common.omni.cf_pipeline import FrameQueue, Relay
from dynamo.common.omni.cf_session import (
    CFSessionRequestError,
    build_cf_annotations,
    has_cf_session,
    parse_cf_session,
)
from dynamo.common.omni.cmaf_video import CMAF_ANNOTATION
from dynamo.common.storage import get_fs
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.omni import blocks
from dynamo.sglang.omni.args import OmniSglangConfig
from dynamo.sglang.omni.output_formatter import VideoFormatter

logger = logging.getLogger(__name__)


class OmniSglangStageRouter:
    """Pipelines the stage workers and packages their frames for the frontend."""

    def __init__(self, config: OmniSglangConfig) -> None:
        self.config = config
        self.stage_names = config.resolved_stage_names
        self.stage_clients: dict[str, Any] = {}
        # (stage name, session id) -> instance id. The session's state lives on
        # that instance; sending its next chunk anywhere else would either fail
        # loudly (unknown session, block_idx > 0) or, worse, start a new shot.
        self._pinned: dict[tuple[str, str], int] = {}

        media_fs = (
            get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
        )
        self._formatter = VideoFormatter(
            model_name=config.served_model_name or config.model,
            media_fs=media_fs,
            media_http_url=config.media_output_http_url,
            default_fps=config.default_video_fps,
        )

    def set_stage_client(self, stage_name: str, client: Any) -> None:
        self.stage_clients[stage_name] = client
        logger.info("Router: registered stage client '%s'", stage_name)

    # -- request entry ------------------------------------------------------

    async def generate(
        self,
        request: dict,
        context: Any = None,  # noqa: ARG002 -- the router mints its own request id
    ) -> AsyncGenerator[dict, None]:
        """Serve one video request, session-scoped or one-shot.

        Both go down the same lane. A request with no session annotation is given
        one -- a single scene, cut, closed at the end -- rather than being served by
        a separate code path: the pipeline is the same shape either way, and the
        alternative is two implementations of it where only one gets exercised.
        """
        request_id = uuid.uuid4().hex[:12]
        try:
            request = self._ensure_session(request, request_id)
        except CFSessionRequestError as e:
            logger.warning("Router: malformed session request: %s", e)
            yield self._formatter.failure(request_id, str(e))
            return

        # Closed explicitly rather than left to ``async for``: when the client
        # disconnects, closing *this* generator does not close the one it delegates
        # to. Until the event loop's finalizer runs it, the stream's two producer
        # tasks are still alive with nobody draining them.
        stream = self._generate_cf_stream(request, request_id)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    def _ensure_session(self, request: dict, request_id: str) -> dict:
        """Give a plain request an implicit single-scene session.

        Keeps ``POST /v1/videos`` working unchanged: the client sees one clip and
        never learns a session existed. ``close=True`` is what makes it safe -- the
        implicit session releases its window when the request ends, so a one-shot
        caller cannot leave a card holding state it will never continue.
        """
        nvext = request.get("nvext")
        nvext = dict(nvext) if isinstance(nvext, dict) else {}
        if has_cf_session(nvext):
            # Validate here so a malformed scene is refused before any stage is
            # dispatched, with the client's own error rather than a stage's.
            parse_cf_session(nvext)
            return request

        existing = [a for a in (nvext.get("annotations") or []) if isinstance(a, str)]
        nvext["annotations"] = existing + build_cf_annotations(
            f"auto-{request_id}", transition="cut", close=True
        )
        return {**request, "nvext": nvext}

    # -- the pipelined lane -------------------------------------------------

    async def _generate_cf_stream(
        self, request: dict, request_id: str
    ) -> AsyncGenerator[dict, None]:
        """Run the rollout, the decode and the encode concurrently."""
        cf = parse_cf_session(request.get("nvext"))
        assert cf is not None  # _ensure_session guarantees it
        session_id = cf.session_id

        rollout_stage = self.stage_names[0]
        decode_stage = self.stage_names[1] if len(self.stage_names) >= 2 else None
        rollout_client = self.stage_clients.get(rollout_stage)
        decode_client = (
            self.stage_clients.get(decode_stage) if decode_stage is not None else None
        )
        if rollout_client is None or (decode_stage is not None and decode_client is None):
            yield self._formatter.failure(
                request_id, "Router: a stage client is missing; the pipeline is not ready"
            )
            return

        nvext = request.get("nvext") or {}
        fps = int(nvext.get("fps") or self.config.default_video_fps)
        cmaf = CMAF_ANNOTATION in (nvext.get("annotations") or [])

        relay = Relay()
        frames = FrameQueue()
        started = time.monotonic()

        async def rollout() -> None:
            """Forward each rolled-out block to the decode task as it lands."""
            try:
                stage_request = {"cf_request_id": request_id, **request}
                async for data in self._stage_chunks(
                    rollout_stage, rollout_client, session_id, stage_request
                ):
                    if data.get("error"):
                        raise RuntimeError(f"rollout stage: {data['error']}")
                    await relay.put(data)
                    if data.get("cf_last"):
                        break
            except BaseException as e:  # noqa: BLE001 -- re-raised in the consumer
                relay.close(e)
            else:
                relay.close()

        async def decode() -> None:
            """Turn each block into pixels, in order, feeding the encoder as it goes."""
            try:
                async for data in relay:
                    if data.get("cf_last"):
                        await self._finish_decode(
                            decode_stage, decode_client, session_id, request_id, data
                        )
                        break
                    frames.put(
                        await self._pixels_for(
                            decode_stage, decode_client, session_id, request_id, data
                        )
                    )
            except BaseException as e:  # noqa: BLE001 -- re-raised in the consumer
                frames.close(e)
            else:
                frames.close()

        rollout_task = asyncio.create_task(rollout())
        decode_task = asyncio.create_task(decode())
        try:
            if cmaf:
                # The live formatter reports its own failures as a CMAF failure
                # frame: by then the client holds a partial asset and needs to be
                # told it is partial rather than complete.
                async for chunk in self._formatter.stream_video_cmaf_live(
                    frames, request_id, fps=fps
                ):
                    yield chunk
            else:
                async for chunk in self._collect_clip(frames, request_id, fps):
                    yield chunk
        finally:
            # The consumer may have stopped early (client disconnect), which leaves
            # both producers blocked on a handoff nobody will drain again.
            for task in (rollout_task, decode_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(rollout_task, decode_task, return_exceptions=True)
            if cf.close:
                self._unpin(session_id)
            logger.info(
                "[CF_STREAM] router: session %s request %s done in %.2fs (cmaf=%s)",
                session_id,
                request_id,
                time.monotonic() - started,
                cmaf,
            )

    async def _collect_clip(
        self, frames: FrameQueue, request_id: str, fps: int
    ) -> AsyncGenerator[dict, None]:
        """Drain the stream and answer as one mp4.

        The pipelining still happened -- the stages overlapped exactly as in the
        live path -- only the delivery waits for the end. Draining runs on a thread
        because :class:`FrameQueue` is a blocking iterable by design; the encoder is
        the usual consumer and it pulls from its own thread.
        """
        try:
            collected = await asyncio.to_thread(list, frames)
        except Exception as e:  # noqa: BLE001 -- a producer's terminal error
            # It has to become a chunk: an exception out of this generator would
            # reach the client as a dropped connection, indistinguishable from a
            # network fault.
            logger.error("Router: CF stream failed for %s: %s", request_id, e)
            yield self._formatter.failure(request_id, str(e))
            return
        if not collected:
            yield self._formatter.failure(request_id, "the stream produced no frames")
            return
        yield await self._formatter.format_video(
            np.concatenate(collected, axis=0), request_id, fps=fps
        )

    async def _pixels_for(
        self,
        decode_stage: Optional[str],
        decode_client: Any,
        session_id: str,
        request_id: str,
        data: dict,
    ) -> np.ndarray:
        """Get one block's pixels, dispatching a decode hop only if it needs one.

        The kind of reference the rollout published decides this, not the stage
        count: a worker holding the decoder role already produced frames, so the
        block is read straight off the transport. That is the aggregated lane, and
        it is the same three lines as the disaggregated one minus the RPC.
        """
        ref = data.get("cf_ref")
        if not isinstance(ref, dict):
            raise RuntimeError(
                f"block {data.get('cf_block')} carried no payload reference"
            )
        if ref.get("kind") == blocks.KIND_FRAMES:
            return await asyncio.to_thread(blocks.take_frames, ref)

        if decode_client is None or decode_stage is None:
            raise RuntimeError(
                f"block {data.get('cf_block')} needs a decode stage: the rollout "
                "worker returned a handoff, so its --stage-roles stop before the "
                "decoder, but this deployment has only one stage"
            )
        return await self._decode_block(
            decode_stage, decode_client, session_id, request_id, data, ref
        )

    async def _decode_block(
        self,
        decode_stage: str,
        decode_client: Any,
        session_id: str,
        request_id: str,
        data: dict,
        ref: dict,
    ) -> np.ndarray:
        """Send one handoff to the decode stage and read the frames back."""
        block = int(data.get("cf_block") or 0)
        stage_request = {
            "cf_request_id": request_id,
            "cf_session": session_id,
            "cf_block": block,
            "cf_ref": ref,
        }
        out: dict | None = None
        async for chunk in self._stage_chunks(
            decode_stage, decode_client, session_id, stage_request
        ):
            if chunk.get("error"):
                raise RuntimeError(f"decode stage, block {block}: {chunk['error']}")
            out = chunk
        if out is None:
            raise RuntimeError(f"decode stage returned nothing for block {block}")
        frame_ref = out.get("cf_ref")
        if not isinstance(frame_ref, dict) or frame_ref.get("kind") != blocks.KIND_FRAMES:
            raise RuntimeError(
                f"decode stage, block {block}: expected decoded frames, got {frame_ref!r}"
            )
        return await asyncio.to_thread(blocks.take_frames, frame_ref)

    async def _finish_decode(
        self,
        decode_stage: Optional[str],
        decode_client: Any,
        session_id: str,
        request_id: str,
        data: dict,
    ) -> None:
        """Forward the rollout's terminal marker so the decode stage can release.

        Carries ``cf_close`` through rather than releasing unconditionally: a scene
        that will be continued needs its conv cache to still be there when the next
        request arrives, and dropping it would make the next block's first frame a
        seam. Nothing is decoded here, so nothing is queued for the encoder.
        """
        if decode_client is None or decode_stage is None:
            return
        terminal = {
            "cf_request_id": request_id,
            "cf_session": session_id,
            "cf_block": int(data.get("cf_block") or 0),
            "cf_last": True,
            "cf_close": bool(data.get("cf_close")),
        }
        async for chunk in self._stage_chunks(
            decode_stage, decode_client, session_id, terminal
        ):
            if chunk.get("error"):
                # Not fatal to the video: every frame is already with the client.
                # Worth a line, because it means an instance may still be holding a
                # session it was asked to drop.
                logger.warning(
                    "Router: decode stage failed to close session %s: %s",
                    session_id,
                    chunk["error"],
                )

    # -- addressing ---------------------------------------------------------

    async def _stage_chunks(
        self, stage_name: str, client: Any, session_id: str, stage_request: dict
    ) -> AsyncGenerator[dict, None]:
        """Send one request to a session's pinned instance, yielding its responses.

        One response per block on the rollout stage, so the chunks are *not* merged
        the way a one-shot stage's are: merging with ``dict.update`` would keep only
        the last block, silently, since each chunk is individually well-formed.
        """
        instance = self._instance_for(stage_name, client, session_id)
        async for chunk in await client.direct(stage_request, instance):
            data = chunk.data()
            if isinstance(data, (str, bytes)):
                data = json.loads(data)
            yield data

    def _instance_for(self, stage_name: str, client: Any, session_id: str) -> int:
        """The instance this session's state lives on, choosing one if it has none.

        A live pin is honoured as long as the instance is still there. If it is
        gone -- a restarted worker -- the session's window went with it, and the
        next chunk will be refused by the engine (an unknown session at
        ``block_idx > 0``) rather than quietly restarting the shot on a new card.

        A new session prefers an instance with no session pinned to it, because each
        instance holds exactly one: putting a second session on a busy instance
        evicts the first. When every instance is busy the request still proceeds --
        refusing it would be worse for a demo -- but the log says what is about to
        happen, since the symptom otherwise appears in the *other* session.
        """
        instance_ids = list(client.instance_ids())
        if not instance_ids:
            raise RuntimeError(f"stage '{stage_name}' has no instances")

        pinned = self._pinned.get((stage_name, session_id))
        if pinned is not None and pinned in instance_ids:
            return pinned

        busy = {
            iid
            for (stage, _sid), iid in self._pinned.items()
            if stage == stage_name
        }
        idle = [iid for iid in instance_ids if iid not in busy]
        if idle:
            chosen = idle[0]
        else:
            # crc32 rather than hash(): ``str.__hash__`` is salted per process, so
            # two routers would disagree about where the same session belongs.
            chosen = instance_ids[
                zlib.crc32(session_id.encode()) % len(instance_ids)
            ]
            logger.warning(
                "Router: every '%s' instance already holds a session; session %s "
                "will evict the one on instance %s. Each SGLang instance holds one "
                "realtime session, so concurrent streams need more stage workers.",
                stage_name,
                session_id,
                chosen,
            )
        self._pinned[(stage_name, session_id)] = chosen
        logger.info(
            "Router: session %s pinned to '%s' instance %s", session_id, stage_name, chosen
        )
        return chosen

    def _unpin(self, session_id: str) -> None:
        """Forget a closed session's pins so its instances count as idle again."""
        for key in [k for k in self._pinned if k[1] == session_id]:
            del self._pinned[key]


async def init_omni_stage_router(
    runtime: DistributedRuntime,
    config: OmniSglangConfig,
    shutdown_endpoints: list,
) -> None:
    """Bring up the router: discover the stages, register the model, serve."""
    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )
    shutdown_endpoints[:] = [generate_endpoint]

    router = OmniSglangStageRouter(config)

    # Wait for every stage before registering the model. Registration is what makes
    # the frontend advertise it, and a model that is advertised before its stages
    # exist answers the first request with a missing-client error.
    for stage_name in router.stage_names:
        client = await runtime.endpoint(
            f"{config.namespace}.{stage_name}.{config.endpoint}"
        ).client()
        await client.wait_for_instances()
        router.set_stage_client(stage_name, client)

    await register_model(
        ModelInput.Text,
        ModelType.Videos,
        generate_endpoint,
        config.model,
        config.served_model_name,
        # The router is the user-visible front of an internal pipeline; the stage
        # workers are private. From the frontend's topology view it serves
        # end-to-end with no peer dependencies.
        worker_type=WorkerType.Aggregated,
        needs=[],
    )
    logger.info(
        "Router: registered '%s' at '%s.%s.%s' over stages %s",
        config.served_model_name or config.model,
        config.namespace,
        config.component,
        config.endpoint,
        router.stage_names,
    )

    try:
        await generate_endpoint.serve_endpoint(
            router.generate,
            graceful_shutdown=True,
            metrics_labels=[],
        )
    except Exception as e:
        logger.error("Router: endpoint failed: %s", e)
        raise
