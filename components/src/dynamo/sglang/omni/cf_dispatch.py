# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session-scoped Causal-Forcing streaming: SGLang-side dispatch.

:mod:`dynamo.common.omni.cf_session` decides *what* a request asked for. This
decides how that becomes work for an SGLang instance, and the answer is much
plainer than on the vLLM-Omni path: there is no session registry to reach and no
control RPC to route. SGLang already has a realtime-session framework --
``RealtimeSessionCache`` binds per-chunk ``Req``s by ``req.realtime_session_id``
and everything downstream keys "start of a shot" off ``batch.block_idx == 0`` --
so a scene is driven by *sending chunks*, one ``Req`` at a time, over the ordinary
scheduler protocol.

That makes ``block_idx`` the whole protocol, and it is scene-relative exactly as
SGLang's own realtime adapter makes it (``chunk.index - scene_start_chunk_index``):

* ``block_idx == 0`` **is** the cut. It resets the DiT's attention window and the
  VAE's conv cache, and the four-step first-block schedule keys off it. A cut is
  therefore not a separate control message -- it is where the numbering restarts.
* ``block_idx > 0`` continues the shot, attending to what the previous chunks left
  in the window.

So the only state Dynamo has to keep is a counter per session, which is what
:class:`CFSessionBook` is. It cannot be authoritative about whether the *engine*
still holds the session -- that cache is inside the scheduler process, holds one
session at a time, and has no query message -- and it deliberately does not try
to be. A ``continue`` whose session this book has never seen is refused here; a
``continue`` whose session the engine has since dropped is refused by the engine,
which raises on ``block_idx > 0`` for an unknown session rather than starting a
fresh window. Either way the client is told, and neither way does a shot silently
restart from noise -- which is the failure worth designing against, because it
looks like a rendering choice rather than an error.
"""

from __future__ import annotations

import logging
import pickle
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynamo.common.omni.cf_session import CFSceneRequest
from dynamo.common.utils.video_utils import parse_size

logger = logging.getLogger(__name__)

# The distilled model's own geometry, not generic defaults. 81 pixel frames is 21
# latent frames, which is exactly Causal-Forcing's attention window at 480x832;
# 16 fps and one denoising step per block are what it was distilled for. The
# first block's four-step schedule is applied by the denoising stage from the
# pipeline config, so it is not set here.
CF_DEFAULT_NUM_FRAMES = 81
CF_DEFAULT_FPS = 16
CF_DEFAULT_NUM_INFERENCE_STEPS = 1
CF_DEFAULT_SIZE = "832x480"

# Wan's VAE compresses time 4:1 with the first latent frame standing alone, so a
# clip of ``1 + 4k`` pixel frames is ``1 + k`` latent frames. Chunk 0 therefore
# emits one pixel frame and every later chunk emits four.
_VAE_TEMPORAL_RATIO = 4

# Upper bound for a drawn seed. SGLang validates ``seed`` as a non-negative int and
# hands it to ``torch.Generator.manual_seed``, which takes a 64-bit value; 2**32 is
# the range the vLLM-Omni path draws from, kept the same so a seed logged by one
# backend can be replayed on the other.
_SEED_RANGE = 2**32


class CFDispatchError(RuntimeError):
    """Raised when a chunk could not be delivered, or when a scene cannot be run.

    Separate from ``CFSessionRequestError`` (a malformed request, the client's
    fault) because this is a scheduling or delivery failure, which is ours.
    """


def latent_frames(num_frames: int) -> int:
    """Latent frames in a clip of ``num_frames`` pixel frames."""
    return (max(1, int(num_frames)) - 1) // _VAE_TEMPORAL_RATIO + 1


@dataclass(frozen=True)
class SceneGeometry:
    """Everything a chunk ``Req`` needs that is not the prompt or the block index.

    Constant across a scene by construction: these fields size the KV window and
    the latent grid, so changing one mid-shot would invalidate the very cache the
    session exists to keep.

    ``seed`` as read off the request is ``None`` when the client did not state one.
    It must be an int by the time a ``Req`` is built -- SGLang refuses ``None`` --
    and which int it is is a *session* decision, not a per-request one, so
    :meth:`CFSessionBook.seed_for_scene` resolves it and the worker substitutes the
    answer back in before any chunk is dispatched.
    """

    height: int
    width: int
    num_frames: int
    fps: int
    seed: int | None = None

    @property
    def latents(self) -> int:
        return latent_frames(self.num_frames)


def scene_geometry(request: dict, default_fps: int) -> SceneGeometry:
    """Read a scene's geometry off the frontend request.

    ``nvext`` wins over the top-level fields where both exist, matching every
    other diffusion handler in the tree: the top-level OpenAI fields are the
    portable subset and ``nvext`` is where a client says what it actually wants.
    """
    nvext = request.get("nvext")
    nvext = nvext if isinstance(nvext, dict) else {}
    width, height = parse_size(request.get("size") or CF_DEFAULT_SIZE)
    fps = int(nvext.get("fps") or default_fps or CF_DEFAULT_FPS)
    num_frames = nvext.get("num_frames")
    if not num_frames:
        seconds = request.get("seconds")
        num_frames = int(seconds) * fps if seconds else CF_DEFAULT_NUM_FRAMES
    seed = nvext.get("seed")
    return SceneGeometry(
        height=int(nvext.get("height") or height),
        width=int(nvext.get("width") or width),
        num_frames=int(num_frames),
        fps=fps,
        seed=int(seed) if seed is not None else None,
    )


def scene_chunk_count(
    scene: CFSceneRequest, geometry: SceneGeometry, chunk_size: int
) -> int:
    """How many chunks this scene is worth.

    ``cf_scene.latents`` is the client's way of saying "hold this shot for N
    latent frames", which is the unit the rollout actually advances in; when it is
    absent the scene runs for the clip length implied by the request. Rounded up,
    because a partial chunk is still a chunk -- the alternative silently drops the
    tail of a scene whose length is not a multiple of the chunk size.
    """
    latents = scene.latents if scene.latents is not None else geometry.latents
    size = max(1, int(chunk_size))
    return max(1, -(-int(latents) // size))


class CFSessionBook:
    """Per-session state that cannot live inside the engine.

    Two things, for the same reason: the engine is *told* them rather than deriving
    them. The first is the scene-relative ``block_idx`` the next chunk carries. The
    second is the noise seed, which has to be identical on every chunk of a session
    -- SGLang's realtime input stage keeps one ``torch.Generator`` per session and
    reuses it only while ``batch.seed`` is unchanged (see ``_can_reuse_generator``
    in ``stages/realtime/input_validation.py``). Send a fresh seed per chunk and the
    generator is rebuilt each block, so every block draws the *same* noise -- which
    does not fail, it renders a shot that judders in place.
    """

    def __init__(self) -> None:
        self._next: dict[str, int] = {}
        self._seeds: dict[str, int] = {}

    def known(self, session_id: str) -> bool:
        return session_id in self._next

    def seed_for_scene(
        self, session_id: str, requested: int | None, transition: str
    ) -> int:
        """The seed every chunk of this scene will carry.

        A stated seed wins and is remembered, so the scenes that follow a
        ``continue`` keep the generator the client asked for. Otherwise a ``cut``
        draws a new one -- a new shot should not reuse the previous shot's noise
        sequence, or two cuts on one prompt come out the same -- and a
        ``continue`` inherits whatever the shot it is continuing was started with.
        """
        if requested is not None:
            self._seeds[session_id] = int(requested)
            return int(requested)
        seed = self._seeds.get(session_id)
        if seed is None or transition == "cut":
            seed = random.randrange(_SEED_RANGE)
            self._seeds[session_id] = seed
        return seed

    def begin_scene(self, session_id: str, transition: str) -> None:
        """Position the counter for a scene that is about to start.

        A ``cut`` restarts the numbering, which *is* the cut: block 0 is what
        clears the attention window and the conv cache, and what selects the
        four-step first-block schedule. A ``continue`` keeps counting so the new
        scene attends to the shot already on screen -- and is refused outright if
        there is no such shot, rather than quietly becoming a cut.
        """
        if transition == "cut":
            self._next[session_id] = 0
            return
        if session_id not in self._next:
            raise CFDispatchError(
                f"session {session_id!r} has no shot in progress, so this scene "
                "cannot continue one: there is no attention window to attend to, "
                "and rendering anyway would silently restart from noise. Send a "
                'scene with transition "cut" to start a shot under this id.'
            )

    def take(self, session_id: str) -> int:
        """Return the next block index and advance the counter."""
        index = self._next.get(session_id)
        if index is None:
            raise CFDispatchError(
                f"session {session_id!r} was not started; call begin_scene first"
            )
        self._next[session_id] = index + 1
        return index

    def forget(self, session_id: str) -> None:
        self._next.pop(session_id, None)
        self._seeds.pop(session_id, None)


class CFChunkDispatcher:
    """Drives one SGLang instance a chunk at a time, whatever roles it holds.

    Role-agnostic by design. A rollout instance answers a chunk with
    ``OutputBatch.stage_handoff``; a decode instance answers the same shape of
    call with pixels; an aggregated instance holding all three roles answers with
    pixels too. So there is one ``forward`` here and the caller decides what to do
    with what comes back -- which is what lets the 2-stage split and the
    aggregated deploy share the stage worker unchanged.
    """

    def __init__(self, engine: Any, stage_id: int) -> None:
        self.engine = engine
        self.stage_id = stage_id
        self.book = CFSessionBook()

    @property
    def chunk_size(self) -> int:
        return self.engine.chunk_size

    def build_chunk_req(
        self,
        session_id: str,
        block_idx: int,
        prompt: str,
        geometry: SceneGeometry,
    ) -> Any:
        """Build the ``Req`` for one chunk of a session.

        ``num_inference_steps=1`` and an empty ``negative_prompt`` are the
        distilled model's contract, not a performance choice: it takes one step
        per block and was distilled without classifier-free guidance, so a larger
        step count or a real negative prompt would be doing work the weights do
        not expect. ``save_output=False`` keeps the scheduler from writing an mp4
        per chunk -- the frames are going to the router, not to disk.
        """
        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        sampling_params = SamplingParams(
            request_id=f"{session_id}_{block_idx}",
            prompt=prompt,
            negative_prompt="",
            height=geometry.height,
            width=geometry.width,
            num_frames=geometry.num_frames,
            num_inference_steps=CF_DEFAULT_NUM_INFERENCE_STEPS,
            fps=geometry.fps,
            seed=geometry.seed,
            save_output=False,
            realtime_chunk_size=self.chunk_size,
        )
        req = Req(sampling_params=sampling_params)
        req.realtime_session_id = session_id
        req.block_idx = block_idx
        return req

    async def rollout_chunk(
        self,
        session_id: str,
        block_idx: int,
        prompt: str,
        geometry: SceneGeometry,
    ) -> Any:
        """Run one chunk on this instance and return its ``OutputBatch``."""
        req = self.build_chunk_req(session_id, block_idx, prompt, geometry)
        # ``return_raw_frames`` is set on the way out too, so an aggregated
        # instance (which decodes inline) answers with frame bytes rather than
        # device tensors. A rollout-only instance stops before the decoder and
        # ignores it.
        req.return_raw_frames = True
        return await self.engine.forward([req])

    async def decode_chunk(self, handoff_payload: bytes) -> Any:
        """Decode one chunk from the rollout stage's handoff.

        ``StageHandoff.to_req`` rebuilds the ``Req`` the decoder needs, including
        the ``realtime_session_id`` and ``block_idx`` that put this chunk in its
        place in the shot -- so the conv cache continues rather than resetting,
        and block boundaries stay seamless. The live session object is
        deliberately *not* carried across the handoff, which is what keeps one
        instance's device-resident state out of another instance's request.
        """
        handoff = pickle.loads(handoff_payload)
        req = handoff.to_req()
        req.return_raw_frames = True
        return await self.engine.forward([req])

    async def release(self, session_id: str) -> None:
        """Release a session on this instance, tolerating a missing one.

        Best-effort: a session that was never opened, or that the engine has
        already evicted, is not an error -- the caller asked for it to be gone and
        it is. Failing here would turn a successfully delivered video into a
        failed request.
        """
        self.book.forget(session_id)
        try:
            released = await self.engine.release_session(session_id)
        except Exception as e:  # noqa: BLE001 -- close is best-effort
            logger.warning(
                "Stage %d: releasing session %s failed: %s", self.stage_id, session_id, e
            )
            return
        logger.info(
            "Stage %d: session %s released=%s", self.stage_id, session_id, released
        )


def frames_from_output(output: Any) -> np.ndarray:
    """Canonical ``(T, H, W, 3) uint8`` frames from an ``OutputBatch``.

    SGLang returns raw frames as flat RGB byte payloads plus a geometry dict,
    which keeps the scheduler hop free of tensor serialization. Reassembling them
    needs the stride, so a payload holding several frames splits cleanly -- chunk
    0 of a Causal-Forcing stream emits one pixel frame and later chunks emit four,
    so a fixed frames-per-payload assumption would be wrong for both.
    """
    metadata = output.raw_frame_metadata or {}
    batches = output.raw_frame_batches or []
    if not metadata or not batches:
        raise CFDispatchError(
            "the instance returned no raw frames; a decoder-less instance answers "
            "with stage_handoff instead, and this call expected pixels"
        )
    height = int(metadata["height"])
    width = int(metadata["width"])
    channels = int(metadata.get("channels", 3))
    stride = height * width * channels

    frames: list[np.ndarray] = []
    for batch in batches:
        for payload in batch:
            if len(payload) % stride:
                raise CFDispatchError(
                    f"raw frame payload is {len(payload)} bytes, not a multiple of "
                    f"the {stride}-byte frame stride for {width}x{height}x{channels}"
                )
            for offset in range(0, len(payload), stride):
                frames.append(
                    np.frombuffer(
                        payload[offset : offset + stride], dtype=np.uint8
                    ).reshape(height, width, channels)
                )
    if not frames:
        raise CFDispatchError("the instance returned an empty raw frame batch")
    return np.stack(frames, axis=0)


def handoff_payload(output: Any) -> bytes | None:
    """Pickle an ``OutputBatch``'s stage handoff, or None when it carries pixels.

    The one place the two roles are told apart, and it is a property of the
    *answer* rather than of configuration: an instance whose roles stop before
    the decoder sets ``stage_handoff``, and one that decodes sets frames. Reading
    it off the output means the stage worker does not have to be told which half
    of the pipeline it is, and the aggregated deploy needs no special case.
    """
    handoff = getattr(output, "stage_handoff", None)
    if handoff is None:
        return None
    return pickle.dumps(handoff, protocol=pickle.HIGHEST_PROTOCOL)
