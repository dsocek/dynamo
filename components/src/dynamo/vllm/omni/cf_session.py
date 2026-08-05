# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-scoped Causal-Forcing streaming: request-side protocol.

Causal-Forcing rolls video out autoregressively behind a KV window, so a scene can
continue the previous one instead of starting from noise. That only works if the
window survives *between* requests, which means a client needs a way to say "this
prompt belongs to the shot I opened earlier."

The carrier is ``nvext.annotations``. It is typed ``Option<Vec<String>>`` in Rust
(``lib/llm/src/protocols/openai/videos/nvext.rs``) and reaches Python as a raw list of
strings, so it passes arbitrary payloads through with no frontend changes -- the same
channel the binary-CMAF route uses for its own opt-in flag. Unknown *fields* on
``nvext`` would instead be dropped silently at deserialization, since ``NvExt`` is a
typed struct; annotations are the one place a new key can travel end to end today.

Wire format, one annotation per directive::

    cf_session=<id>                       bind this request to a session
    cf_scene={"transition":"cut",...}     scene parameters as compact JSON
    cf_close                              close the session after this request

``cf_scene`` carries:

    transition  "cut" (new location, drops history) or "continue" (default)
    latents     scene length in latent frames; omitted -> pipeline default
    seed        per-scene noise seed (see below)

``seed`` reproduces a shot. For a ``cut`` that is exact: a cut zeroes the attention
history, so the same seed and prompt give the same pixels. A ``continue`` also attends
to the preceding scenes' KV window, so it reproduces exactly when the scenes before it
are unchanged -- true when replaying a storyboard, false once an earlier scene is
edited. The field is accepted either way; only the guarantee narrows.

This module only parses and validates. Dispatch onto the worker-side registry lives in
``stage_worker``, which reaches the pipeline via ``collective_rpc``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Annotation keys. Kept prefixed so they cannot collide with the frontend's own
# annotations (e.g. experimental_binary_cmaf) on the shared list.
CF_SESSION_PREFIX = "cf_session="
CF_SCENE_PREFIX = "cf_scene="
CF_CLOSE_ANNOTATION = "cf_close"

_VALID_TRANSITIONS = ("continue", "cut")

# A session id becomes a dict key on the worker and appears in logs, so bound it
# rather than accepting arbitrary client input.
MAX_SESSION_ID_LEN = 128


class CFSessionRequestError(ValueError):
    """Raised when session annotations are present but malformed.

    Deliberately not tolerant. A dropped or misread scene directive would not fail --
    it would render, quietly, with the wrong length or the wrong history, and that is
    far more expensive to notice than a rejected request.
    """


@dataclass(frozen=True)
class CFSceneRequest:
    """One scene's parameters, as parsed off the wire."""

    transition: str = "continue"
    latents: int | None = None
    seed: int | None = None

    def push_kwargs(self) -> dict[str, Any]:
        """Kwargs for the worker's ``session_push``, omitting unset fields so the
        pipeline's own defaults apply rather than ones invented here."""
        kwargs: dict[str, Any] = {"transition": self.transition}
        if self.latents is not None:
            kwargs["latents"] = self.latents
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs


@dataclass(frozen=True)
class CFSessionRequest:
    """A request's session intent: which session, what scene, and whether to close."""

    session_id: str
    scene: CFSceneRequest
    close: bool = False


def _annotations(nvext: Any) -> list[str]:
    """Read annotations off either a raw dict or a VideoNvExt.

    stage_router sees the raw request dict while omni_handler sees the Pydantic model,
    and this protocol has to be readable from both.
    """
    if nvext is None:
        return []
    raw = (
        nvext.get("annotations")
        if isinstance(nvext, dict)
        else getattr(nvext, "annotations", None)
    )
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        raise CFSessionRequestError(
            f"nvext.annotations must be a list of strings, got {type(raw).__name__}."
        )
    return [a for a in raw if isinstance(a, str)]


def has_cf_session(nvext: Any) -> bool:
    """True when the request carries a session binding.

    Cheap and total: a request with no session annotation is an ordinary one-shot
    generation and must keep working exactly as before.
    """
    try:
        return any(a.startswith(CF_SESSION_PREFIX) for a in _annotations(nvext))
    except CFSessionRequestError:
        return False


def _parse_scene(payload: str) -> CFSceneRequest:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as e:
        raise CFSessionRequestError(
            f"{CF_SCENE_PREFIX}<json> is not valid JSON: {e}"
        ) from e
    if not isinstance(raw, dict):
        raise CFSessionRequestError(
            f"{CF_SCENE_PREFIX}<json> must be a JSON object, got {type(raw).__name__}."
        )

    unknown = set(raw) - {"transition", "latents", "seed"}
    if unknown:
        # Refuse rather than ignore: a typo'd key would otherwise silently fall back
        # to the default and render the wrong shot.
        raise CFSessionRequestError(
            f"unknown cf_scene field(s) {sorted(unknown)}; allowed: transition, latents, seed."
        )

    transition = raw.get("transition", "continue")
    if transition not in _VALID_TRANSITIONS:
        raise CFSessionRequestError(
            f"cf_scene.transition must be one of {_VALID_TRANSITIONS}, got {transition!r}."
        )

    latents = _coerce_int(raw, "latents", minimum=1)
    seed = _coerce_int(raw, "seed")
    return CFSceneRequest(transition=transition, latents=latents, seed=seed)


def _coerce_int(raw: dict, key: str, *, minimum: int | None = None) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    # bool is an int subclass; accepting it would turn seed=true into seed=1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise CFSessionRequestError(
            f"cf_scene.{key} must be an integer, got {value!r}."
        )
    if minimum is not None and value < minimum:
        raise CFSessionRequestError(
            f"cf_scene.{key} must be >= {minimum}, got {value}."
        )
    return value


def parse_cf_session(nvext: Any) -> CFSessionRequest | None:
    """Parse session annotations, or return None when the request carries none.

    Raises :class:`CFSessionRequestError` when annotations are present but malformed.
    """
    annotations = _annotations(nvext)

    session_ids = [
        a[len(CF_SESSION_PREFIX) :]
        for a in annotations
        if a.startswith(CF_SESSION_PREFIX)
    ]
    scenes = [
        a[len(CF_SCENE_PREFIX) :] for a in annotations if a.startswith(CF_SCENE_PREFIX)
    ]
    close = CF_CLOSE_ANNOTATION in annotations

    if not session_ids:
        if scenes or close:
            raise CFSessionRequestError(
                f"{CF_SCENE_PREFIX!r}/{CF_CLOSE_ANNOTATION!r} require a {CF_SESSION_PREFIX!r}<id> annotation."
            )
        return None

    if len(session_ids) > 1:
        # Which session owns the KV window is not guessable, and picking wrong would
        # append this scene to somebody else's shot.
        raise CFSessionRequestError(
            f"multiple {CF_SESSION_PREFIX!r} annotations: {session_ids}."
        )
    if len(scenes) > 1:
        raise CFSessionRequestError(
            f"multiple {CF_SCENE_PREFIX!r} annotations; one scene per request."
        )

    session_id = session_ids[0].strip()
    if not session_id:
        raise CFSessionRequestError(f"{CF_SESSION_PREFIX!r} needs a non-empty id.")
    if len(session_id) > MAX_SESSION_ID_LEN:
        raise CFSessionRequestError(
            f"session id is {len(session_id)} chars, over the {MAX_SESSION_ID_LEN} limit."
        )

    scene = _parse_scene(scenes[0]) if scenes else CFSceneRequest()
    return CFSessionRequest(session_id=session_id, scene=scene, close=close)


def build_cf_annotations(
    session_id: str,
    *,
    transition: str | None = None,
    latents: int | None = None,
    seed: int | None = None,
    close: bool = False,
) -> list[str]:
    """Build the annotation list a client sends. Used by tests and example clients so
    the encoder and parser cannot drift apart."""
    annotations = [f"{CF_SESSION_PREFIX}{session_id}"]
    scene: dict[str, Any] = {}
    if transition is not None:
        scene["transition"] = transition
    if latents is not None:
        scene["latents"] = latents
    if seed is not None:
        scene["seed"] = seed
    if scene:
        annotations.append(
            f"{CF_SCENE_PREFIX}{json.dumps(scene, separators=(',', ':'), sort_keys=True)}"
        )
    if close:
        annotations.append(CF_CLOSE_ANNOTATION)
    return annotations
