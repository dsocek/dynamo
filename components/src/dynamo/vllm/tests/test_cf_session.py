# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CF session annotation protocol."""

import json
from typing import ClassVar

import pytest
from dynamo.vllm.omni.cf_session import (
    CF_CLOSE_ANNOTATION,
    CFSessionRequestError,
    build_cf_annotations,
    has_cf_session,
    parse_cf_session,
)

pytestmark = pytest.mark.unit


def _nvext(*annotations):
    return {"annotations": list(annotations)}


# -- the non-session path must be untouched ---------------------------------


@pytest.mark.parametrize(
    "nvext",
    [
        None,
        {},
        {"annotations": None},
        {"annotations": []},
        _nvext("experimental_binary_cmaf"),
    ],
)
def test_requests_without_session_annotations_are_passthrough(nvext):
    """An ordinary one-shot generation must keep working exactly as before."""
    assert has_cf_session(nvext) is False
    assert parse_cf_session(nvext) is None


def test_cmaf_annotation_coexists_with_a_session():
    """Both ride the same list; CMAF streaming and sessions must compose."""
    nvext = _nvext("experimental_binary_cmaf", "cf_session=abc")
    assert has_cf_session(nvext) is True
    assert parse_cf_session(nvext).session_id == "abc"


# -- parsing ----------------------------------------------------------------


def test_session_without_a_scene_uses_defaults():
    got = parse_cf_session(_nvext("cf_session=s1"))
    assert got.session_id == "s1"
    assert got.scene.transition == "continue"
    # Unset -> the pipeline's own defaults apply, not ones invented in the parser.
    assert got.scene.latents is None
    assert got.scene.seed is None
    assert got.close is False
    assert got.scene.push_kwargs() == {"transition": "continue"}


def test_full_scene_parses():
    got = parse_cf_session(
        _nvext("cf_session=s1", 'cf_scene={"transition":"cut","latents":21,"seed":7}')
    )
    assert got.scene.transition == "cut"
    assert got.scene.latents == 21
    assert got.scene.seed == 7
    assert got.scene.push_kwargs() == {"transition": "cut", "latents": 21, "seed": 7}


def test_close_annotation():
    assert parse_cf_session(_nvext("cf_session=s1", CF_CLOSE_ANNOTATION)).close is True


def test_seed_zero_is_honoured_not_treated_as_unset():
    """0 is a legitimate seed; a falsy check would silently drop it."""
    got = parse_cf_session(_nvext("cf_session=s1", 'cf_scene={"seed":0}'))
    assert got.scene.seed == 0
    assert got.scene.push_kwargs()["seed"] == 0


def test_negative_seed_is_allowed_but_latents_must_be_positive():
    assert (
        parse_cf_session(_nvext("cf_session=s1", 'cf_scene={"seed":-5}')).scene.seed
        == -5
    )
    with pytest.raises(CFSessionRequestError, match="latents"):
        parse_cf_session(_nvext("cf_session=s1", 'cf_scene={"latents":0}'))


def test_pydantic_style_nvext_also_works():
    """stage_router sees a raw dict, omni_handler sees a model; both must parse."""

    class FakeNvExt:
        annotations: ClassVar = ["cf_session=s1", 'cf_scene={"transition":"cut"}']

    got = parse_cf_session(FakeNvExt())
    assert got.session_id == "s1"
    assert got.scene.transition == "cut"


def test_non_string_annotations_are_ignored_not_crashed_on():
    got = parse_cf_session({"annotations": ["cf_session=s1", 42, None]})
    assert got.session_id == "s1"


# -- malformed input is rejected, never guessed -----------------------------


def test_scene_without_session_is_rejected():
    """A scene with no session would otherwise render as a fresh one-shot clip."""
    with pytest.raises(CFSessionRequestError, match="require a"):
        parse_cf_session(_nvext('cf_scene={"transition":"cut"}'))
    with pytest.raises(CFSessionRequestError, match="require a"):
        parse_cf_session(_nvext(CF_CLOSE_ANNOTATION))


def test_ambiguous_session_is_rejected():
    with pytest.raises(CFSessionRequestError, match="multiple"):
        parse_cf_session(_nvext("cf_session=a", "cf_session=b"))


def test_multiple_scenes_rejected():
    with pytest.raises(CFSessionRequestError, match="one scene per request"):
        parse_cf_session(
            _nvext("cf_session=s1", 'cf_scene={"latents":4}', 'cf_scene={"latents":8}')
        )


def test_empty_session_id_rejected():
    with pytest.raises(CFSessionRequestError, match="non-empty"):
        parse_cf_session(_nvext("cf_session=   "))


def test_overlong_session_id_rejected():
    with pytest.raises(CFSessionRequestError, match="limit"):
        parse_cf_session(_nvext("cf_session=" + "x" * 200))


@pytest.mark.parametrize(
    "payload,match",
    [
        ("not json", "not valid JSON"),
        ("[1,2,3]", "must be a JSON object"),
        ('{"transition":"dissolve"}', "transition must be"),
        ('{"latents":"21"}', "must be an integer"),
        ('{"seed":1.5}', "must be an integer"),
        ('{"seed":true}', "must be an integer"),
        ('{"latnets":21}', "unknown cf_scene field"),
    ],
)
def test_malformed_scene_payloads_rejected(payload, match):
    with pytest.raises(CFSessionRequestError, match=match):
        parse_cf_session(_nvext("cf_session=s1", f"cf_scene={payload}"))


def test_typo_in_field_name_is_rejected_not_defaulted():
    """The failure this guards against is silent: a typo'd key would fall back to the
    default and render the wrong shot with no error anywhere."""
    with pytest.raises(CFSessionRequestError, match="unknown cf_scene field"):
        parse_cf_session(_nvext("cf_session=s1", 'cf_scene={"seeed":7}'))


# -- encoder/parser round trip ---------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"transition": "cut"},
        {"latents": 21},
        {"seed": 0},
        {"transition": "cut", "latents": 16, "seed": 12345},
        {"transition": "continue", "latents": 4, "seed": -1, "close": True},
    ],
)
def test_build_then_parse_round_trips(kwargs):
    """Pins the encoder and parser together so the wire format cannot drift."""
    close = kwargs.pop("close", False)
    annotations = build_cf_annotations("sess-1", close=close, **kwargs)
    got = parse_cf_session({"annotations": annotations})
    assert got.session_id == "sess-1"
    assert got.close is close
    assert got.scene.transition == kwargs.get("transition", "continue")
    assert got.scene.latents == kwargs.get("latents")
    assert got.scene.seed == kwargs.get("seed")


def test_built_annotations_survive_a_json_hop():
    """Annotations cross Rust->Python as JSON; structured payloads must survive it."""
    annotations = build_cf_annotations("s1", transition="cut", latents=21, seed=7)
    wire = json.loads(json.dumps({"nvext": {"annotations": annotations}}))
    got = parse_cf_session(wire["nvext"])
    assert (got.scene.transition, got.scene.latents, got.scene.seed) == ("cut", 21, 7)
