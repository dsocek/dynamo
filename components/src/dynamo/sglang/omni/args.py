# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for ``python -m dynamo.sglang.omni``.

Two parsers, one command line. The Dynamo flags (namespace, endpoint, media
output, plus the omni role flags below) are parsed here; everything else is
handed to SGLang's own :func:`prepare_server_args`, which is the only thing that
knows how to build a ``multimodal_gen`` ``ServerArgs``. Both use
``parse_known_args``, so a flag either side does not recognise falls through
rather than aborting the launch -- the same arrangement
``dynamo.vllm.omni.args`` uses for ``OmniEngineArgs``.

What this backend does *not* need is a stage-configs YAML. vLLM-Omni composes a
pipeline from a list of stage configs, so its router reads the YAML to learn how
many stages there are and what each is called. SGLang composes the same split
from ``--stage-roles`` on each instance (a role set per process, resolved inside
the pipeline), so the only thing left for Dynamo to know is how many stage
workers there are and what to call their endpoints -- hence ``--num-stages`` and
``--stage-names`` instead of a config file. Both the workers and the router take
the same two flags so a name cannot drift between them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument

logger = logging.getLogger(__name__)

# Default per-stage endpoint component names. Prefixed so they cannot collide
# with the ``backend`` component the router itself serves, or with an ordinary
# ``dynamo.sglang`` worker sharing the namespace.
DEFAULT_STAGE_NAME_PREFIX = "cf_stage"


def default_stage_names(num_stages: int) -> list[str]:
    """Endpoint component name per stage, in pipeline order."""
    return [f"{DEFAULT_STAGE_NAME_PREFIX}{i}" for i in range(num_stages)]


class OmniSglangArgGroup(ArgGroup):
    """CLI arguments for the Dynamo SGLang-Omni Causal-Forcing pipeline."""

    name = "dynamo-sglang-omni"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group(
            "SGLang-Omni Options",
            "Roles and topology for the disaggregated Causal-Forcing video pipeline.",
        )

        add_argument(
            g,
            flag_name="--stage-id",
            env_var="DYN_OMNI_STAGE_ID",
            default=None,
            arg_type=int,
            help=(
                "Run this process as stage <id> of the pipeline: an internal Dynamo "
                "worker wrapping one SGLang instance. Stage 0 rolls blocks out "
                "(--stage-roles encoder,denoiser), stage 1 decodes them "
                "(--stage-roles decoder). Mutually exclusive with --omni-router."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--omni-router",
            env_var="DYN_OMNI_ROUTER",
            default=False,
            help=(
                "Run as the stage router: the model the frontend sees, which "
                "pipelines the stage workers and packages their frames. Mutually "
                "exclusive with --stage-id."
            ),
        )
        add_argument(
            g,
            flag_name="--num-stages",
            env_var="DYN_OMNI_NUM_STAGES",
            default=2,
            arg_type=int,
            help=(
                "Number of stage workers in the pipeline. 2 is the disaggregated "
                "DiT|VAE split; 1 is the aggregated deploy, where one worker holds "
                "all three roles and the router only overlaps the encoder."
            ),
        )
        add_argument(
            g,
            flag_name="--stage-names",
            env_var="DYN_OMNI_STAGE_NAMES",
            default=None,
            help=(
                "Comma-separated endpoint component name per stage, in pipeline "
                f"order. Defaults to {DEFAULT_STAGE_NAME_PREFIX}0,"
                f"{DEFAULT_STAGE_NAME_PREFIX}1,... Pass the same value to every "
                "stage worker and to the router; they address each other by it."
            ),
        )
        add_argument(
            g,
            flag_name="--default-video-fps",
            env_var="DYN_OMNI_DEFAULT_VIDEO_FPS",
            default=16,
            arg_type=int,
            help=(
                "Frames per second used when a request does not state one. 16 is "
                "what Causal-Forcing Wan2.1-T2V was distilled at."
            ),
        )
        add_negatable_bool_argument(
            g,
            flag_name="--sglang-remote",
            env_var="DYN_OMNI_SGLANG_REMOTE",
            default=False,
            help=(
                "Attach to an SGLang scheduler that is already running at "
                "--scheduler-port instead of launching one in this process. For "
                "attaching to a `sglang serve` started by hand, which is how a "
                "stage is debugged without the Dynamo plane in the way."
            ),
        )


class OmniSglangConfig(DynamoRuntimeConfig):
    """Resolved configuration for one SGLang-Omni process."""

    # The router is the user-visible model and serves ``{ns}.backend.generate``.
    # Stage workers ignore this and serve ``{ns}.{stage_name}.generate`` instead,
    # which is what keeps them off the frontend's model list.
    component: str = "backend"
    endpoint: Optional[str] = None

    model: str = ""
    served_model_name: Optional[str] = None

    # ``sglang.multimodal_gen.runtime.server_args.ServerArgs``. Untyped here so
    # importing this module does not import SGLang -- the router process needs
    # the config shape but never builds an engine.
    server_args: Any = None

    stage_id: Optional[int] = None
    omni_router: bool = False
    num_stages: int = 2
    stage_names: Optional[str] = None
    default_video_fps: int = 16
    sglang_remote: bool = False

    @property
    def resolved_stage_names(self) -> list[str]:
        """Endpoint component name per stage, in pipeline order."""
        if self.stage_names:
            names = [n.strip() for n in self.stage_names.split(",") if n.strip()]
            if names:
                return names
        return default_stage_names(self.num_stages)

    @property
    def stage_name(self) -> str:
        """This process's own stage name. Only meaningful when ``stage_id`` is set."""
        if self.stage_id is None:
            raise ValueError("stage_name is only defined for a stage worker")
        return self.resolved_stage_names[self.stage_id]

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)

        if self.stage_id is None and not self.omni_router:
            raise ValueError(
                "one of --stage-id or --omni-router is required; this entrypoint "
                "only serves the disaggregated Causal-Forcing pipeline. For an "
                "ordinary one-shot text-to-video worker use "
                "`python -m dynamo.sglang --video-generation-worker`."
            )
        if self.stage_id is not None and self.omni_router:
            raise ValueError("--stage-id and --omni-router are mutually exclusive")

        if self.num_stages not in (1, 2):
            # Not an arbitrary cap: the pipeline this serves is a DiT|VAE split.
            # A third stage would have no role to hold, and the router's block
            # bookkeeping assumes the decode hop is the last one.
            raise ValueError(
                f"--num-stages must be 1 (aggregated) or 2 (DiT|VAE), got {self.num_stages}"
            )
        names = self.resolved_stage_names
        if len(names) != self.num_stages:
            raise ValueError(
                f"--stage-names has {len(names)} entries but --num-stages is "
                f"{self.num_stages}; they must agree, and every process in the "
                "deployment must be given the same pair."
            )
        if len(set(names)) != len(names):
            raise ValueError(f"--stage-names must be unique, got {names}")

        if self.stage_id is not None and not (0 <= self.stage_id < self.num_stages):
            raise ValueError(
                f"--stage-id {self.stage_id} is out of range for --num-stages "
                f"{self.num_stages}"
            )

        if self.default_video_fps <= 0:
            raise ValueError("--default-video-fps must be > 0")

        if self.server_args is not None and not getattr(
            self.server_args, "model_path", None
        ):
            raise ValueError("--model-path is required")


def _prepare_server_args(prepare_server_args: Any, argv: list[str]) -> Any:
    """Build a ``ServerArgs`` from ``argv``, working around how SGLang reads it.

    ``ServerArgs.get_provided_args`` decides which flags the user actually set by
    scanning ``sys.argv`` -- not the argv it was handed. That is invisible in a real
    launch, where the two agree, but it silently drops *every* flag when the argv is
    passed programmatically: ``--model-path`` included, and the failure surfaces
    much further down as "model_path is required in kwargs".

    So ``sys.argv`` is made to agree for the duration of the call. Restored
    afterwards because other code reads it too, and a process whose argv is a lie
    about how it was started is a bad thing to leave behind.
    """
    original = sys.argv
    sys.argv = [original[0] if original else "dynamo.sglang.omni", *argv]
    try:
        return prepare_server_args(argv)
    finally:
        sys.argv = original


def parse_omni_args(argv: list[str] | None = None) -> OmniSglangConfig:
    """Parse the command line into an :class:`OmniSglangConfig`.

    ``argv`` defaults to ``sys.argv[1:]``; it is a parameter so tests can drive
    the whole parse without touching process state.
    """
    from sglang.multimodal_gen.runtime.server_args import prepare_server_args

    parser = argparse.ArgumentParser(
        description="Dynamo SGLang-Omni Causal-Forcing worker",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    OmniSglangArgGroup().add_arguments(parser)

    args, unknown = parser.parse_known_args(argv)
    config = OmniSglangConfig.from_cli_args(args)

    if config.endpoint is None:
        config.endpoint = "generate"

    # Everything the Dynamo parser did not claim is SGLang's. ``prepare_server_args``
    # tolerates the leftovers of ours it sees for the same reason.
    config.server_args = _prepare_server_args(prepare_server_args, unknown)
    config.model = getattr(config.server_args, "model_path", "") or ""
    # SGLang owns ``--served-model-name``, so read the user-facing name back off
    # its ServerArgs rather than adding a second flag that could disagree.
    config.served_model_name = (
        getattr(config.server_args, "served_model_name", None) or config.model
    )

    config.validate()
    return config
