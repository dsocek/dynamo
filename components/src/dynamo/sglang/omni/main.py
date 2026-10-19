# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for ``python -m dynamo.sglang.omni``.

One command serves every role in the deployment; ``--stage-id`` or ``--omni-router``
picks which. That is deliberate: the stage workers and the router have to agree on
the stage names and the block protocol, and one module cannot disagree with itself.
"""

from __future__ import annotations

import asyncio
import logging
import os

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import fetch_model
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang.omni.args import parse_omni_args
from dynamo.sglang.omni.stage_router import init_omni_stage_router
from dynamo.sglang.omni.stage_worker import init_omni_stage

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


async def worker() -> None:
    config = parse_omni_args()
    dump_config(config.dump_config_to, config)

    # The router never loads weights, but it does need the model's identity to
    # register it -- and a local path that does not exist yet has to be fetched by
    # somebody. The stage workers would each fetch it anyway; doing it here means a
    # cold cache costs one download rather than one per process.
    if config.model and not os.path.exists(config.model):
        await fetch_model(config.model)

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )
    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    if config.stage_id is not None:
        await init_omni_stage(runtime, config, shutdown_endpoints, shutdown_event)
        logger.debug("Stage %d exited", config.stage_id)
    else:
        await init_omni_stage_router(runtime, config, shutdown_endpoints)
        logger.debug("Router exited")


def main() -> None:
    uvloop.run(worker())
