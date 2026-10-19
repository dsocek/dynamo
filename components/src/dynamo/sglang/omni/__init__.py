# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-Omni Causal-Forcing video streaming for Dynamo.

The disaggregated text-to-video pipeline: two stage workers (rollout, then decode),
a stage router that pipelines them and packages the frames, and the block transport
between them.

Deliberately empty of re-exports. Importing this package must not import SGLang --
the router process is one of its consumers and never builds an engine, and
``dynamo.sglang.omni.args`` is imported by tests that have no GPU. Import the
submodule you need::

    from dynamo.sglang.omni.args import parse_omni_args
    from dynamo.sglang.omni.stage_worker import init_omni_stage
    from dynamo.sglang.omni.stage_router import init_omni_stage_router
"""
