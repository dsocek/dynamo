# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic pieces of the omni (generative-media) serving path.

What lives here is everything about streaming generated video that does not
depend on which engine produced the pixels: the Causal-Forcing session protocol
a client speaks (:mod:`cf_session`), the handoffs a pipelined stream is built
from (:mod:`cf_pipeline`), and the binary-CMAF wire format the frontend's
``/v1/videos/stream/binary/cmaf`` route expects (:mod:`cmaf_video`).

Two backends implement that path -- ``dynamo.vllm.omni`` and
``dynamo.sglang.omni`` -- and they differ only in how a stage worker reaches its
engine. Sharing these modules is not de-duplication for its own sake: the
session annotations and the CMAF tags are a *contract* with the browser client,
so two copies would be two chances for one backend to drift out of a protocol
the other still speaks.

Deliberately import-light: nothing is re-exported here, so a caller importing
``dynamo.common.omni.cf_session`` pays for that module alone and neither backend
pulls the other's dependencies in.
"""
