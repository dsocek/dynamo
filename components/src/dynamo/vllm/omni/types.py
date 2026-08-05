# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for disaggregated omni stage workers and connectors.
"""

import dataclasses
import logging
from typing import Any, AsyncGenerator, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator


@runtime_checkable
class StageEngine(Protocol):
    """Any engine that can generate outputs for a single pipeline stage.

    Matches AsyncOmni — the only vllm_omni engine with a consistent async
    generator interface for both LLM and diffusion.
    """

    engine: Any  # AsyncOmniEngine — exposes output_processors for registration

    def generate(
        self,
        prompt: Any,
        request_id: str = "",
        *,
        sampling_params_list: Any = None,
    ) -> AsyncGenerator[Any, None]:
        ...

    def get_tokenizer(self) -> Any:
        """Return the tokenizer (may be async — callers should await)."""
        ...


class StageOutput(BaseModel):
    """Validated output dict from a stage worker.

    Unknown keys are silently dropped (extra="ignore") to prevent arbitrary
    stage output from accumulating across stages. Only protocol fields pass through.
    finished/error are consumed by the router and not forwarded to subsequent stages.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_dropped_keys(cls, values: Any) -> Any:
        if isinstance(values, dict):
            known = {
                "shm_meta",
                "original_prompt",
                "stage_connector_refs",
                "sampling_params_list",
                "finished",
                "error",
                # Per-chunk block-stream control signals (work-item b, §8): a
                # non-terminal chunk carries only these (no tensor — latents
                # travel via the connector under {request_id}_c{n}).
                "chunk_index",
                "is_last",
                "cf_session",
                "cf_block",
                "cf_last",
            }
            dropped = set(values.keys()) - known
            if dropped:
                logging.warning(
                    "StageOutput: dropping unexpected keys from stage response: %s",
                    sorted(dropped),
                )
        return values

    # TODO: shm_meta should be gone, its a WAR right now to send final output to the router via shm
    shm_meta: dict | None = None
    original_prompt: dict | None = None
    # stage_connector_refs maps stage_id (str key from JSON) → opaque connector metadata
    # returned by connector.put(). This metadata is an address ticket passed to
    # connector.get(metadata=...) by the next stage to locate and fetch the data.
    # The format is connector-specific and opaque to the router:
    #   SHM connector:     {"shm": {"name": "<block_name>", "size": N}, "size": N}
    #                   or {"inline_bytes": b"...", "size": N}  (small payloads)
    #   Mooncake (RDMA):   {"source_host": "...", "source_port": N, "data_size": N, ...}
    # Keys arrive as strings from JSON; workers normalize them to int via _int_keyed().
    stage_connector_refs: dict[str, Any] | None = None
    sampling_params_list: dict | None = None
    finished: bool | None = None
    error: str | None = None
    # Two independent streaming lanes share this envelope, and they are gated by
    # different fields on purpose -- see cf_streaming_stage() and the n_streamed
    # branch in OmniStageWorker.generate.
    #
    # Lane A, "single-shot streamed" (chunk_index/is_last): one ordinary request
    # whose stage streams its blocks out as they are produced. The stage itself
    # decides -- the engine emits non-terminal chunks because the YAML set
    # stream_dit_blocks -- so no client opt-in exists and no session outlives the
    # request. chunk_index is the 0-based block index, is_last marks the terminal
    # signal.
    chunk_index: int | None = None
    is_last: bool | None = None
    # Lane B, "session pipelined" (cf_session/cf_block/cf_last): many requests
    # against one long-lived stream, so every chunk has to reach the *same* VAE
    # worker -- the decoder's temporal feat_cache is what makes chunk boundaries
    # seam-free, and it lives in one process. cf_session names that stream so the
    # next stage can address its decode cursor; cf_block is the block ordinal,
    # which the next stage needs because decode order is the stream's order, not
    # arrival order.
    cf_session: str | None = None
    cf_block: int | None = None
    # Marks the last block of the stream. Needed because a pipelined stage emits
    # many chunks and ``finished`` is per-chunk: without this the consumer could
    # not tell "this block is done" from "the stream is done", and for video the
    # two are indistinguishable downstream -- both look like a shorter clip.
    # (Lane A's is_last is the same idea; the two lanes keep separate fields so
    # a consumer can never mistake one lane's terminal for the other's.)
    cf_last: bool | None = None

    def to_next_stage_request(self, request_id: str) -> dict:
        """Build the request dict for the next stage: only inter-stage protocol fields.

        shm_meta is intentionally excluded — it is final-stage → router only.
        """
        fields = self.model_dump(
            include={
                "original_prompt",
                "stage_connector_refs",
                "sampling_params_list",
                "cf_session",
                "cf_block",
                "cf_last",
            },
            exclude_none=True,
        )
        fields["request_id"] = request_id
        return fields


class StageRequest(BaseModel):
    """Validated request dict received by a stage worker from the router.

    extra="ignore" handles all three request shapes:
      Stage 0:   {request_id, engine_inputs, original_prompt, stage_connector_refs: {}}
      Stage N>0: {request_id, original_prompt, stage_connector_refs: {"0": ref0, ...}}
      Direct:    raw frontend request (no router, single-stage deployment)
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str | None = None
    engine_inputs: Any = None
    original_prompt: dict | None = None
    # stage_connector_refs: address tickets from previous stages (same format as
    # StageOutput.stage_connector_refs). Callers normalize string keys to int via _int_keyed().
    stage_connector_refs: dict[str, Any] | None = None
    sampling_params_list: dict | None = None
    # Causal-Forcing streaming: see StageOutput.cf_session. Present only on the
    # per-block requests a pipelined rollout produces; absent on one-shot
    # requests, which is what keeps the batch path byte-identical.
    cf_session: str | None = None
    cf_block: int | None = None
    cf_last: bool | None = None


def _int_keyed(d: dict | None) -> dict[int, Any]:
    """Normalize JSON-deserialized string keys back to int for stage_connector_refs."""
    if not d:
        return {}
    return {int(k): v for k, v in d.items()}


@dataclasses.dataclass
class OmniInterStageRequest:
    """Protocol message passed between stage workers via the router.

    The router passes this opaquely without inspecting stage_connector_refs.
    Workers accumulate connector refs as the pipeline progresses, allowing
    any stage to reconstruct stage_list for N-stage processor functions.

    JSON-serializable: original_prompt is a TypedDict (dict subclass) with
    no tensors. Tensors (token_ids, images) travel via the connector payload.
    """

    request_id: str

    # OmniPromptType | list | None — typed as Any to avoid importing vllm_omni at
    # module level. Set once by the router at pipeline start, never modified by workers.
    original_prompt: Any

    # Grows as the pipeline progresses: {} → {0: ref0} → {0: ref0, 1: ref1} → ...
    stage_connector_refs: dict[int, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "original_prompt": self.original_prompt,
            "stage_connector_refs": self.stage_connector_refs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OmniInterStageRequest":
        return cls(
            request_id=d["request_id"],
            original_prompt=d["original_prompt"],
            stage_connector_refs=_int_keyed(d.get("stage_connector_refs")),
        )
