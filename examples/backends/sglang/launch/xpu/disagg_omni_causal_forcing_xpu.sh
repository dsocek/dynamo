#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 2-stage DISAGGREGATED Causal-Forcing (Wan2.1) streaming text-to-VIDEO on Intel XPU,
# served by the SGLang-Omni backend.
#
#   Stage 0: text encode + framewise rollout  (--stage-roles encoder,denoiser) -> latents
#   Stage 1: VAE decode                       (--stage-roles decoder)          -> frames
#   Router : pipelines the two stages and packages the video (the model the frontend sees)
#   Frontend: OpenAI-compatible HTTP ingress (POST /v1/videos)
#
# XPU cards: 2 (one per stage). The DiT stage and the VAE stage each get a single
# card of their own via ZE_AFFINITY_MASK, so neither can touch the other's -- and
# the split is the point: stage 1 decodes block N while stage 0 rolls out block N+1.
#
# Requires the SGLang-Omni tree with the Causal-Forcing pipeline on PYTHONPATH:
#   git clone https://github.com/dsocek/sglang.git sglang-omni
#   cd sglang-omni && git checkout cf_wan_pipe_sglang
#
# Prereqs: etcd + nats-server running. NATS needs a raised payload ceiling because
# a live CMAF segment travels the request plane as base64:
#   etcd --listen-client-urls http://0.0.0.0:${PORT_ETCD:-2370} \
#        --advertise-client-urls http://0.0.0.0:${PORT_ETCD:-2370} \
#        --listen-peer-urls http://localhost:2371 --data-dir /tmp/etcd &
#   printf 'max_payload: 64MB\n' > /tmp/nats.conf
#   nats-server -js --port ${PORT_NATS:-4223} -c /tmp/nats.conf &
#
# Video is encoded by the router, which picks VA-API on an XPU host. A container
# without an Intel VA-API driver installed (no /usr/lib/*/dri/*_drv_video.so) fails
# every request with "Failed to initialise VAAPI connection" from ffmpeg; force
# software encoding there with DYN_VIDEO_HW_ACCEL=cpu.
#
# Usage (every var below is overridable):
#   SGLANG_OMNI_REPO=/path/to/sglang-omni \
#   MODEL=/path/to/causal-forcing-1step-sglang \
#   DIT_CARD=0 VAE_CARD=2 DYN_HTTP_PORT=8880 \
#   bash disagg_omni_causal_forcing_xpu.sh
#
# The browser demo (live CMAF, storyboard scenes) runs against this stack unchanged:
#   python3 examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \
#       --frontend-port ${DYN_HTTP_PORT:-8880} --proxy-port 8080
#   then open http://localhost:8080/

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# launch_utils only: the diffusion server args have no VRAM budget knob, so there
# is nothing for gpu_utils.sh's build_sglang_gpu_mem_args (--max-total-tokens, an
# LLM flag) to set -- same as the other sglang diffusion launch scripts.
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# ---- Model and repo ----
MODEL="${MODEL:-./causal-forcing-1step-sglang}"
SGLANG_OMNI_REPO="${SGLANG_OMNI_REPO:-../sglang-omni}"
export PYTHONPATH="${SGLANG_OMNI_REPO}/python${PYTHONPATH:+:$PYTHONPATH}"

# ---- Runtime plane (match the etcd/nats already running on this box) ----
IP_LOCAL="${IP_LOCAL:-$(hostname -I | awk '{print $1}')}"
PORT_NATS="${PORT_NATS:-4223}"
PORT_ETCD="${PORT_ETCD:-2370}"
HTTP_PORT="${DYN_HTTP_PORT:-8880}"

export NATS_SERVER="nats://${IP_LOCAL}:${PORT_NATS}"
export ETCD_ENDPOINTS="http://${IP_LOCAL}:${PORT_ETCD}"
export DYN_REQUEST_PLANE=tcp
export ETCD_LEASE_TTL=600
# Fresh namespace per run, so stale discovery entries from a previous stack cannot
# route a request straight into a stage worker that is no longer there.
export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo-sglang-cf-$(date +%s)}"
export HF_HOME="${HF_HOME:-/mnt/bigtmp/hf}"

MEDIA_URL="${MEDIA_URL:-file:///tmp/dynamo_media_cf}"

# ---- Cards: one per stage ----
# Masked per process rather than shared, so each SGLang instance sees exactly one
# device and indexes it as xpu:0. Override with the free cards on your box.
DIT_CARD="${DIT_CARD:-0}"
# VAE_CARD may be a comma-separated list to spatially shard the decode across
# cards (see the parallel-decode block below); the DiT stage stays on one card
# because Ulysses SP over PCIe costs more than it saves on a 1-step block.
VAE_CARD="${VAE_CARD:-2}"
VAE_NUM_GPUS="${VAE_NUM_GPUS:-$(awk -F, '{print NF}' <<<"$VAE_CARD")}"

# ---- Ports ----
# Dynamo system (metrics/health) ports, one per process.
SYSTEM_PORT_DIT="${DYN_SYSTEM_PORT_DIT:-8091}"
SYSTEM_PORT_VAE="${DYN_SYSTEM_PORT_VAE:-8092}"
SYSTEM_PORT_ROUTER="${DYN_SYSTEM_PORT_ROUTER:-8093}"
# Each stage worker launches its own SGLang scheduler, so the scheduler and
# torch-distributed master ports must differ between the two instances. SGLang
# steps a busy scheduler port forward on its own (--strict-ports is off), but the
# master port does not, and a collision there fails the launch with EADDRINUSE.
SCHED_PORT_DIT="${SCHED_PORT_DIT:-5591}"
SCHED_PORT_VAE="${SCHED_PORT_VAE:-5592}"
MASTER_PORT_DIT="${MASTER_PORT_DIT:-30105}"
MASTER_PORT_VAE="${MASTER_PORT_VAE:-30205}"

# ---- Engine knobs that must not be left to SGLang's defaults ----
# ServerArgs._adjust_offload turns dit_cpu_offload on whenever it is left unset
# (it is a `bool | None` that defaults to None, and every video branch of that
# method sets it True). That default is meant for one-shot clips, where paying a
# host->device weight copy once per generation buys headroom. A Causal-Forcing
# stream calls the DiT once per *block* -- 21 times for an 81-frame shot, and
# forever in a live session -- so the copy lands on the critical path of every
# block and the stream stops keeping up with playback. Pinned off, which is what
# the hand-launched instances this pipeline was measured on actually ran with.
# Passed to both stages: the decode stage never loads the DiT, so there it is a
# no-op, and keeping the two invocations symmetric means a later change to the
# role split cannot silently reintroduce the offload.
DIT_CPU_OFFLOAD="${DIT_CPU_OFFLOAD:-false}"

# Attention backend for the DiT. Left unset, ServerArgs picks `fa` on XPU --
# _set_default_attention_backend only has a considered default for ROCm, and
# nothing measured stands behind `fa` here. At the shape a Causal-Forcing block
# actually runs (832x480: 1560 query tokens attending over a full 21-frame KV
# window = 32760 keys, 12 heads x 128, 30 layers) `torch_sdpa` is much faster on
# this card: 3.72 ms/layer / 84 TFLOPS against 7.49 ms / 42 TFLOPS, on a card whose
# GEMM peak is 155 TFLOPS.
#
# End to end at 832x480, 241 frames, 1 DiT + 3 VAE: 8.55 -> 9.78 fps. The reason it
# is worth more than the per-layer delta suggests is that it drops the DiT's serial
# work below the VAE's, which is what lets extra decode cards matter at all --
# on `fa` the DiT capped every config at ~8.5 fps and the 3rd and 4th VAE card
# bought literally nothing.
#
# Passed to stage 0 only, unlike --dit-cpu-offload above: the decode stage has its
# own attention (the VAE mid-block) at a completely different shape, which nothing
# here measured, so it keeps the tree's default.
DIT_ATTENTION_BACKEND="${DIT_ATTENTION_BACKEND:-torch_sdpa}"

# The Wan VAE can split the latent by height across the decode ranks, exchanging
# halo rows between the spatial-parallel convs. Left on `auto`, SGLang refuses to:
# the heuristic wants >= auto_parallel_decode_min_latent_elements_per_rank (4096)
# latent elements per rank, and a Causal-Forcing block is a *single* latent frame
# -- 1*60*104 = 6240 at 832x480, so 3120 per rank on two cards. The threshold is
# calibrated for one-shot decode of a whole clip; on this path it under-counts by
# the frame count and silently turns the shard off, leaving every rank to decode
# the full height redundantly *and* pay the group sync. Asked for explicitly it
# scales cleanly: 0.548s -> 0.419s -> 0.346s -> 0.299s per block on 1, 2, 3 and 4
# cards (832x480, 241 frames, sc09dell08-b70 / Arc Pro B70).
#
# Note where that stops paying. Decode cards only help while the decode is the
# slower half; once it drops below the rollout (~0.439s per block on one card, with
# the attention backend set above) the DiT is the wall and further shards are free
# capacity, not speed. That crossover sits right at the third card:
#
#   VAE cards  decode/block  total GPUs  fps
#   1          0.548s        2           6.89
#   2          0.419s        3           8.72
#   3          0.346s        4           9.78   <- decode now below the 0.439s DiT
#   4          0.299s        5           9.83   <- +0.5%, i.e. nothing
#
# So 4 cards is the knee for a single stream. Spend a 5th card on a second stream
# rather than on a fourth shard.
VAE_PARALLEL_DECODE_MODE="${VAE_PARALLEL_DECODE_MODE:-spatial_shard}"

# oneCCL picks its `topo` algorithm by default. On these cards -- which it reports
# as PCIe-connected, no XeLink -- topo hangs at non-power-of-two world sizes: a
# bare xccl script at world_size 3 or 5 clears `barrier` and then never returns
# from `all_reduce`, and its `all_gather` hangs too, while world_size 2 and 4 pass
# every collective (including across the host's two PCI root complexes). Sharding
# the decode over three or five cards therefore deadlocks in the first mid-block
# halo exchange.
#
# Scope the override to non-power-of-two world sizes and nothing else: `ring` is
# correct everywhere but *much* slower where topo works -- for a 4.79MB all_reduce,
# 2.005ms against 0.171ms at world_size 2 and 3.034ms against 0.266ms at world_size
# 4, i.e. 2.4 GB/s against 27-28. Applying it unconditionally silently taxes every
# collective in the deployment by an order of magnitude.
if (( VAE_NUM_GPUS & (VAE_NUM_GPUS - 1) )); then
    export CCL_ALLREDUCE="${CCL_ALLREDUCE:-ring}"
    export CCL_ALLGATHER="${CCL_ALLGATHER:-ring}"
    export CCL_ALLGATHERV="${CCL_ALLGATHERV:-ring}"
fi

print_launch_banner --no-curl \
    "Launching Disaggregated Causal-Forcing T2V (SGLang-Omni, Intel XPU)" \
    "$MODEL" "$HTTP_PORT" \
    "Namespace:   $DYN_NAMESPACE" \
    "Runtime:     NATS=$NATS_SERVER ETCD=$ETCD_ENDPOINTS" \
    "Cards:       DiT stage -> XPU $DIT_CARD, VAE stage -> XPU $VAE_CARD ($VAE_NUM_GPUS shard(s), $VAE_PARALLEL_DECODE_MODE)" \
    "Media:       $MEDIA_URL"

# ---- Stage 0: text encode + rollout -> latents ----
echo "Starting stage 0 (encoder,denoiser) on XPU ${DIT_CARD}..."
ZE_AFFINITY_MASK="$DIT_CARD" \
DYN_SYSTEM_PORT="$SYSTEM_PORT_DIT" \
    python3 -m dynamo.sglang.omni \
    --stage-id 0 \
    --num-stages 2 \
    --model-path "$MODEL" \
    --stage-roles encoder,denoiser \
    --dit-cpu-offload "$DIT_CPU_OFFLOAD" \
    --attention-backend "$DIT_ATTENTION_BACKEND" \
    --scheduler-port "$SCHED_PORT_DIT" \
    --master-port "$MASTER_PORT_DIT" &

# ---- Stage 1: VAE decode -> frames ----
echo "Starting stage 1 (decoder) on XPU ${VAE_CARD}..."
ZE_AFFINITY_MASK="$VAE_CARD" \
DYN_SYSTEM_PORT="$SYSTEM_PORT_VAE" \
    python3 -m dynamo.sglang.omni \
    --stage-id 1 \
    --num-stages 2 \
    --model-path "$MODEL" \
    --stage-roles decoder \
    --dit-cpu-offload "$DIT_CPU_OFFLOAD" \
    --num-gpus "$VAE_NUM_GPUS" \
    --vae-config.parallel-decode-mode "$VAE_PARALLEL_DECODE_MODE" \
    --scheduler-port "$SCHED_PORT_VAE" \
    --master-port "$MASTER_PORT_VAE" &

# ---- Router: the model the frontend serves ----
# No engine and no card: it addresses the two stage workers by name. It waits for
# both stage endpoints to appear before registering the model, so starting it now
# rather than after a sleep is safe -- and the model shows up on /v1/models only
# once the pipeline behind it can actually serve a request.
echo "Starting router..."
DYN_SYSTEM_PORT="$SYSTEM_PORT_ROUTER" \
    python3 -m dynamo.sglang.omni \
    --omni-router \
    --num-stages 2 \
    --model-path "$MODEL" \
    --output-modalities video \
    --media-output-fs-url "$MEDIA_URL" &

# ---- Frontend: OpenAI-compatible HTTP ingress ----
echo "Starting frontend on :${HTTP_PORT}..."
python3 -m dynamo.frontend --http-port "$HTTP_PORT" &

# Readiness is "the model is registered", not "the port answers": the frontend
# serves /health long before the two stages have loaded their weights, so
# wait_for_ready alone would report a stack that still 404s every request.
echo "Waiting for the pipeline to register its model (weights load on two cards)..."
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
_started=$SECONDS
while (( SECONDS - _started < READY_TIMEOUT )); do
    if curl -sf --max-time 2 "http://localhost:${HTTP_PORT}/v1/models" | grep -q '"id"'; then
        echo "Pipeline ready after $(( SECONDS - _started ))s."
        break
    fi
    sleep 3
done

print_curl_footer <<CURL
  # One-shot clip (81 frames at 16 fps, one denoising step per block):
  curl -s http://localhost:${HTTP_PORT}/v1/videos \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "prompt": "${EXAMPLE_PROMPT_VISUAL}",
      "size": "832x480",
      "response_format": "url",
      "nvext": { "fps": 16, "num_frames": 81 }
    }' | jq

  # Live CMAF in the browser (storyboard scenes over one session):
  python3 examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \\
      --frontend-port ${HTTP_PORT} --proxy-port 8080
  # then open http://localhost:8080/
CURL

wait_any_exit
