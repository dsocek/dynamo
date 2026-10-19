#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-stage AGGREGATED Causal-Forcing (Wan2.1) streaming text-to-VIDEO on Intel XPU,
# served by the SGLang-Omni backend.
#
#   Worker  : text encode + framewise rollout + VAE decode in ONE process
#             (--num-stages 1 --stage-roles encoder,denoiser,decoder)
#   Router  : packages the video (the model the frontend serves)
#   Frontend: OpenAI-compatible HTTP ingress (POST /v1/videos)
#
# This is the baseline the disaggregated script is measured against, and the right
# choice when only one card is free. Because one process does both the rollout and
# the decode, the two cannot overlap: block N is decoded before block N+1 starts
# rolling out, so the per-block cost is the *sum* of the two. Disaggregation exists
# to overlap them -- see disagg_omni_causal_forcing_xpu.sh.
#
# Measured on sc09dell08-b70 (Intel Arc Pro B70), 832x480, 241 frames, median of 2,
# every row launched with the flags this script sets, one vintage:
#
#   config                              GPUs  per-block  fps   streams 8 fps?
#   aggregated, 1 card                  1     0.884s     4.46  no
#   aggregated, TP=2                    2     0.716s     5.33  no
#   aggregated, TP=4                    4     0.581s     6.71  no
#   disaggregated 1 DiT + 1 VAE         2     0.547s     6.89  no
#   disaggregated 1 DiT + 3 VAE         4     0.402s     9.78  yes
#
# Note the last two rows: two cards spent on *disaggregation* beat four spent on
# aggregated tensor parallelism (6.89 against 6.71), because splitting the stages
# overlaps the rollout with the decode while TP only makes each of them narrower.
# Aggregated is the right choice when one card is free, not a way to scale.
#
# NUM_GPUS is limited to 1, 2 or 4 by the model, not by the host: at world size 3
# or 5 the tensor-parallel split aborts during load in ensure_divisibility, because
# neither the T5 vocabulary (256384) nor the DiT FFN width (8960) -- nor, at 5, the
# DiT hidden size (1536) -- divides evenly. This script rejects those up front
# rather than letting a launch die halfway through loading weights.
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
#   CARDS=0 DYN_HTTP_PORT=8880 \
#   bash agg_omni_causal_forcing_xpu.sh
#
#   CARDS=0,1     # tensor-parallel over two cards
#   CARDS=0,1,3,5 # tensor-parallel over four cards
#
# The browser demo (live CMAF, storyboard scenes) runs against this stack unchanged:
#   python3 examples/custom_backend/cmaf_binary_video_streaming/run_proxy.py \
#       --frontend-port ${DYN_HTTP_PORT:-8880} --proxy-port 8080
#   then open http://localhost:8080/

set -e

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
# route a request straight into a worker that is no longer there.
export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo-sglang-cf-agg-$(date +%s)}"
export HF_HOME="${HF_HOME:-/mnt/bigtmp/hf}"

MEDIA_URL="${MEDIA_URL:-file:///tmp/dynamo_media_cf}"

# ---- Cards ----
# Masked rather than shared, so the worker sees exactly the cards listed and
# indexes them from xpu:0. Override with the free cards on your box; they need not
# be contiguous (0,1,3,5 is a valid four-card set).
CARDS="${CARDS:-0}"
NUM_GPUS="${NUM_GPUS:-$(awk -F, '{print NF}' <<<"$CARDS")}"

case "$NUM_GPUS" in
    1|2|4) ;;
    *)
        echo "ERROR: NUM_GPUS=$NUM_GPUS (CARDS=$CARDS) is not supported by this model." >&2
        echo "       Aggregated tensor parallelism needs a width that divides the T5" >&2
        echo "       vocabulary (256384), the DiT FFN (8960) and the DiT hidden size" >&2
        echo "       (1536): only 1, 2 and 4 qualify. A launch at 3 or 5 aborts in" >&2
        echo "       ensure_divisibility partway through loading weights." >&2
        echo "       To use 3 or 5 cards, disaggregate instead:" >&2
        echo "         DIT_CARD=0 VAE_CARD=1,3 bash $SCRIPT_DIR/disagg_omni_causal_forcing_xpu.sh" >&2
        exit 1
        ;;
esac

# ---- Ports ----
SYSTEM_PORT_WORKER="${DYN_SYSTEM_PORT_WORKER:-8091}"
SYSTEM_PORT_ROUTER="${DYN_SYSTEM_PORT_ROUTER:-8093}"
SCHED_PORT="${SCHED_PORT:-5591}"
MASTER_PORT="${MASTER_PORT:-30105}"

# ---- Engine knobs that must not be left to SGLang's defaults ----
# ServerArgs._adjust_offload turns dit_cpu_offload on whenever it is left unset
# (it is a `bool | None` defaulting to None, and every video branch of that method
# sets it True). That default is meant for one-shot clips, where a host->device
# weight copy is paid once per generation. A Causal-Forcing stream calls the DiT
# once per *block* -- 61 times for a 241-frame shot, and forever in a live session
# -- so the copy lands on the critical path of every block and the stream stops
# keeping up with playback. Pinned off.
DIT_CPU_OFFLOAD="${DIT_CPU_OFFLOAD:-false}"

# Attention backend. Left unset, ServerArgs picks `fa` on XPU --
# _set_default_attention_backend only has a considered default for ROCm, and nothing
# measured stands behind `fa` here. At the shape a Causal-Forcing block actually runs
# (832x480: 1560 query tokens attending over a full 21-frame KV window = 32760 keys,
# 12 heads x 128, 30 layers) `torch_sdpa` is much faster on this card: 3.72 ms/layer
# / 84 TFLOPS against 7.49 ms / 42 TFLOPS, on a card whose GEMM peak is 155 TFLOPS.
# Worth 7-8% on the aggregated rows above, and far more once the stages are split
# (see disagg_omni_causal_forcing_xpu.sh: there it is what lets extra decode cards
# matter at all).
ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch_sdpa}"

# The aggregated worker also owns the VAE, so the same parallel-decode caveat as
# the disaggregated decode stage applies: left on `auto`, SGLang refuses to shard
# because the heuristic wants >= auto_parallel_decode_min_latent_elements_per_rank
# (4096) latent elements per rank and a Causal-Forcing block is a *single* latent
# frame -- 1*60*104 = 6240 at 832x480, so 3120 per rank on two cards. The
# threshold is calibrated for one-shot decode of a whole clip; on this path it
# under-counts by the frame count, silently leaving every rank to decode the full
# height redundantly *and* pay the group sync. Asked for explicitly it is a
# straight win, and it is a large part of why the aggregated multi-card rows above
# improve at all: much of 0.884s -> 0.716s -> 0.581s is the decode sharding, not
# the DiT scaling.
VAE_PARALLEL_DECODE_MODE="${VAE_PARALLEL_DECODE_MODE:-spatial_shard}"

EXTRA_ARGS=()
if (( NUM_GPUS > 1 )); then
    # --num-gpus alone is NOT tensor parallelism: it resolves to tp_size=1 with
    # sp_degree=NUM_GPUS (Ulysses sequence parallelism), which adds comms without
    # splitting the weights and makes a 1-step Causal-Forcing block slower. Ask
    # for TP explicitly.
    EXTRA_ARGS+=(--num-gpus "$NUM_GPUS" --tp-size "$NUM_GPUS")
    EXTRA_ARGS+=(--vae-config.parallel-decode-mode "$VAE_PARALLEL_DECODE_MODE")
fi

if (( NUM_GPUS >= 4 )); then
    # At TP=4 the inductor-compiled fuse_scale_shift_kernel dies with
    # UR_RESULT_ERROR_DEVICE_LOST on these cards. torch.compile is worth about 2%
    # on this pipeline, so disabling dynamo costs far less than the crash.
    export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
fi

print_launch_banner --no-curl \
    "Launching Aggregated Causal-Forcing T2V (SGLang-Omni, Intel XPU)" \
    "$MODEL" "$HTTP_PORT" \
    "Namespace:   $DYN_NAMESPACE" \
    "Runtime:     NATS=$NATS_SERVER ETCD=$ETCD_ENDPOINTS" \
    "Cards:       XPU $CARDS ($NUM_GPUS-way$( (( NUM_GPUS > 1 )) && echo " TP, $VAE_PARALLEL_DECODE_MODE decode" ))" \
    "Media:       $MEDIA_URL"

# Clean up the whole process group on the way out. Installed here, not at the top
# of the script: `kill 0` signals this shell too, so an early `exit 1` from the
# unsupported-card-count guard above would be reported as 143 instead of 1 and a
# caller could not tell a rejected configuration from a killed run. Nothing before
# this point has started a background process, so there is nothing to leak.
trap 'echo Cleaning up...; kill 0' EXIT

# ---- Worker: encode + rollout + decode in one process ----
echo "Starting aggregated worker (encoder,denoiser,decoder) on XPU ${CARDS}..."
ZE_AFFINITY_MASK="$CARDS" \
DYN_SYSTEM_PORT="$SYSTEM_PORT_WORKER" \
    python3 -m dynamo.sglang.omni \
    --stage-id 0 \
    --num-stages 1 \
    --model-path "$MODEL" \
    --stage-roles encoder,denoiser,decoder \
    --dit-cpu-offload "$DIT_CPU_OFFLOAD" \
    --attention-backend "$ATTENTION_BACKEND" \
    --scheduler-port "$SCHED_PORT" \
    --master-port "$MASTER_PORT" \
    "${EXTRA_ARGS[@]}" &

# ---- Router: the model the frontend serves ----
# No engine and no card: it addresses the worker by name, and registers the model
# only once the pipeline behind it can actually serve a request.
echo "Starting router..."
DYN_SYSTEM_PORT="$SYSTEM_PORT_ROUTER" \
    python3 -m dynamo.sglang.omni \
    --omni-router \
    --num-stages 1 \
    --model-path "$MODEL" \
    --output-modalities video \
    --media-output-fs-url "$MEDIA_URL" &

# ---- Frontend: OpenAI-compatible HTTP ingress ----
echo "Starting frontend on :${HTTP_PORT}..."
python3 -m dynamo.frontend --http-port "$HTTP_PORT" &

# Readiness is "the model is registered", not "the port answers": the frontend
# serves /health long before the worker has loaded its weights, so wait_for_ready
# alone would report a stack that still 404s every request.
echo "Waiting for the pipeline to register its model (weights load on ${NUM_GPUS} card(s))..."
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
