#!/bin/bash
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# 2-stage DISAGGREGATED Causal-Forcing (Wan2.1) text-to-VIDEO on Intel XPU.
#   Stage 0: DiT  — text encode + framewise rollout -> video latents
#   Stage 1: VAE  — decode latents -> mp4
#   Router : orchestrates the 2-stage DAG, formats the video response
#   Frontend: OpenAI-compatible HTTP ingress (POST /v1/videos)
#
# Serves the vLLM-Omni "causal_forcing_disagg" pipeline. Requires the vLLM-Omni
# enablement branch on PYTHONPATH:
#   git clone https://github.com/dsocek/vllm-omni.git
#   cd vllm-omni && git checkout causal-forcing-wan-video
#
# Usage (all vars overridable; defaults suit a 2-card XPU box):
#   VLLM_OMNI_REPO=/path/to/vllm-omni \
#   MODEL=/path/to/causal-forcing-1step \
#   ZE_AFFINITY_MASK=0,2 DYN_HTTP_PORT=8880 \
#   bash disagg_omni_causal_forcing_xpu.sh
#
# XPU / card policy: the two physical cards in ZE_AFFINITY_MASK (default 0,2)
# are the only ones used. Every worker is masked to them; the deploy YAML's
# logical devices (stage0 devices:"0", stage1 devices:"1") map to the first and
# second card in the mask -> nothing can touch another card even if misrouted.
#
# Modeled on ../disagg_omni_glm_image.sh (AR|DiT image), adapted for:
#   - XPU: ZE_AFFINITY_MASK instead of CUDA_VISIBLE_DEVICES, VLLM_TARGET_DEVICE=xpu
#   - video output: --output-modalities video, /v1/videos endpoint
#
# Prereqs: etcd + nats-server running (nats needs `max_payload: 64MB`).
#   etcd --listen-client-urls http://0.0.0.0:${PORT_ETCD:-2370} \
#        --advertise-client-urls http://0.0.0.0:${PORT_ETCD:-2370} \
#        --listen-peer-urls http://localhost:2371 --data-dir /tmp/etcd &
#   printf 'max_payload: 64MB\n' > /tmp/nats.conf
#   nats-server -js --port ${PORT_NATS:-4223} -c /tmp/nats.conf &
set -e

trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-./causal-forcing-1step}"

# Import the vLLM-Omni REPO TREE (has the causal_forcing_disagg pipeline +
# DiT/VAE classes + fixes) instead of the installed dist-packages copy.
# Prepending PYTHONPATH makes every worker import the repo tree. Point this at
# your clone of dsocek/vllm-omni @ causal-forcing-wan-video.
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-../vllm-omni}"
export PYTHONPATH="${VLLM_OMNI_REPO}${PYTHONPATH:+:$PYTHONPATH}"

# vllm-omni's built-in disaggregated deploy YAML (pipeline: causal_forcing_disagg).
if [ -z "${STAGE_CONFIG:-}" ]; then
    STAGE_CONFIG="${VLLM_OMNI_REPO}/vllm_omni/deploy/causal_forcing_disagg.yaml"
fi

# --- Runtime plane (match the running etcd/nats on this box) ---
IP_LOCAL="${IP_LOCAL:-$(hostname -I | awk '{print $1}')}"
PORT_NATS="${PORT_NATS:-4223}"
PORT_ETCD="${PORT_ETCD:-2370}"
HTTP_PORT="${DYN_HTTP_PORT:-8880}"

export VLLM_TARGET_DEVICE=xpu
export VLLM_XPU_ENABLE_XPU_GRAPH=0
export HF_HOME="${HF_HOME:-/mnt/bigtmp/hf}"
export NATS_SERVER="nats://${IP_LOCAL}:${PORT_NATS}"
export ETCD_ENDPOINTS="http://${IP_LOCAL}:${PORT_ETCD}"
export DYN_REQUEST_PLANE=tcp
export ETCD_LEASE_TTL=600
export PYTHONHASHSEED=0
# Fresh namespace each run so stale discovery can't route straight to a stage.
export DYN_NAMESPACE="${DYN_NAMESPACE:-dynamo-omni-cf-$(date +%s)}"

# Which two physical cards to use, for EVERY worker. The deploy YAML's logical
# devices 0/1 map to the first/second card in this mask (stage0/DiT -> first,
# stage1/VAE -> second). Override via ZE_AFFINITY_MASK env; defaults to 0,2.
# Only free cards should be listed (others are reserved by other users).
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0,2}"

MEDIA_URL="${MEDIA_URL:-file:///tmp/dynamo_media_cf}"

echo "Model:        ${MODEL}"
echo "Stage config: ${STAGE_CONFIG}"
echo "Namespace:    ${DYN_NAMESPACE}"
echo "Runtime:      NATS=${NATS_SERVER}  ETCD=${ETCD_ENDPOINTS}"
echo "Cards:        ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK} (stage0/DiT->first, stage1/VAE->second)"
echo "HTTP:         http://localhost:${HTTP_PORT}/v1/videos"
echo

# Stage 0: DiT worker -> latents. Logical device 0 = physical card 0.
echo "Starting Stage 0 (DiT)..."
DYN_SYSTEM_PORT=8091 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 0 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities video \
    --media-output-fs-url "$MEDIA_URL" \
    --enforce-eager &
sleep 25

# Stage 1: VAE worker -> mp4. Logical device 1 = physical card 2.
echo "Starting Stage 1 (VAE)..."
DYN_SYSTEM_PORT=8092 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --stage-id 1 \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities video \
    --media-output-fs-url "$MEDIA_URL" \
    --enforce-eager &
sleep 25

# Router: discovers the two stage workers, orchestrates DiT -> VAE, formats video.
echo "Starting Router..."
DYN_SYSTEM_PORT=8093 \
    python -m dynamo.vllm.omni \
    --model "$MODEL" \
    --omni-router \
    --stage-configs-path "$STAGE_CONFIG" \
    --output-modalities video \
    --media-output-fs-url "$MEDIA_URL" &
sleep 8

# Frontend: OpenAI-compatible HTTP ingress.
echo "Starting Frontend on :${HTTP_PORT}..."
python -m dynamo.frontend --http-port "${HTTP_PORT}" &

echo
echo "Ready. Example request:"
cat <<CURL
curl -s http://localhost:${HTTP_PORT}/v1/videos \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "${MODEL}",
    "prompt": "A serene lakeside sunrise with mist over the water.",
    "size": "832x480",
    "response_format": "url",
    "nvext": { "num_inference_steps": 1, "num_frames": 81 }
  }' | jq
CURL

wait
