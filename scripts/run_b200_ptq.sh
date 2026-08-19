#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# B200 / tokenspeed_yxr convenience launcher. Paths assume the scratch bind:
#   host /scratch/yxr/cache  ->  container /home/runner/.cache

set -euo pipefail

export PATH="/home/runner/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-/home/runner/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_TRUST_REMOTE_CODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/home/runner/.cache/venvs/modelopt-ptq}"
MODELOPT_HOME="${MODELOPT_HOME:-/home/runner/.cache/Model-Optimizer}"
SNAP="${SNAP:-${HF_HOME}/hub/models--meituan-longcat--LongCat-2.0/snapshots/834bf5ffe3047aa9f6cc7a64a9bc068b146b8274}"
PREP="${PREP:-${HF_HOME}/longcat-2.0-ptq-prepared}"
EXPORT="${EXPORT:-${HF_HOME}/longcat-2.0-nvfp4}"
CALIB_SIZE="${CALIB_SIZE:-512}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

exec bash "${ROOT}/longcat2_ptq/run_ptq.sh" \
  --src "${SNAP}" \
  --prepared "${PREP}" \
  --export "${EXPORT}" \
  --hf-ptq "${MODELOPT_HOME}/examples/llm_ptq/hf_ptq.py" \
  --calib-size "${CALIB_SIZE}" \
  "$@"
