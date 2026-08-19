#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Portable full-job launcher. Override paths with env vars or flags.
#
#   source "$CACHE/venvs/modelopt-ptq/bin/activate"
#   bash scripts/run_ptq.sh --src /path/to/LongCat-2.0 --export /path/to/out

set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_TRUST_REMOTE_CODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${CACHE:-${HOME}/.cache}"
VENV="${VENV:-${CACHE}/venvs/modelopt-ptq}"
MODELOPT_HOME="${MODELOPT_HOME:-${CACHE}/Model-Optimizer}"
SNAP="${SNAP:-}"
PREP="${PREP:-}"
EXPORT="${EXPORT:-${HF_HOME}/longcat-2.0-nvfp4}"
CALIB_SIZE="${CALIB_SIZE:-512}"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SNAP="$2"; shift 2 ;;
    --export) EXPORT="$2"; shift 2 ;;
    --prepared) PREP="$2"; shift 2 ;;
    --calib-size) CALIB_SIZE="$2"; shift 2 ;;
    --hf-ptq) MODELOPT_PTQ="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ -z "${SNAP}" ]]; then
  echo "Usage: $0 --src <LongCat-2.0 snapshot dir> --export <out_dir>" >&2
  echo "Optional env: CACHE HF_HOME CUDA_VISIBLE_DEVICES VENV MODELOPT_HOME CALIB_SIZE" >&2
  exit 2
fi

if [[ -z "${PREP}" ]]; then
  PREP="${EXPORT%/}-prepared"
fi

MODELOPT_PTQ="${MODELOPT_PTQ:-${MODELOPT_HOME}/examples/llm_ptq/hf_ptq.py}"
if [[ ! -f "${MODELOPT_PTQ}" ]]; then
  echo "hf_ptq.py not found at ${MODELOPT_PTQ}. Run scripts/install_deps.sh first." >&2
  exit 2
fi

if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  export CUDA_VISIBLE_DEVICES
fi

exec bash "${ROOT}/longcat2_ptq/run_ptq.sh" \
  --src "${SNAP}" \
  --prepared "${PREP}" \
  --export "${EXPORT}" \
  --hf-ptq "${MODELOPT_PTQ}" \
  --calib-size "${CALIB_SIZE}" \
  "${EXTRA[@]+"${EXTRA[@]}"}"
