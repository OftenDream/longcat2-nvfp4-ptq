#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Prepare LongCat-2.0 + run NVIDIA ModelOpt NVFP4 PTQ (Kimi-K2.5-NVFP4 style).
#
# Prerequisites:
#   pip install "nvidia-modelopt[hf]==0.41.0"
#   Model-Optimizer tag 0.41.0 (examples/llm_ptq/hf_ptq.py)
#   Multi-GPU Blackwell node(s); keep weights on a large disk (HF_HOME).
#
# Example:
#   export HF_HOME=/home/runner/.cache/huggingface
#   bash longcat2_ptq/run_ptq.sh \
#     --src $HF_HOME/hub/models--meituan-longcat--LongCat-2.0/snapshots/<rev> \
#     --export $HF_HOME/longcat-2.0-nvfp4

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="${ROOT}/recipes/longcat2_nvfp4_mlp_only-kv_fp8.yaml"
SRC=""
EXPORT=""
PREPARED=""
CALIB_SIZE="${CALIB_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MODELOPT_PTQ="${MODELOPT_PTQ:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --export) EXPORT="$2"; shift 2 ;;
    --prepared) PREPARED="$2"; shift 2 ;;
    --recipe) RECIPE="$2"; shift 2 ;;
    --calib-size) CALIB_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --hf-ptq) MODELOPT_PTQ="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${SRC}" || -z "${EXPORT}" ]]; then
  echo "Usage: $0 --src <longcat2_snapshot> --export <out_dir> [--prepared <dir>]" >&2
  exit 2
fi

if [[ -z "${PREPARED}" ]]; then
  PREPARED="${EXPORT%/}-prepared"
fi

python3 "${ROOT}/longcat2_ptq/analyze_modules.py" "${SRC}"
python3 "${ROOT}/longcat2_ptq/prepare_snapshot.py" \
  --src "${SRC}" \
  --dst "${PREPARED}" \
  --symlink-weights

if [[ -z "${MODELOPT_PTQ}" ]]; then
  for cand in \
    "${MODELOPT_HOME:-}/examples/llm_ptq/hf_ptq.py" \
    "${ROOT}/../Model-Optimizer/examples/llm_ptq/hf_ptq.py" \
    "${HOME}/.cache/Model-Optimizer/examples/llm_ptq/hf_ptq.py" \
    "${HOME}/Model-Optimizer/examples/llm_ptq/hf_ptq.py"
  do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      MODELOPT_PTQ="${cand}"
      break
    fi
  done
  if [[ -z "${MODELOPT_PTQ}" ]]; then
    echo "Set --hf-ptq /path/to/Model-Optimizer/examples/llm_ptq/hf_ptq.py" >&2
    exit 2
  fi
fi

echo "Running ModelOpt PTQ..."
echo "  model : ${PREPARED}"
echo "  recipe: ${RECIPE}"
echo "  export: ${EXPORT}"

# Fast CPU patch tests first (seconds). Full GPU job only if these pass.
export PYTHONPATH="${ROOT}/longcat2_ptq:${PYTHONPATH:-}"
python3 "${ROOT}/longcat2_ptq/test_patch_modelopt_export.py"

# Prefer smoke before spending a day:
#   CALIB_SIZE=2 bash longcat2_ptq/run_ptq_smoke.sh
#
# Kimi-style patches: PR#785 weight amax lazy-calib only (no meta-unsafe pre-pass).
python3 - <<PY
from patch_modelopt_export import apply_longcat_modelopt_export_patches
apply_longcat_modelopt_export_patches()
import runpy, sys
sys.argv = [
    "hf_ptq.py",
    "--model", r"""${PREPARED}""",
    "--recipe", r"""${RECIPE}""",
    "--export_path", r"""${EXPORT}""",
    "--calib_size", r"""${CALIB_SIZE}""",
    "--batch_size", r"""${BATCH_SIZE}""",
    "--trust_remote_code",
]
runpy.run_path(r"""${MODELOPT_PTQ}""", run_name="__main__")
PY

echo "PTQ finished: ${EXPORT}"
echo "Post-steps:"
echo "  1) Confirm hf_quant_config.json has quant_algo=NVFP4, group_size=16, kv FP8"
echo "  2) Copy ignored OE tables / MTP weights from BF16 snapshot if serving needs them"
echo "  3) Serve with tokenspeed --quantization nvfp4 --kv-cache-dtype fp8 --attention-backend dsa"
echo "  4) If export fails but modelopt_state was saved, retry with export_only.py (no recalib)"
