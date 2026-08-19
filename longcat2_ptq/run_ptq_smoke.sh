#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Cheap end-to-end smoke for LongCat NVFP4 export wiring.
# Same stack as the full job, but calib_size defaults to 2 (~minutes of calib
# after model load, not ~22h). Use this to validate patches before a full run.
#
# Example:
#   CALIB_SIZE=2 bash longcat2_ptq/run_ptq_smoke.sh
#
# Notes:
#   * Still loads the full BF16 shard set once (~5-10 min).
#   * Export may still be heavy if it gets far; the goal is to fail fast on
#     meta/amax/MoE wiring bugs at the start of export.
#   * Do NOT run concurrently with a full 8-GPU PTQ job.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIB_SIZE="${CALIB_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXPORT="${EXPORT:-${HF_HOME:-$HOME/.cache/huggingface}/longcat-2.0-nvfp4-smoke}"
PREPARED="${PREPARED:-${HF_HOME:-$HOME/.cache/huggingface}/longcat-2.0-ptq-prepared}"
PATCH_DIR="${PATCH_DIR:-${ROOT}/longcat2_ptq}"
MODELOPT_PTQ="${MODELOPT_PTQ:-}"

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
fi

if [[ ! -d "${PREPARED}" ]]; then
  echo "Prepared snapshot missing: ${PREPARED}" >&2
  exit 2
fi
if [[ -z "${MODELOPT_PTQ}" || ! -f "${MODELOPT_PTQ}" ]]; then
  echo "hf_ptq.py missing. Set MODELOPT_PTQ or -- clone Model-Optimizer 0.41.0" >&2
  exit 2
fi

echo "=== LongCat NVFP4 SMOKE ==="
echo "  prepared : ${PREPARED}"
echo "  export   : ${EXPORT}"
echo "  calib    : ${CALIB_SIZE} (smoke)"
echo "  patches  : ${PATCH_DIR}"

export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"
python3 "${PATCH_DIR}/test_patch_modelopt_export.py"

mkdir -p "${EXPORT}"
export PYTHONPATH="${PATCH_DIR}:${PREPARED}:$(dirname "${MODELOPT_PTQ}"):${PYTHONPATH:-}"

python3 - <<PY
import json, sys
from pathlib import Path

p = Path(r"""${PREPARED}""") / "config.json"
cfg = json.loads(p.read_text())
cfg.pop("auto_map", None)
cfg["model_type"] = "longcat"
cfg["architectures"] = ["LongcatCausalLM"]
p.write_text(json.dumps(cfg, indent=2) + "\n")

sys.path.insert(0, r"""${PATCH_DIR}""")
sys.path.insert(0, r"""${PREPARED}""")
sys.path.insert(0, r"""$(dirname "${MODELOPT_PTQ}")""")

from patch_modelopt_export import apply_longcat_modelopt_export_patches
apply_longcat_modelopt_export_patches()

from configuration_longcat2_ptq import LongcatConfig
from modeling_longcat2_ptq import LongcatCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("longcat", LongcatConfig, exist_ok=True)
AutoModelForCausalLM.register(LongcatConfig, LongcatCausalLM, exist_ok=True)

import torch
import modelopt.torch.export.unified_export_hf as ue
import modelopt.torch.opt as mto
_orig = ue.export_hf_checkpoint
state_path = Path(r"""${EXPORT}""") / "modelopt-state.pt"

def _export(model, *args, **kwargs):
    try:
        torch.save(mto.modelopt_state(model), state_path)
        print(f"[smoke] saved modelopt_state -> {state_path}", flush=True)
    except Exception as e:
        print(f"[smoke] modelopt_state save failed: {e}", flush=True)
    return _orig(model, *args, **kwargs)

ue.export_hf_checkpoint = _export

import runpy
sys.argv = [
    "hf_ptq.py",
    "--pyt_ckpt_path", r"""${PREPARED}""",
    "--qformat", "nvfp4_mlp_only",
    "--kv_cache_qformat", "fp8",
    "--export_path", r"""${EXPORT}""",
    "--dataset", "cnn_dailymail",
    "--calib_size", r"""${CALIB_SIZE}""",
    "--batch_size", r"""${BATCH_SIZE}""",
    "--trust_remote_code",
    "--use_seq_device_map",
    "--gpu_max_mem_percentage", "0.85",
]
runpy.run_path(r"""${MODELOPT_PTQ}""", run_name="__main__")
print("[smoke] completed OK")
PY
