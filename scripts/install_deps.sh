#!/usr/bin/env bash
# Copyright (c) 2026 LightSeek Foundation
#
# Install the supported ModelOpt stack on this machine:
#   nvidia-modelopt[hf]==0.41.0
#   Model-Optimizer tag 0.41.0 (hf_ptq.py)
#
# Safe to re-run. Override locations with CACHE / VENV / MODELOPT_HOME.

set -euo pipefail

CACHE="${CACHE:-${HOME}/.cache}"
VENV="${VENV:-${CACHE}/venvs/modelopt-ptq}"
MODELOPT_HOME="${MODELOPT_HOME:-${CACHE}/Model-Optimizer}"

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install "nvidia-modelopt[hf]==0.41.0"

if [[ ! -f "${MODELOPT_HOME}/examples/llm_ptq/hf_ptq.py" ]]; then
  rm -rf "${MODELOPT_HOME}"
  git clone --depth 1 --branch 0.41.0 \
    https://github.com/NVIDIA/Model-Optimizer.git "${MODELOPT_HOME}"
else
  git -C "${MODELOPT_HOME}" fetch --depth 1 origin tag 0.41.0 || true
  git -C "${MODELOPT_HOME}" checkout 0.41.0
fi

python - <<'PY'
import modelopt, transformers
print("modelopt", modelopt.__version__)
print("transformers", transformers.__version__)
print("hf_ptq OK")
PY

echo "Installed:"
echo "  venv            ${VENV}"
echo "  Model-Optimizer ${MODELOPT_HOME}"
echo "Activate with:  source ${VENV}/bin/activate"
