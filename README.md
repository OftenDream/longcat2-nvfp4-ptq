# LongCat-2.0 NVFP4 PTQ

Quantize [meituan-longcat/LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0)
(also marketed as LongCat-Flash-Pro / OE+DSA) to **NVFP4** on any multi-GPU
Blackwell machine. The recipe follows
[nvidia/Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4).

This repo ships:

1. A **pinned ModelOpt stack**: `nvidia-modelopt[hf]==0.41.0` plus NVIDIA
   `Model-Optimizer` tag `0.41.0`
2. **LongCat export patches** (0.41.0 does not include
   [PR #785](https://github.com/NVIDIA/Model-Optimizer/pull/785))
3. **Scripts** that prepare the snapshot, calibrate, and export an HF NVFP4
   checkpoint

No host-specific login or container is required. Point `--src` at a local BF16
snapshot and `--export` at a local output directory.

## What gets quantized

| Quantize (NVFP4 W4A4, group 16) | Keep BF16 |
|---------------------------------|-----------|
| shortcut dense `mlps.*` | attention / MLA |
| MoE `mlp.experts.*` | DSA indexer |
| | OE tables / projections |
| | router, `lm_head`, MTP |

KV cache: **FP8**. Calibration: `cnn_dailymail`, `algorithm=max`, default
`calib_size=512`.

## Requirements

- NVIDIA **Blackwell** (B200 / GB200, …); 8 GPUs recommended
- Disk: ~**3.3 TB** for BF16, ~**0.8 TB** for the export, plus calib cache
- Python 3.12 and a CUDA build of PyTorch
- Access to GitHub and PyPI (or your own mirrors)

## 1. Install ModelOpt and the scripts

```bash
git clone https://github.com/OftenDream/longcat2-nvfp4-ptq.git
cd longcat2-nvfp4-ptq

# Optional: put the venv and Model-Optimizer clone on a large disk
export CACHE=/data/ptq-cache          # default: $HOME/.cache
bash scripts/install_deps.sh
source $CACHE/venvs/modelopt-ptq/bin/activate
```

`install_deps.sh` will:

- create a venv and install **`nvidia-modelopt[hf]==0.41.0`** (pins
  `transformers` to 4.57.x)
- clone **`Model-Optimizer` tag 0.41.0** to `$CACHE/Model-Optimizer`
  (provides `hf_ptq.py`)

Do not bump ModelOpt unless you re-validate the patches.

## 2. Stage the BF16 weights

Place the LongCat-2.0 BF16 snapshot anywhere on the machine:

```bash
export HF_HOME=/data/huggingface
huggingface-cli download meituan-longcat/LongCat-2.0 --local-dir "$HF_HOME/LongCat-2.0"
# or reuse an existing snapshot
export SNAP=$HF_HOME/LongCat-2.0
export EXPORT=$HF_HOME/longcat-2.0-nvfp4
```

`SNAP` only needs `config.json` plus `model-*-of-*.safetensors`.

## 3. Run PTQ

Smoke first (still loads the full model; calib is 2 samples):

```bash
export MODELOPT_HOME=$CACHE/Model-Optimizer
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # match your GPUs
CALIB_SIZE=2 bash longcat2_ptq/run_ptq_smoke.sh
```

Full job (default `calib_size=512`, hours to about a day):

```bash
bash scripts/run_ptq.sh \
  --src "$SNAP" \
  --export "$EXPORT"
```

`run_ptq.sh` will:

1. run `prepare_snapshot.py` (config / PTQ modeling; weights default to symlinks)
2. apply `patch_modelopt_export.py`
3. call ModelOpt `hf_ptq.py` with this repo's recipe

## 4. After export

1. Check `$EXPORT/hf_quant_config.json`: `quant_algo=NVFP4`, `group_size=16`, KV FP8
2. If serving needs OE / MTP, copy those tables back from the BF16 snapshot
3. TokenSpeed example: `--quantization nvfp4 --kv-cache-dtype fp8 --attention-backend dsa`
4. If export failed but `modelopt-state.pt` was saved, skip recalibration:

```bash
python longcat2_ptq/export_only.py \
  --prepared "${EXPORT}-prepared" \
  --modelopt-state "$EXPORT/modelopt-state.pt" \
  --export "$EXPORT"
```

## ModelOpt patches (applied automatically)

0.41.0 does not include PR #785. We do not fork upstream. The scripts
monkeypatch just before `hf_ptq`:

1. Backport PR #785: lazy-calibrate missing weight `_amax` for dead experts
2. Guard `_export_quantized_weight` again
3. Fill **input** amax only after accelerate hooks are removed (real tensors)
4. Skip `nn.Identity` zero-experts; do not mark `LongcatFlashMoE` as upstream `is_moe`

Do not call `set_expert_quantizer_amax` as a pre-pass. With
`--use_seq_device_map` many expert weights are still meta and
`Tensor.item()` raises.

## Layout

```text
scripts/install_deps.sh                 install ModelOpt 0.41.0 + hf_ptq
scripts/run_ptq.sh                      full-job entry point (any host)
longcat2_ptq/run_ptq.sh                 same job, invoked by scripts/run_ptq.sh
longcat2_ptq/run_ptq_smoke.sh           calib_size=2 wiring check
longcat2_ptq/prepare_snapshot.py
longcat2_ptq/patch_modelopt_export.py   0.41.0 export patches
longcat2_ptq/export_only.py             re-export from a saved state
recipes/longcat2_nvfp4_mlp_only-kv_fp8.yaml
```
