# LongCat-2.0 NVFP4 PTQ

Standalone toolkit for quantizing **LongCat-2.0** (also marketed as
LongCat-Flash-Pro / OE+DSA) to **NVFP4**, Kimi-K2.5-NVFP4 style.

Extracted from TokenSpeed branch `feature/longcat-2-nvfp4-b200` (`f842d87c`).
This repo is intentionally independent of TokenSpeed so the scripts survive
container rebuilds.

## What gets quantized

| Quantize (NVFP4 W4A4, group 16) | Keep BF16 |
|---------------------------------|-----------|
| shortcut dense `mlps.*` | attention / MLA |
| MoE `mlp.experts.*` | DSA indexer |
| | OE tables / projections |
| | router, `lm_head`, MTP |

KV cache recipe: **FP8**. Calibration: `cnn_dailymail`, algorithm `max`,
default `calib_size=512`.

## ModelOpt patches (do not skip)

NVIDIA ModelOpt **0.41.0** does not include [PR #785](https://github.com/NVIDIA/Model-Optimizer/pull/785).
We do **not** fork Model-Optimizer. `longcat2_ptq/patch_modelopt_export.py`
monkeypatches at runtime:

1. Backport PR #785: lazy-calibrate missing weight `_amax` for dead experts
2. Guard `_export_quantized_weight` so export does not assert on empty amax
3. Fill **input** amax only after accelerate hooks are removed (weights are real tensors)
4. Skip `nn.Identity` zero-experts; do **not** flip upstream `is_moe(LongcatFlashMoE)`

Do not call `set_expert_quantizer_amax` as a pre-pass: with
`--use_seq_device_map` many expert weights are still meta and
`Tensor.item()` crashes.

## Persistent B200 layout

Host `/scratch/yxr/cache` is bind-mounted to container `/home/runner/.cache`.
Keep this repo and Model-Optimizer on that disk.

| Role | Host | Container |
|------|------|-----------|
| This repo | `/scratch/yxr/cache/longcat2-nvfp4-ptq` | `/home/runner/.cache/longcat2-nvfp4-ptq` |
| Model-Optimizer 0.41.0 | `/scratch/yxr/cache/Model-Optimizer` | `/home/runner/.cache/Model-Optimizer` |
| PTQ venv | `/scratch/yxr/cache/venvs/modelopt-ptq` | `/home/runner/.cache/venvs/modelopt-ptq` |
| BF16 snapshot | `.../huggingface/hub/models--meituan-longcat--LongCat-2.0/snapshots/834bf5ffe3047aa9f6cc7a64a9bc068b146b8274` | same under `HF_HOME` |
| Prepared snapshot | `.../huggingface/longcat-2.0-ptq-prepared` | same |
| NVFP4 export | `.../huggingface/longcat-2.0-nvfp4` | same |

Login:

```bash
ssh ubuntu@44.251.69.196
ssh yineng@34.26.31.13
docker exec -it tokenspeed_yxr bash
```

## Steps

### 1. Dependencies (once)

Inside `tokenspeed_yxr`:

```bash
bash /home/runner/.cache/longcat2-nvfp4-ptq/scripts/install_deps.sh
```

This creates/updates:

- venv with `nvidia-modelopt[hf]==0.41.0` (pins `transformers` ~4.57)
- `Model-Optimizer` checkout at tag `0.41.0`

### 2. Activate env

```bash
source /home/runner/.cache/venvs/modelopt-ptq/bin/activate
export HF_HOME=/home/runner/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODELOPT_HOME=/home/runner/.cache/Model-Optimizer
```

### 3. Optional: inspect module families

```bash
SNAP=$HF_HOME/hub/models--meituan-longcat--LongCat-2.0/snapshots/834bf5ffe3047aa9f6cc7a64a9bc068b146b8274
python /home/runner/.cache/longcat2-nvfp4-ptq/longcat2_ptq/analyze_modules.py "$SNAP"
```

### 4. Prepare snapshot (config + remote-code shim, symlink weights)

```bash
PREP=$HF_HOME/longcat-2.0-ptq-prepared
python /home/runner/.cache/longcat2-nvfp4-ptq/longcat2_ptq/prepare_snapshot.py \
  --src "$SNAP" --dst "$PREP" --symlink-weights
```

`run_ptq.sh` does this automatically if `--prepared` is omitted.

### 5. Smoke first (recommended)

```bash
CALIB_SIZE=2 bash /home/runner/.cache/longcat2-nvfp4-ptq/longcat2_ptq/run_ptq_smoke.sh
```

Do not run smoke concurrently with a full 8-GPU job.

### 6. Full PTQ (`calib_size=512`, hours to ~1 day)

```bash
bash /home/runner/.cache/longcat2-nvfp4-ptq/scripts/run_b200_ptq.sh
```

Equivalent:

```bash
bash /home/runner/.cache/longcat2-nvfp4-ptq/longcat2_ptq/run_ptq.sh \
  --src "$SNAP" \
  --prepared "$PREP" \
  --export $HF_HOME/longcat-2.0-nvfp4 \
  --hf-ptq $MODELOPT_HOME/examples/llm_ptq/hf_ptq.py
```

### 7. After export

1. Check `hf_quant_config.json`: `quant_algo=NVFP4`, `group_size=16`, KV FP8
2. If serving needs them, copy ignored OE / MTP weights from the BF16 snapshot
3. Serve: `tokenspeed --quantization nvfp4 --kv-cache-dtype fp8 --attention-backend dsa`
4. If export failed but `modelopt-state.pt` was saved:

```bash
python /home/runner/.cache/longcat2-nvfp4-ptq/longcat2_ptq/export_only.py \
  --prepared "$PREP" \
  --modelopt-state $HF_HOME/longcat-2.0-nvfp4/modelopt-state.pt \
  --export $HF_HOME/longcat-2.0-nvfp4
```

## Layout

```text
longcat2_ptq/run_ptq.sh                 full job
longcat2_ptq/run_ptq_smoke.sh           calib_size=2 wiring check
longcat2_ptq/prepare_snapshot.py        HF config + modeling shim
longcat2_ptq/patch_modelopt_export.py   ModelOpt 0.41 monkeypatches
longcat2_ptq/export_only.py             export from saved state (no recalib)
longcat2_ptq/modeling_longcat2_ptq.py   PTQ modeling shell
recipes/longcat2_nvfp4_mlp_only-kv_fp8.yaml
scripts/install_deps.sh
scripts/run_b200_ptq.sh
```
