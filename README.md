# LongCat-2.0 NVFP4 PTQ

Turn [LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0) into an
NVFP4 checkpoint. You need a Blackwell box (B200 / GB200, 8 GPUs is easiest)
and a lot of disk (~3.3 TB for the original model, ~0.8 TB for the output).

## Run

**1. Clone this repo**

```bash
git clone https://github.com/OftenDream/longcat2-nvfp4-ptq.git
cd longcat2-nvfp4-ptq
```

**2. Install the pinned ModelOpt (once)**

Put the install on a big disk if `$HOME` is small:

```bash
export CACHE=/data/ptq-cache          # or leave unset to use $HOME/.cache
bash scripts/install_deps.sh
source $CACHE/venvs/modelopt-ptq/bin/activate
```

**3. Get the BF16 model onto this machine**

```bash
export SNAP=/data/LongCat-2.0
export EXPORT=/data/longcat-2.0-nvfp4

# skip this if you already have the weights
huggingface-cli download meituan-longcat/LongCat-2.0 --local-dir "$SNAP"
```

`$SNAP` should be a folder that contains `config.json` and `model-*-of-*.safetensors`.

**4. Quantize**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # your GPU ids
bash scripts/run_ptq.sh --src "$SNAP" --export "$EXPORT"
```

This takes hours to about a day. When it finishes, the NVFP4 model is in `$EXPORT`.

Optional tiny check before the full job (still loads the whole model):

```bash
CALIB_SIZE=2 bash longcat2_ptq/run_ptq_smoke.sh
```

## The two ModelOpt changes (how users get them)

Install the **stock** NVIDIA stack only (`nvidia-modelopt==0.41.0` and
`Model-Optimizer` tag `0.41.0`). Do **not** look for a forked ModelOpt wheel.

Both changes live in this repo and are applied at runtime when you run
`scripts/run_ptq.sh` / `run_ptq_smoke.sh` / `export_only.py`:

| # | Change | Why |
|---|--------|-----|
| 1 | Backport [PR #785](https://github.com/NVIDIA/Model-Optimizer/pull/785) | 0.41.0 asserts if a dead MoE expert has no weight `_amax`. The patch lazy-calibrates amax from the real weight at export time (same path that unblocked Kimi-K2.5-NVFP4). |
| 2 | LongCat MoE export timing | Fill **input** amax only after accelerate hooks are removed, skip `nn.Identity` zero-experts, and do **not** flip upstream `is_moe(LongcatFlashMoE)`. A pre-pass `set_expert_quantizer_amax` on meta weights raises `Tensor.item()`. |

Source: [`longcat2_ptq/patch_modelopt_export.py`](longcat2_ptq/patch_modelopt_export.py).
`run_ptq.sh` imports it and calls `apply_longcat_modelopt_export_patches()` in the
same process as `hf_ptq.py`.

If you invoke `hf_ptq.py` yourself, load the patch first:

```bash
export PYTHONPATH=/path/to/longcat2-nvfp4-ptq/longcat2_ptq:$PYTHONPATH
python - <<'PY'
from patch_modelopt_export import apply_longcat_modelopt_export_patches
apply_longcat_modelopt_export_patches()
import runpy, sys
sys.argv = ["hf_ptq.py", "--model", "...", "--recipe", "...", "--export_path", "...", "--trust_remote_code"]
runpy.run_path("/path/to/Model-Optimizer/examples/llm_ptq/hf_ptq.py", run_name="__main__")
PY
```
