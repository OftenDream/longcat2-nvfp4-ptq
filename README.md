# LongCat-2.0 NVFP4 PTQ

在**任意**多卡 Blackwell 机器上，把 [meituan-longcat/LongCat-2.0](https://huggingface.co/meituan-longcat/LongCat-2.0)
（对外也叫 LongCat-Flash-Pro / OE+DSA）量化成 **NVFP4**。策略对齐
[nvidia/Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)。

本仓库提供：

1. **锁定版本的 ModelOpt**：`nvidia-modelopt[hf]==0.41.0` + NVIDIA
   `Model-Optimizer` tag `0.41.0`
2. **LongCat 导出补丁**（0.41.0 缺少 [PR #785](https://github.com/NVIDIA/Model-Optimizer/pull/785)）
3. **一键脚本**：准备 snapshot → 校准 → 导出 HF NVFP4 checkpoint

不依赖任何特定机器、容器或内网路径。输入是本机上的 BF16 snapshot 目录，
输出是本机上的 NVFP4 目录。

## 量化范围

| 量化（NVFP4 W4A4，group 16） | 保持 BF16 |
|------------------------------|-----------|
| shortcut dense `mlps.*` | attention / MLA |
| MoE `mlp.experts.*` | DSA indexer |
| | OE tables / projections |
| | router、`lm_head`、MTP |

KV：`FP8`。校准：`cnn_dailymail`，`algorithm=max`，默认 `calib_size=512`。

## 环境要求

- NVIDIA **Blackwell**（B200 / GB200 等），建议 8 卡
- 磁盘：BF16 约 **3.3 TB**，导出约 **0.8 TB**，再留校准缓存
- Python 3.12、CUDA 能被 PyTorch 看到
- 能访问 GitHub 与 PyPI（或你们自己的镜像）

## 1. 安装 ModelOpt 与脚本

```bash
git clone https://github.com/OftenDream/longcat2-nvfp4-ptq.git
cd longcat2-nvfp4-ptq

# 可选：把依赖装到指定盘
export CACHE=/data/ptq-cache          # 默认 $HOME/.cache
bash scripts/install_deps.sh
source $CACHE/venvs/modelopt-ptq/bin/activate
```

`install_deps.sh` 会：

- 创建 venv，安装 **`nvidia-modelopt[hf]==0.41.0`**（会把 `transformers` 钉在 4.57.x）
- clone **`Model-Optimizer` tag 0.41.0** 到 `$CACHE/Model-Optimizer`（提供 `hf_ptq.py`）

不要换更新的 ModelOpt，除非你愿意自己验证补丁。

## 2. 准备权重

把 LongCat-2.0 BF16 snapshot 放到本机任意目录，例如：

```bash
export HF_HOME=/data/huggingface
huggingface-cli download meituan-longcat/LongCat-2.0 --local-dir "$HF_HOME/LongCat-2.0"
# 或使用已有 snapshot 路径
export SNAP=$HF_HOME/LongCat-2.0
export EXPORT=$HF_HOME/longcat-2.0-nvfp4
```

`SNAP` 只需是含 `config.json` + `model-*-of-*.safetensors` 的目录。

## 3. 跑量化

先用 2 条校准样本验证通路（仍会加载整模，但校准很快）：

```bash
export MODELOPT_HOME=$CACHE/Model-Optimizer
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 按你的卡改
CALIB_SIZE=2 bash longcat2_ptq/run_ptq_smoke.sh
```

全量（默认 `calib_size=512`，通常数小时到一天）：

```bash
bash scripts/run_ptq.sh \
  --src "$SNAP" \
  --export "$EXPORT"
```

`run_ptq.sh` 会自动：

1. `prepare_snapshot.py`：改 config / 挂上 PTQ modeling，权重默认 symlink
2. 应用 `patch_modelopt_export.py`
3. 调用 ModelOpt `hf_ptq.py` + 本仓库 recipe

## 4. 导出后检查

1. `$EXPORT/hf_quant_config.json`：`quant_algo=NVFP4`，`group_size=16`，KV FP8
2. serving 如果需要 OE / MTP，从 BF16 snapshot 拷回未被导出的表
3. TokenSpeed 示例：`--quantization nvfp4 --kv-cache-dtype fp8 --attention-backend dsa`
4. 若 export 失败但留下了 `modelopt-state.pt`，可跳过重校准：

```bash
python longcat2_ptq/export_only.py \
  --prepared "${EXPORT}-prepared" \
  --modelopt-state "$EXPORT/modelopt-state.pt" \
  --export "$EXPORT"
```

## ModelOpt 补丁（脚本会自动打上）

0.41.0 没有 PR #785。我们不改上游 git，只在调用 `hf_ptq` 前 monkeypatch：

1. 回移植 PR #785：dead expert 缺 `_amax` 时从权重补算
2. `_export_quantized_weight` 前再 guard 一次
3. 卸掉 accelerate hook 之后再补 **input** amax（此时权重已是真实 tensor）
4. 跳过 `nn.Identity` zero-expert；不要把 `LongcatFlashMoE` 标成上游 `is_moe`

不要在 export 前调用 `set_expert_quantizer_amax`：`--use_seq_device_map` 下
大量 expert 仍是 meta，`Tensor.item()` 会炸。

## 目录

```text
scripts/install_deps.sh                 安装 ModelOpt 0.41.0 + hf_ptq
scripts/run_ptq.sh                      任意机器上的全量入口
longcat2_ptq/run_ptq.sh                 同上（被 scripts/run_ptq.sh 调用）
longcat2_ptq/run_ptq_smoke.sh           calib_size=2 通路检查
longcat2_ptq/prepare_snapshot.py
longcat2_ptq/patch_modelopt_export.py   0.41.0 导出补丁
longcat2_ptq/export_only.py             从已保存 state 再导出
recipes/longcat2_nvfp4_mlp_only-kv_fp8.yaml
```
