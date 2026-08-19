# Copyright (c) 2026 LightSeek Foundation
"""Re-run LongCat NVFP4 HF export from a saved ModelOpt *state* (no recalib).

After calib, the launcher should have written a lightweight
``modelopt_state`` (quantizer amax / configs, not full weights)::

    python longcat2_ptq/export_only.py \\
      --prepared \$HF_HOME/longcat-2.0-ptq-prepared \\
      --modelopt-state \$HF_HOME/longcat-2.0-nvfp4-modelopt-state.pt \\
      --export \$HF_HOME/longcat-2.0-nvfp4

Still loads the BF16 prepared snapshot once (minutes), then restores
quantizer state and exports — skips the ~22h calib loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", required=True, type=Path)
    ap.add_argument(
        "--modelopt-state",
        required=True,
        type=Path,
        help="torch.save(mto.modelopt_state(model)) path",
    )
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--patch-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--gpu-max-mem-percentage", type=float, default=0.85)
    args = ap.parse_args()

    sys.path.insert(0, str(args.patch_dir))
    sys.path.insert(0, str(args.prepared))

    from patch_modelopt_export import apply_longcat_modelopt_export_patches

    apply_longcat_modelopt_export_patches()

    cfg_path = args.prepared / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("auto_map", None)
    cfg["model_type"] = "longcat"
    cfg["architectures"] = ["LongcatCausalLM"]
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    from configuration_longcat2_ptq import LongcatConfig
    from modeling_longcat2_ptq import LongcatCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    AutoConfig.register("longcat", LongcatConfig, exist_ok=True)
    AutoModelForCausalLM.register(LongcatConfig, LongcatCausalLM, exist_ok=True)

    import torch
    import modelopt.torch.opt as mto
    from modelopt.torch.export import export_hf_checkpoint

    print(f"Loading BF16 prepared model from {args.prepared}", flush=True)
    # Mirror hf_ptq seq_device_map / offload settings used for the full job.
    max_mem = {i: f"{int(args.gpu_max_mem_percentage * 180)}GiB" for i in range(torch.cuda.device_count())}
    max_mem["cpu"] = "4000GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.prepared,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        low_cpu_mem_usage=True,
    )

    state = torch.load(args.modelopt_state, map_location="cpu", weights_only=False)
    print(f"Restoring ModelOpt state from {args.modelopt_state}", flush=True)
    mto.restore_from_modelopt_state(model, state)

    args.export.mkdir(parents=True, exist_ok=True)
    print(f"Exporting HF NVFP4 checkpoint -> {args.export}", flush=True)
    export_hf_checkpoint(model, export_dir=str(args.export))

    try:
        tok = AutoTokenizer.from_pretrained(args.prepared, trust_remote_code=True)
        tok.save_pretrained(args.export)
    except Exception as exc:
        print(f"tokenizer save skipped: {exc}", flush=True)

    print("export_only done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
