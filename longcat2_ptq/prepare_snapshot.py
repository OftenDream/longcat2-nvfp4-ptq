#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
"""Prepare a LongCat-2.0 snapshot for ModelOpt hf_ptq (config + remote code)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _patch_config(config: dict) -> dict:
    out = dict(config)
    out["architectures"] = ["LongcatCausalLM"]
    out["model_type"] = "longcat"
    if isinstance(out.get("routed_scaling_factor"), int):
        out["routed_scaling_factor"] = float(out["routed_scaling_factor"])
    if "rope_parameters" not in out and "rope_scaling" in out:
        out["rope_parameters"] = out["rope_scaling"]
    # transformers LongCat-Flash RoPE registry has `yarn`, not `deepseek_yarn`.
    rope = out.get("rope_parameters")
    if isinstance(rope, dict) and rope.get("rope_type") == "deepseek_yarn":
        rope = dict(rope)
        rope["rope_type"] = "yarn"
        out["rope_parameters"] = rope
        if "rope_scaling" in out and isinstance(out["rope_scaling"], dict):
            out["rope_scaling"] = dict(out["rope_scaling"])
            out["rope_scaling"]["rope_type"] = "yarn"
    out["auto_map"] = {
        "AutoConfig": "configuration_longcat2_ptq.LongcatConfig",
        "AutoModelForCausalLM": "modeling_longcat2_ptq.LongcatCausalLM",
    }
    out["oe_embed_dim"] = int(out.get("oe_embed_dim", 512))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Original LongCat-2.0 snapshot or model dir",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Writable prepared dir (config + modeling; can share weights via symlink)",
    )
    parser.add_argument(
        "--symlink-weights",
        action="store_true",
        help="Symlink safetensors / index from src instead of copying",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).resolve().parent
    for name in (
        "configuration_longcat2_ptq.py",
        "modeling_longcat2_ptq.py",
        "__init__.py",
    ):
        shutil.copy2(here / name, dst / name)

    config = json.loads((src / "config.json").read_text())
    patched = _patch_config(config)
    (dst / "config.json").write_text(json.dumps(patched, indent=2) + "\n")

    # Tokenizer / misc sidecar files.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenization_llama.py",
        "generation_config.json",
    ):
        src_file = src / name
        if src_file.exists():
            target = dst / name
            if target.exists() or target.is_symlink():
                target.unlink()
            if args.symlink_weights:
                target.symlink_to(src_file)
            else:
                shutil.copy2(src_file, target)

    weight_names = ["model.safetensors.index.json"]
    weight_names += sorted(p.name for p in src.glob("model-*.safetensors"))
    for name in weight_names:
        src_file = src / name
        if not src_file.exists():
            continue
        target = dst / name
        if target.exists() or target.is_symlink():
            target.unlink()
        if args.symlink_weights:
            target.symlink_to(src_file)
        else:
            print(f"copying {name} (large) ...")
            shutil.copy2(src_file, target)

    print(f"Prepared PTQ snapshot at {dst}")
    print("auto_map:", patched["auto_map"])
    print(
        "Next: run ModelOpt hf_ptq with "
        "recipes/longcat2_nvfp4_mlp_only-kv_fp8.yaml"
    )


if __name__ == "__main__":
    main()
