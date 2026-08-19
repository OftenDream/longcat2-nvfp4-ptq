#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
"""Print LongCat-2.0 weight-key families vs ModelOpt NVFP4 mlp_only globs."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def _norm(key: str) -> str:
    key = re.sub(r"\.layers\.\d+\.", ".layers.*.", key)
    key = re.sub(r"\.experts\.\d+\.", ".experts.*.", key)
    key = re.sub(r"\.mlps\.\d+\.", ".mlps.*.", key)
    key = re.sub(r"\.self_attn\.\d+\.", ".self_attn.*.", key)
    return key


def _classify(key: str) -> str:
    if ".mlp.experts." in key and key.endswith(".weight"):
        return "QUANT_target_moe_expert"
    if ".mlps." in key and key.endswith((".weight",)):
        return "QUANT_target_dense_mlp"
    if ".mlp.router." in key:
        return "EXCLUDE_router"
    if ".self_attn." in key and ".indexer." in key:
        return "EXCLUDE_dsa_indexer"
    if ".self_attn." in key:
        return "EXCLUDE_attention"
    if "oe_embed" in key:
        return "EXCLUDE_oe"
    if key.endswith("lm_head.weight") or ".embed_tokens." in key:
        return "EXCLUDE_embed_or_lm_head"
    if ".mtp." in key:
        return "EXCLUDE_mtp"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to LongCat-2.0 snapshot (contains model.safetensors.index.json)",
    )
    args = parser.parse_args()
    index_path = args.snapshot / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    keys = list(weight_map)

    by_class: collections.Counter[str] = collections.Counter()
    patterns: collections.Counter[str] = collections.Counter()
    for key in keys:
        by_class[_classify(key)] += 1
        patterns[_norm(key)] += 1

    print(f"snapshot: {args.snapshot}")
    print(f"num_tensors: {len(keys)}")
    print("\n== class counts ==")
    for name, count in sorted(by_class.items(), key=lambda x: (-x[1], x[0])):
        print(f"{count:6d}  {name}")

    print("\n== top patterns ==")
    for pattern, count in patterns.most_common(40):
        print(f"{count:6d}  {pattern}")

    quant = sum(v for k, v in by_class.items() if k.startswith("QUANT_"))
    excl = sum(v for k, v in by_class.items() if k.startswith("EXCLUDE_"))
    print(f"\nQUANT_* tensors: {quant}")
    print(f"EXCLUDE_* tensors: {excl}")
    print(
        "ModelOpt globs '*mlps*' / '*mlp*' / '*.experts.*' cover QUANT_* ; "
        "recipe disables EXCLUDE_* families."
    )


if __name__ == "__main__":
    main()
