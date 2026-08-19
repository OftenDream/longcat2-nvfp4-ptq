# Copyright (c) 2026 LightSeek Foundation
"""Fast regression tests for LongCat ModelOpt export patches.

These do **not** load LongCat-2.0. They catch the classes of bugs that previously
burned a full calib day:

* missing NVFP4 weight ``_amax`` (Kimi / PR#785 path)
* ``set_expert_quantizer_amax`` + meta tensors (our unsafe pre-pass)
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import modelopt.torch.quantization as mtq


def _make_nvfp4_linear(in_f: int = 32, out_f: int = 16, *, with_amax: bool) -> nn.Module:
    """Quantize a tiny Linear with the real ModelOpt NVFP4 path (has calibrator)."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(in_f, out_f, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = Tiny()
    cfg = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "enable": True,
            },
            "*input_quantizer": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "enable": True,
            },
            "default": {"enable": False},
        },
        "algorithm": "max",
    }

    def loop(m):
        m(torch.randn(4, in_f))

    if with_amax:
        mtq.quantize(model, cfg, forward_loop=loop)
    else:
        # Insert quantizers but clear weight amax to simulate a dead expert.
        mtq.quantize(model, cfg, forward_loop=loop)
        wq = model.fc.weight_quantizer
        if hasattr(wq, "_amax"):
            delattr(wq, "_amax")
    return model.fc


def test_pr785_lazy_weight_amax():
    from modelopt.torch.export import quant_utils as qu
    from patch_modelopt_export import apply_longcat_modelopt_export_patches

    apply_longcat_modelopt_export_patches()
    m = _make_nvfp4_linear(with_amax=False)
    assert m.weight_quantizer.amax is None

    qu._ensure_weight_quantizer_calibrated(m.weight_quantizer, m.weight, "Tiny.weight")
    assert m.weight_quantizer.amax is not None
    assert float(m.weight_quantizer.amax.float().max()) > 0


def test_ensure_skips_meta_weight_without_crashing():
    from modelopt.torch.export import quant_utils as qu
    from patch_modelopt_export import apply_longcat_modelopt_export_patches

    apply_longcat_modelopt_export_patches()
    m = _make_nvfp4_linear(with_amax=False)
    meta_w = torch.empty_like(m.weight, device="meta")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qu._ensure_weight_quantizer_calibrated(m.weight_quantizer, meta_w, "Tiny.meta")
    assert any("meta tensor" in str(w.message) for w in caught)
    assert m.weight_quantizer.amax is None


def test_set_expert_quantizer_amax_meta_weight_is_unsafe():
    """Document why a pre-remove_hook weight amax pass is invalid under offload."""
    from modelopt.torch.export.layer_utils import set_expert_quantizer_amax

    m = _make_nvfp4_linear(with_amax=False)
    m.weight = nn.Parameter(torch.empty_like(m.weight, device="meta"), requires_grad=False)
    try:
        set_expert_quantizer_amax(modules=[m], quantizer_attrs=["weight_quantizer"])
    except RuntimeError as e:
        assert "meta" in str(e).lower()
        return
    # Some ModelOpt versions may avoid .item(); that's fine — the point is we
    # must not rely on this path before hooks are removed.


def test_input_only_amax_fallback_does_not_need_weights():
    from modelopt.torch.export.layer_utils import set_expert_quantizer_amax

    calibrated = _make_nvfp4_linear(with_amax=True)
    dead = _make_nvfp4_linear(with_amax=False)
    dead.weight = nn.Parameter(torch.empty_like(dead.weight, device="meta"), requires_grad=False)
    set_expert_quantizer_amax(
        modules=[calibrated, dead],
        quantizer_attrs=["input_quantizer"],
    )
    assert dead.input_quantizer.amax is not None


if __name__ == "__main__":
    test_pr785_lazy_weight_amax()
    print("PASS test_pr785_lazy_weight_amax")
    test_ensure_skips_meta_weight_without_crashing()
    print("PASS test_ensure_skips_meta_weight_without_crashing")
    test_set_expert_quantizer_amax_meta_weight_is_unsafe()
    print("PASS test_set_expert_quantizer_amax_meta_weight_is_unsafe")
    test_input_only_amax_fallback_does_not_need_weights()
    print("PASS test_input_only_amax_fallback_does_not_need_weights")
    print("All fast export-patch tests passed.")
