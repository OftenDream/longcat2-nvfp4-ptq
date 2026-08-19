# Copyright (c) 2026 LightSeek Foundation
#
# Monkeypatches for NVIDIA ModelOpt 0.41.x export of LongCat-2.0 MoE NVFP4.
#
# Why Kimi-K2.5-NVFP4 does not hit our previous failures
# -----------------------------------------------------
# Official Kimi NVFP4 export is unblocked by Model-Optimizer PR #785: during
# NVFP4 export, if a weight quantizer has no ``_amax`` (dead MoE expert),
# lazily calibrate amax from the weight tensor. That runs inside
# ``get_weight_scaling_factor*`` / ``_export_quantized_weight`` *after*
# accelerate hooks are removed and weights are real device tensors.
#
# ModelOpt 0.41.0 predates PR #785, so we backport that helper here.
#
# We intentionally do *not* call ``set_expert_quantizer_amax`` as a LongCat
# pre-pass before ``_export_hf_checkpoint``. That helper falls back to
# ``weight.abs().max()`` and then ``Tensor.item()``; with
# ``--use_seq_device_map`` many expert weights are still meta at that point
# ("offloaded to the cpu"), which raises:
#   RuntimeError: Tensor.item() cannot be called on meta tensors
# Upstream only uses ``set_expert_quantizer_amax`` for recognized MoE types
# and only for ``input_quantizer``; weight amax is handled by PR #785.
# LongCat also mixes ``nn.Identity`` zero-experts, so flipping ``is_moe`` on
# is unsafe for the upstream list-comprehension.
#
# Import this module once before calling ``hf_ptq`` / ``export_hf_checkpoint``.

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


def apply_longcat_modelopt_export_patches() -> None:
    """Apply Kimi-style NVFP4 amax export patches (idempotent)."""
    _patch_ensure_weight_quantizer_calibrated()
    _patch_export_quantized_weight_guard()
    _patch_longcat_input_amax_after_hooks()


def _is_meta_tensor(t: torch.Tensor | None) -> bool:
    return t is not None and bool(getattr(t, "is_meta", False))


def _iter_real_expert_linears(experts, linear_name: str) -> list[nn.Module]:
    """Skip ``nn.Identity`` zero-experts that have no proj modules."""
    out: list[nn.Module] = []
    for expert in experts:
        if hasattr(expert, linear_name):
            out.append(getattr(expert, linear_name))
    return out


def _is_longcat_moe(module: nn.Module) -> bool:
    name = type(module).__name__.lower()
    return "longcat" in name and "moe" in name


def _module_has_materialized_weight(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and not _is_meta_tensor(weight)


def _patch_ensure_weight_quantizer_calibrated() -> None:
    """Port of Model-Optimizer PR #785 for NVFP4 export on ModelOpt 0.41."""
    from modelopt.torch.export import quant_utils as qu
    from modelopt.torch.quantization.model_calib import (
        enable_stats_collection,
        finish_stats_collection,
    )
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    from modelopt.torch.quantization.utils import quantizer_attr_names

    if getattr(qu, "_longcat2_amax_patched", False):
        return

    def _ensure_weight_quantizer_calibrated(
        weight_quantizer: TensorQuantizer,
        weight: torch.Tensor,
        module_name: str = "",
    ) -> None:
        if hasattr(weight_quantizer, "_amax") and weight_quantizer._amax is not None:
            return
        if _is_meta_tensor(weight):
            # Same situation as early LongCat export pre-pass: cannot calibrate
            # from meta storage. Caller should run after accelerate hooks remove
            # / weights are materialized (Kimi / PR#785 timing).
            warnings.warn(
                f"Weight quantizer{f' for {module_name}' if module_name else ''} "
                "missing amax but weight is still a meta tensor; skip lazy "
                "calibration (weight not materialized yet)."
            )
            return
        warnings.warn(
            f"Weight quantizer{f' for {module_name}' if module_name else ''} "
            "was not calibrated. Computing amax from weights. This may occur "
            "if some MoE experts were not activated during calibration; "
            "consider increasing --calib_size."
        )
        # Prefer the official PR#785 stats-collection path when a calibrator
        # exists (normal after mtq.quantize). Fall back to direct amax assign
        # for partially constructed quantizers.
        if hasattr(weight_quantizer, "_calibrator"):
            weight_quantizer.reset_amax()
            enable_stats_collection(weight_quantizer)
            weight_quantizer(weight)
            finish_stats_collection(weight_quantizer)
        else:
            weight_quantizer.amax = torch.amax(torch.abs(weight.detach())).float()

    _orig_get_wsf = qu.get_weight_scaling_factor
    _orig_get_wsf2 = qu.get_weight_scaling_factor_2

    nvfp4_formats = {
        qu.QUANTIZATION_NVFP4,
        qu.QUANTIZATION_NVFP4_AWQ,
        getattr(qu, "QUANTIZATION_NVFP4_SVDQUANT", "nvfp4_svdquant"),
        qu.QUANTIZATION_W4A8_NVFP4_FP8,
    }

    def get_weight_scaling_factor(module: nn.Module, weight_name: str = "weight") -> torch.Tensor:
        weight = getattr(module, weight_name)
        weight_quantizer = getattr(
            module, quantizer_attr_names(weight_name).weight_quantizer, None
        )
        if weight_quantizer is None:
            return None
        if isinstance(weight_quantizer, SequentialQuantizer):
            return _orig_get_wsf(module, weight_name)

        if qu.get_quantization_format(module) in nvfp4_formats:
            module_name = f"{type(module).__name__}.{weight_name}"
            _ensure_weight_quantizer_calibrated(weight_quantizer, weight, module_name)
        return _orig_get_wsf(module, weight_name)

    def get_weight_scaling_factor_2(module: nn.Module, weight_name: str = "weight") -> torch.Tensor:
        weight_quantizer = getattr(
            module, quantizer_attr_names(weight_name).weight_quantizer, None
        )
        if weight_quantizer is None:
            return None
        if qu.get_quantization_format(module) in nvfp4_formats and not isinstance(
            weight_quantizer, SequentialQuantizer
        ):
            weight = getattr(module, weight_name)
            module_name = f"{type(module).__name__}.{weight_name}"
            _ensure_weight_quantizer_calibrated(weight_quantizer, weight, module_name)
        return _orig_get_wsf2(module, weight_name)

    qu._ensure_weight_quantizer_calibrated = _ensure_weight_quantizer_calibrated
    qu.get_weight_scaling_factor = get_weight_scaling_factor
    qu.get_weight_scaling_factor_2 = get_weight_scaling_factor_2

    try:
        import modelopt.torch.export.unified_export_hf as ue

        ue.get_weight_scaling_factor = get_weight_scaling_factor
        ue.get_weight_scaling_factor_2 = get_weight_scaling_factor_2
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not rebind unified_export_hf weight scale helpers: {exc}")

    _ = NVFP4QTensor
    qu._longcat2_amax_patched = True


def _patch_export_quantized_weight_guard() -> None:
    """Ensure weight amax exists before ``_export_quantized_weight`` touches ``_amax``."""
    import modelopt.torch.export.unified_export_hf as ue
    from modelopt.torch.export import quant_utils as qu
    from modelopt.torch.quantization.utils import quantizer_attr_names

    if getattr(ue, "_longcat2_export_weight_guard_patched", False):
        return

    _orig = ue._export_quantized_weight
    ensure = getattr(qu, "_ensure_weight_quantizer_calibrated", None)
    if ensure is None:
        return

    def _export_quantized_weight(sub_module, dtype, weight_name: str = "weight"):
        weight_quantizer = getattr(
            sub_module, quantizer_attr_names(weight_name).weight_quantizer, None
        )
        weight = getattr(sub_module, weight_name, None)
        if weight_quantizer is not None and isinstance(weight, torch.Tensor):
            ensure(
                weight_quantizer,
                weight,
                f"{type(sub_module).__name__}.{weight_name}",
            )
        return _orig(sub_module, dtype, weight_name)

    ue._export_quantized_weight = _export_quantized_weight
    ue._longcat2_export_weight_guard_patched = True


def _patch_longcat_input_amax_after_hooks() -> None:
    """Fill missing *input* amax for LongCat experts after accelerate hooks are gone.

    Matches upstream MoE export timing (post ``remove_hook_from_module``), and
    only touches modules whose weights are materialized. Weight amax remains
    PR #785's responsibility.
    """
    import modelopt.torch.export.unified_export_hf as ue
    from modelopt.torch.export.layer_utils import set_expert_quantizer_amax

    if getattr(ue, "_longcat2_input_amax_patched", False):
        return

    _orig_remove = None
    try:
        from accelerate.hooks import remove_hook_from_module as _orig_remove
    except ImportError:
        _orig_remove = None

    if _orig_remove is None:
        return

    linear_names = ["gate_proj", "up_proj", "down_proj"]

    def remove_hook_from_module(module, recurse=False):
        result = _orig_remove(module, recurse=recurse)
        # Only run once on the top-level recurse=True call used by export.
        if not recurse or not isinstance(module, nn.Module):
            return result
        if getattr(module, "_longcat2_input_amax_done", False):
            return result

        filled = 0
        skipped_meta = 0
        for _, sub_module in module.named_modules():
            if not _is_longcat_moe(sub_module) or not hasattr(sub_module, "experts"):
                continue
            if not isinstance(sub_module.experts, (list, nn.ModuleList)):
                continue
            for linear_name in linear_names:
                modules = [
                    m
                    for m in _iter_real_expert_linears(sub_module.experts, linear_name)
                    if _module_has_materialized_weight(m)
                ]
                skipped_meta += len(
                    _iter_real_expert_linears(sub_module.experts, linear_name)
                ) - len(modules)
                if not modules:
                    continue
                # Upstream MoE path only reconciles input_quantizer here.
                set_expert_quantizer_amax(
                    modules=modules,
                    quantizer_attrs=["input_quantizer"],
                )
                filled += len(modules)

        module._longcat2_input_amax_done = True
        if filled or skipped_meta:
            warnings.warn(
                f"LongCat MoE post-hook input amax pass: filled via "
                f"{filled} materialized expert linears; skipped {skipped_meta} "
                f"meta/unmaterialized modules."
            )
        return result

    # Rebind where export imports it from (local import inside function).
    # Patch accelerate.hooks so the import inside _export_hf_checkpoint sees it.
    import accelerate.hooks as hooks

    hooks.remove_hook_from_module = remove_hook_from_module
    ue._longcat2_input_amax_patched = True


if __name__ == "__main__":
    apply_longcat_modelopt_export_patches()
    print("LongCat ModelOpt export patches applied.")
