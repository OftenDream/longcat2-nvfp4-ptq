# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""HF config wrapper so ModelOpt can load meituan-longcat/LongCat-2.0."""

from __future__ import annotations

from transformers.models.longcat_flash.configuration_longcat_flash import (
    LongcatFlashConfig,
)


class LongcatConfig(LongcatFlashConfig):
    """LongCat-2.0 config (Flash backbone + OE / DSA fields)."""

    model_type = "longcat"

    def __init__(
        self,
        oe_vocab_size_ratio: float | None = None,
        oe_neighbor_num: int | None = None,
        oe_split_num: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        index_k_norm_type: str | None = "rms",
        index_local_tokens: int | None = None,
        index_init_tokens: int | None = None,
        mtp_num_layers: int | None = None,
        **kwargs,
    ):
        # HF LongCat-2.0 uses rope_scaling; transformers Flash expects rope_parameters.
        if "rope_parameters" not in kwargs and "rope_scaling" in kwargs:
            kwargs["rope_parameters"] = kwargs.pop("rope_scaling")
        if "routed_scaling_factor" in kwargs and isinstance(
            kwargs["routed_scaling_factor"], int
        ):
            kwargs["routed_scaling_factor"] = float(kwargs["routed_scaling_factor"])
        rope = kwargs.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("rope_type") == "deepseek_yarn":
            rope = dict(rope)
            rope["rope_type"] = "yarn"
            kwargs["rope_parameters"] = rope

        super().__init__(**kwargs)
        self.oe_vocab_size_ratio = oe_vocab_size_ratio
        self.oe_neighbor_num = oe_neighbor_num
        self.oe_split_num = oe_split_num
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_k_norm_type = index_k_norm_type
        self.index_local_tokens = index_local_tokens
        self.index_init_tokens = index_init_tokens
        self.mtp_num_layers = mtp_num_layers


__all__ = ["LongcatConfig"]
