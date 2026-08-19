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

"""PTQ-oriented HF modeling for LongCat-2.0 (ModelOpt calibration / export).

Targets the ModelOpt PTQ venv (transformers 4.57.x), where ``LongcatFlashMoE``
already stores unfused per-expert ``ModuleList`` modules that match Meituan
checkpoint keys ``mlp.experts.{i}.{gate,up,down}_proj``.

Extra LongCat-2.0 pieces layered on Flash:
  * DSA indexer under ``self_attn.0.indexer.*``
  * OE projection linears ``oe_embed_proj{i}``

Giant OE embedding tables and MTP weights are ignored on load.
"""

from __future__ import annotations

from torch import nn
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashForCausalLM,
    LongcatFlashMLA,
    LongcatFlashModel,
    LongcatFlashPreTrainedModel,
    LongcatFlashRMSNorm,
)

try:
    from .configuration_longcat2_ptq import LongcatConfig
except ImportError:  # loaded as a plain file on PYTHONPATH / HF remote code
    from configuration_longcat2_ptq import LongcatConfig


class LongcatDSAIndexer(nn.Module):
    """Module tree matching ``self_attn.0.indexer.*`` checkpoint keys."""

    def __init__(self, config: LongcatConfig):
        super().__init__()
        index_n_heads = int(config.index_n_heads)
        index_head_dim = int(config.index_head_dim)
        q_lora_rank = int(config.q_lora_rank)
        hidden = int(config.hidden_size)
        self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = nn.Linear(hidden, index_head_dim, bias=False)
        self.weights_proj = nn.Linear(hidden, index_n_heads, bias=False)
        self.k_norm = LongcatFlashRMSNorm(index_head_dim)


class LongcatFlashMLAWithIndexer(LongcatFlashMLA):
    """MLA branch that optionally owns a DSA indexer submodule."""

    def __init__(self, config: LongcatConfig, layer_idx: int, *, with_indexer: bool):
        super().__init__(config, layer_idx)
        if with_indexer and getattr(config, "index_n_heads", None):
            self.indexer = LongcatDSAIndexer(config)


class LongcatModel(LongcatFlashModel):
    """Flash backbone + indexer / OE projection modules for LongCat-2.0 keys."""

    def __init__(self, config: LongcatConfig):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn = nn.ModuleList(
                [
                    LongcatFlashMLAWithIndexer(
                        config, layer_idx * 2 + i, with_indexer=(i == 0)
                    )
                    for i in (0, 1)
                ]
            )
        self._init_oe_modules(config)

    def _init_oe_modules(self, config: LongcatConfig) -> None:
        ratio = getattr(config, "oe_vocab_size_ratio", None)
        n = getattr(config, "oe_neighbor_num", None)
        k = getattr(config, "oe_split_num", None)
        if not ratio or not n or not k or int(n) <= 1:
            self.oe_n_grams = 0
            return

        n_grams = (int(n) - 1) * int(k)
        self.oe_n_grams = n_grams
        oe_dim = int(getattr(config, "oe_embed_dim", 512))
        for i in range(n_grams):
            setattr(
                self,
                f"oe_embed_proj{i}",
                nn.Linear(oe_dim, config.hidden_size, bias=False),
            )


class LongcatCausalLM(LongcatFlashForCausalLM):
    """Entry class matching LongCat-2.0 ``architectures: [LongcatCausalLM]``."""

    config_class = LongcatConfig
    _keys_to_ignore_on_load_unexpected = [
        r"model\.mtp\..*",
        r"model\.oe_embed_tokens\d+\.weight",
    ]

    def __init__(self, config: LongcatConfig):
        LongcatFlashPreTrainedModel.__init__(self, config)
        self.model = LongcatModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
