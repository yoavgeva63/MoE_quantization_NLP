"""A tiny synthetic MoE that mimics the module naming of the real architectures.

Lets the whole pipeline be tested without a GPU or a multi-gigabyte download. The naming
matters: `mlp.gate` must be matched by the router pattern while `mlp.experts.N.gate_proj`
and `mlp.shared_expert_gate` must not, and that distinction is easy to get wrong.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


class TinyConfig:
    def __init__(self, num_experts: int = 8, top_k: int = 2, hidden: int = 16, layers: int = 3):
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        self.hidden_size = hidden
        self.num_hidden_layers = layers
        self.norm_topk_prob = True


class TinyExpert(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Named to collide with the router pattern if it is written carelessly.
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)))


class TinyMoeBlock(nn.Module):
    def __init__(self, config: TinyConfig, shared_expert: bool = False):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyExpert(config.hidden_size) for _ in range(config.num_experts)
        )
        if shared_expert:
            self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = self.gate(flat)
        weights = torch.softmax(logits, dim=-1)
        top = weights.topk(self.top_k, dim=-1)
        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = top.indices[:, slot]
            w = top.values[:, slot : slot + 1]
            for expert_id in idx.unique():
                sel = idx == expert_id
                out[sel] += w[sel] * self.experts[int(expert_id)](flat[sel])
        return out.reshape(x.shape)


class TinyLayer(nn.Module):
    def __init__(self, config: TinyConfig, shared_expert: bool):
        super().__init__()
        self.self_attn = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp = TinyMoeBlock(config, shared_expert)

    def forward(self, x):
        return self.mlp(x + self.self_attn(x))


class TinyInner(nn.Module):
    def __init__(self, config: TinyConfig, shared_expert: bool):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyLayer(config, shared_expert) for _ in range(config.num_hidden_layers)
        )


class TinyMoE(nn.Module):
    """Shaped like `model.layers.{i}.mlp.gate`, matching the real checkpoints."""

    def __init__(self, config: TinyConfig, shared_expert: bool = False):
        super().__init__()
        self.config = config
        self.model = TinyInner(config, shared_expert)
        self.lm_head = nn.Linear(config.hidden_size, 64, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


@pytest.fixture
def tiny_config() -> TinyConfig:
    return TinyConfig()


@pytest.fixture
def tiny_model(tiny_config) -> TinyMoE:
    torch.manual_seed(0)
    return TinyMoE(tiny_config)


@pytest.fixture
def tiny_model_shared(tiny_config) -> TinyMoE:
    torch.manual_seed(0)
    return TinyMoE(tiny_config, shared_expert=True)


@pytest.fixture
def tiny_spec():
    from moequant.registry import ModelSpec

    return ModelSpec(
        key="tiny",
        model_id="tiny/synthetic",
        router_pattern=r".*\.mlp\.gate",
        protect_patterns=(r".*\.mlp\.gate",),
    )


@pytest.fixture
def tiny_spec_shared():
    from moequant.registry import ModelSpec

    return ModelSpec(
        key="tiny-shared",
        model_id="tiny/synthetic-shared",
        router_pattern=r".*\.mlp\.gate",
        protect_patterns=(r".*\.mlp\.gate", r".*\.shared_expert_gate"),
    )
