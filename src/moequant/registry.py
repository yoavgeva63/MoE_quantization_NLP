"""Per-architecture adapters.

Every MoE family names and shapes its router differently, and some do not even use
``nn.Linear``. All of that lives here so the rest of the package can stay
architecture-agnostic: give it a :class:`ModelSpec` and it knows where the routers are,
how many experts they choose between, and how to read logits out of them.
"""

from __future__ import annotations

import functools
import re
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn


@functools.lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)

# How the router module reports its decision.
#   "logits"   -> the module is an nn.Linear and its output *is* the logit tensor
#   "recompute"-> the module returns something else (DeepSeek's MoEGate returns
#                 (topk_idx, topk_weight, aux_loss)); logits must be recomputed as
#                 input @ weight.T from the captured router input
RouterOutputKind = str


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs to know about one MoE architecture."""

    key: str
    model_id: str

    # Regex (full-match semantics) selecting the routing gate modules. Used both for
    # hooking and, via `protect_patterns`, for the quantization policy.
    router_pattern: str

    # Every gate-like module to keep in high precision under the `mixed` policy.
    # Usually the routers, plus any auxiliary gate the architecture carries.
    protect_patterns: tuple[str, ...]

    # Config attribute names, in preference order, for expert count and top-k.
    num_experts_attrs: tuple[str, ...] = ("num_experts", "n_routed_experts")
    top_k_attrs: tuple[str, ...] = ("num_experts_per_tok", "top_k")

    router_output_kind: RouterOutputKind = "logits"
    trust_remote_code: bool = False
    notes: str = ""

    def router_regex(self) -> re.Pattern[str]:
        return _compiled(self.router_pattern)

    def is_router(self, fqn: str) -> bool:
        return self.router_regex().fullmatch(fqn) is not None


MODEL_SPECS: dict[str, ModelSpec] = {
    "olmoe": ModelSpec(
        key="olmoe",
        model_id="allenai/OLMoE-1B-7B-0924",
        router_pattern=r".*\.mlp\.gate",
        protect_patterns=(r".*\.mlp\.gate",),
        notes="16 layers, 64 experts, top-8. Clean nn.Linear router, no shared expert.",
    ),
    "qwen": ModelSpec(
        key="qwen",
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
        router_pattern=r".*\.mlp\.gate",
        # shared_expert_gate is a 1-output sigmoid gate, not a router. It is not hooked
        # for routing metrics, but it is gate-like so `mixed` protects it too.
        protect_patterns=(r".*\.mlp\.gate", r".*\.shared_expert_gate"),
        notes="24 layers, 60 experts, top-4, plus a shared expert with its own gate.",
    ),
    "deepseek": ModelSpec(
        key="deepseek",
        model_id="deepseek-ai/deepseek-moe-16b-base",
        router_pattern=r".*\.mlp\.gate",
        protect_patterns=(r".*\.mlp\.gate",),
        num_experts_attrs=("n_routed_experts",),
        # MoEGate holds a bare nn.Parameter and returns (topk_idx, topk_weight, aux_loss),
        # so its output carries no logits and an nn.Linear walker will not see its weight.
        router_output_kind="recompute",
        trust_remote_code=True,
        notes="STRETCH ONLY. Layer 0 is dense; MoEGate is not an nn.Linear.",
    ),
}


def get_spec(key: str) -> ModelSpec:
    if key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key {key!r}. Known: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[key]


def find_routers(model: nn.Module, spec: ModelSpec) -> OrderedDict[str, nn.Module]:
    """Return router modules keyed by fully qualified name, in forward order."""
    pattern = spec.router_regex()
    found: OrderedDict[str, nn.Module] = OrderedDict()
    for fqn, module in model.named_modules():
        if pattern.fullmatch(fqn):
            found[fqn] = module
    if not found:
        raise RuntimeError(
            f"No router modules matched {spec.router_pattern!r} in {spec.model_id}. "
            "Run scripts/inspect_model.py to dump the real module tree."
        )
    return found


def layer_index(fqn: str) -> int:
    """Extract the transformer layer index from a module FQN.

    Falls back to -1 so callers can sort deterministically even on odd names.
    """
    match = re.search(r"layers\.(\d+)\.", fqn)
    return int(match.group(1)) if match else -1


def router_weight(module: nn.Module) -> torch.Tensor:
    """The [num_experts, hidden] routing matrix, whether or not it lives in an nn.Linear."""
    weight = getattr(module, "weight", None)
    if weight is None:
        raise AttributeError(f"Router module {type(module).__name__} has no .weight")
    return weight


def _first_attr(config: object, names: tuple[str, ...], what: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not read {what} from config; tried {names}")


def resolve_topology(model: nn.Module, spec: ModelSpec) -> dict:
    """Read expert count and top-k from the loaded model, cross-checking against shapes.

    Reading from the config rather than hardcoding means a checkpoint revision cannot
    silently desync us; cross-checking against the router's actual output dimension
    means a config typo cannot either.
    """
    config = model.config
    top_k = _first_attr(config, spec.top_k_attrs, "top_k")
    num_experts = _first_attr(config, spec.num_experts_attrs, "num_experts")

    routers = find_routers(model, spec)
    weight_shapes = {tuple(router_weight(m).shape) for m in routers.values()}
    if len(weight_shapes) != 1:
        raise RuntimeError(f"Routers have inconsistent shapes: {weight_shapes}")
    out_features, hidden = next(iter(weight_shapes))

    if out_features != num_experts:
        raise RuntimeError(
            f"Config says {num_experts} experts but router weight is {out_features}x{hidden}. "
            "The registry pattern is probably matching the wrong module."
        )
    if not 0 < top_k <= num_experts:
        raise RuntimeError(f"Nonsensical top_k={top_k} for {num_experts} experts")

    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden,
        "num_router_layers": len(routers),
        "router_fqns": list(routers),
        "norm_topk_prob": bool(getattr(config, "norm_topk_prob", False)),
    }


def parameter_census(model: nn.Module, spec: ModelSpec) -> dict:
    """Count total, router, and protected parameters.

    The paper claims routers are a "microscopic fraction" of the model. This is where
    that number comes from, rather than from an estimate.
    """
    protect = [re.compile(p) for p in spec.protect_patterns]
    router = spec.router_regex()

    total = router_params = protected_params = 0
    for fqn, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own == 0:
            continue
        total += own
        if router.fullmatch(fqn):
            router_params += own
        if any(p.fullmatch(fqn) for p in protect):
            protected_params += own

    return {
        "total_params": total,
        "router_params": router_params,
        "protected_params": protected_params,
        "router_fraction": router_params / total if total else 0.0,
        "protected_fraction": protected_params / total if total else 0.0,
    }
