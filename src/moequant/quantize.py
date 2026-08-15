"""Quantization policies, expressed as torchao FqnToConfig dictionaries.

We do not implement quantization arithmetic. ``torchao`` already provides selective,
per-module quantization at 1-8 bits, and ``transformers`` wires it in through
``TorchAoConfig``. What lives here is only the *policy*: which modules get quantized,
which are deliberately left alone, and why.

The three Part 1 policies differ in exactly one respect - whether the routers are in the
skip list - so any difference we measure between them is attributable to the router and
nothing else.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .registry import ModelSpec

# Modules every quantized policy leaves in high precision. Standard PTQ practice keeps
# the embedding and output head in full precision; including them would add a confound
# that has nothing to do with routing.
ALWAYS_SKIP = (r".*embed_tokens", r"lm_head")

POLICIES = ("gold", "uniform", "mixed", "placebo", "attention")

SUPPORTED_BITS = (8, 6, 5, 4, 3, 2)


@dataclass(frozen=True)
class QuantPolicy:
    """A fully resolved policy, ready to be turned into a TorchAoConfig."""

    name: str
    bits: int | None
    skip_patterns: tuple[str, ...]
    skip_exact: tuple[str, ...] = ()

    @property
    def is_gold(self) -> bool:
        return self.bits is None

    def describe(self) -> str:
        if self.is_gold:
            return "gold: no quantization, everything stays in the load dtype"
        skips = list(self.skip_patterns) + list(self.skip_exact)
        return f"{self.name}: INT{self.bits} everywhere except {len(skips)} skipped module patterns"


def _weight_dtype(bits: int) -> torch.dtype:
    """Map a bit-width onto the torch sub-byte dtype torchao expects."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits={bits} not supported; choose from {SUPPORTED_BITS}")
    name = f"int{bits}"
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(
            f"torch.{name} is unavailable in torch {torch.__version__}. "
            "Sub-byte dtypes need torch>=2.6; upgrade or drop this bit-width."
        )
    return dtype


def build_policy(
    name: str,
    spec: ModelSpec,
    bits: int | None,
    placebo_modules: tuple[str, ...] = (),
) -> QuantPolicy:
    """Resolve a policy name into the concrete set of modules to leave alone."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}; choose from {POLICIES}")

    if name == "gold":
        return QuantPolicy(name="gold", bits=None, skip_patterns=())

    if bits is None:
        raise ValueError(f"Policy {name!r} requires a bit-width")

    skip: tuple[str, ...] = ALWAYS_SKIP
    exact: tuple[str, ...] = ()

    if name == "uniform":
        pass  # routers included in quantization; this is the negative baseline
    elif name == "mixed":
        skip = skip + spec.protect_patterns
    elif name == "attention":
        # Match the attention block itself as well as its projections, so this works
        # whether the architecture nests q/k/v under `self_attn` or fuses them into it.
        skip = skip + (r".*\.self_attn(\..*)?",)
    elif name == "placebo":
        if not placebo_modules:
            raise ValueError(
                "placebo policy needs an explicit module list; "
                "call sample_placebo_modules() first"
            )
        exact = placebo_modules

    return QuantPolicy(name=name, bits=bits, skip_patterns=skip, skip_exact=exact)


def resolve_skips(policy: QuantPolicy, module_fqns: Sequence[str] = ()) -> tuple[str, ...]:
    """Materialise the policy's regex skips into the exact module FQNs they match.

    This indirection is not cosmetic. ``transformers`` does not hand the config to
    torchao's ``quantize_``; its ``TorchAoHfQuantizer`` quantizes each parameter as it is
    loaded and resolves the mapping with a plain dict lookup on the module FQN. A ``re:``
    key therefore never matches, and every module falls through to ``_default`` while the
    load reports success. torchao's own ``quantize_`` *does* honour ``re:``, which is why
    this only surfaces on a real checkpoint. Exact FQNs are understood by both paths.
    """
    skips = list(policy.skip_exact)
    if policy.skip_patterns:
        if not module_fqns:
            raise ValueError(
                f"Policy {policy.name!r} skips modules by pattern, so it needs the model's "
                "module FQNs to expand them. Pass module_fqns=... (enumerate them from a "
                "meta-device skeleton so no weights are materialised)."
            )
        compiled = [re.compile(p) for p in policy.skip_patterns]
        skips += [fqn for fqn in module_fqns if any(c.fullmatch(fqn) for c in compiled)]
    return tuple(dict.fromkeys(skips))


def to_torchao_config(policy: QuantPolicy, module_fqns: Sequence[str] = ()):
    """Turn a policy into a ``transformers`` quantization_config, or None for gold."""
    if policy.is_gold:
        return None

    from transformers import TorchAoConfig

    try:
        from torchao.quantization import FqnToConfig
    except ImportError:  # torchao < 0.14 called it ModuleFqnToConfig
        from torchao.quantization import ModuleFqnToConfig as FqnToConfig

    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    # One config object for every bit-width keeps the sweep on a single code path, so a
    # trend across bits reflects precision alone and not a change of kernel.
    base = IntxWeightOnlyConfig(
        weight_dtype=_weight_dtype(policy.bits),
        granularity=PerAxis(0),
    )

    mapping: OrderedDict[str, object | None] = OrderedDict()
    for fqn in resolve_skips(policy, module_fqns):
        mapping[fqn] = None
    mapping["_default"] = base

    return TorchAoConfig(quant_type=FqnToConfig(mapping))


def quantizable_modules(model: nn.Module, spec: ModelSpec) -> list[tuple[str, int]]:
    """Every nn.Linear the `_default` config would touch, with its parameter count.

    Used to build the parameter-matched placebo and to predict what a policy should do
    before we load anything expensive.
    """
    always_skip = [re.compile(p) for p in ALWAYS_SKIP]
    out: list[tuple[str, int]] = []
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(p.fullmatch(fqn) for p in always_skip):
            continue
        out.append((fqn, sum(p.numel() for p in module.parameters(recurse=False))))
    return out


def sample_placebo_modules(
    model: nn.Module,
    spec: ModelSpec,
    seed: int = 0,
    tolerance: float = 0.25,
) -> tuple[str, ...]:
    """Pick a random set of modules whose parameter count matches the routers'.

    This is the control that makes a positive Part 1 result interpretable. If protecting
    a random 0.02% of the model helps as much as protecting the routers, then the effect
    was about keeping *some* weights in high precision, not about routers being special.

    Routers themselves are excluded from the candidate pool, and we greedily accumulate
    small modules until within `tolerance` of the router budget.
    """
    import random

    router = spec.router_regex()
    everything = quantizable_modules(model, spec)
    candidates = [(fqn, n) for fqn, n in everything if not router.fullmatch(fqn)]
    budget = sum(n for fqn, n in everything if router.fullmatch(fqn))
    if budget == 0:
        raise RuntimeError("Router budget is zero; the registry pattern matched nothing.")

    rng = random.Random(seed)
    # Prefer candidates no larger than the whole budget, else we overshoot on the first pick.
    pool = [c for c in candidates if c[1] <= budget] or candidates
    rng.shuffle(pool)

    lower, upper = budget * (1 - tolerance), budget * (1 + tolerance)
    chosen: list[str] = []
    accumulated = 0
    for fqn, n in pool:
        if accumulated >= lower:
            break
        # Skip anything that would carry us past the upper bound rather than adding it
        # and failing afterwards; a later, smaller module may still land us in range.
        if accumulated + n > upper:
            continue
        chosen.append(fqn)
        accumulated += n

    achieved = accumulated / budget
    if not 1 - tolerance <= achieved <= 1 + tolerance:
        raise RuntimeError(
            f"Could not parameter-match the placebo: got {achieved:.2f}x the router budget. "
            "Widen `tolerance` or pick a finer-grained candidate pool."
        )
    return tuple(chosen)
