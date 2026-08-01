"""Post-load verification: did the quantization do what we asked?

Every failure mode that worries us in this project is silent. A library that skips a
module it does not recognise still returns a working model and plausible perplexity, and
if the routers were never quantized in the first place then `uniform` and `mixed` are the
same run reported under two names.

These checks run after every load and their output goes into the results JSON, so the
paper can state what was quantized rather than what we intended to quantize.
"""

from __future__ import annotations

import re

import torch
from torch import nn

from .quantize import QuantPolicy
from .registry import ModelSpec, find_routers


def _unwrap(weight: torch.Tensor) -> torch.Tensor:
    return weight.data if isinstance(weight, nn.Parameter) else weight


def weight_kind(weight: torch.Tensor) -> str:
    """Name of the concrete tensor type backing a weight.

    Plain weights report "Tensor"; torchao replaces them with a tensor subclass whose
    name identifies the quantization scheme.
    """
    return type(_unwrap(weight)).__name__


def is_quantized(weight: torch.Tensor) -> bool:
    return weight_kind(weight) != "Tensor"


def dequantize(weight: torch.Tensor) -> torch.Tensor:
    """Materialise a weight as a dense float tensor, whatever backs it."""
    raw = _unwrap(weight)
    for attempt in ("dequantize", "to_dense"):
        method = getattr(raw, attempt, None)
        if callable(method):
            try:
                return method().to(torch.float32)
            except Exception:  # noqa: BLE001 - fall through to the generic path
                pass
    return raw.to(torch.float32)


def effective_levels(weight: torch.Tensor, max_rows: int = 8) -> int:
    """Largest number of distinct values found in any single output channel.

    This is how we confirm a bit-width was really applied without needing the original
    weights for comparison. Under per-axis integer quantization each output channel is
    `scale * q` for integer `q`, so an N-bit weight can show at most 2**N distinct values
    per row. A row with thousands of distinct values was never quantized.
    """
    dense = dequantize(weight)
    if dense.ndim < 2:
        dense = dense.reshape(1, -1)
    rows = min(max_rows, dense.shape[0])
    return max(int(torch.unique(dense[i]).numel()) for i in range(rows))


def audit(model: nn.Module, spec: ModelSpec, policy: QuantPolicy) -> dict:
    """Walk the model and record exactly which weights were quantized."""
    router_re = spec.router_regex()
    skip_res = [re.compile(p) for p in policy.skip_patterns]
    skip_exact = set(policy.skip_exact)

    quantized: list[str] = []
    plain: list[str] = []
    kinds: dict[str, int] = {}
    quantized_params = plain_params = 0

    for fqn, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, (torch.Tensor, nn.Parameter)):
            continue

        kind = weight_kind(weight)
        kinds[kind] = kinds.get(kind, 0) + 1
        numel = _unwrap(weight).numel()

        if is_quantized(weight):
            quantized.append(fqn)
            quantized_params += numel
        else:
            plain.append(fqn)
            plain_params += numel

    routers = list(find_routers(model, spec))
    quantized_set = set(quantized)
    router_states = {fqn: (fqn in quantized_set) for fqn in routers}

    # Modules the policy asked to protect that were nonetheless quantized. Any entry here
    # means torchao's precedence order did not resolve the way the policy intended.
    def _protected(fqn: str) -> bool:
        return fqn in skip_exact or any(r.fullmatch(fqn) for r in skip_res)

    leaked = [fqn for fqn in quantized if _protected(fqn)]

    return {
        "policy": policy.name,
        "bits": policy.bits,
        "num_quantized_modules": len(quantized),
        "num_plain_modules": len(plain),
        "quantized_params": quantized_params,
        "plain_params": plain_params,
        "weight_types": kinds,
        "routers_quantized": router_states,
        "num_routers": len(routers),
        "num_routers_quantized": sum(router_states.values()),
        "num_protected_modules": sum(1 for fqn in plain if _protected(fqn)),
        "protected_but_quantized": leaked,
        "example_quantized": quantized[:5],
        "example_plain": plain[:5],
    }


class VerificationError(RuntimeError):
    """Raised when a loaded model does not match its declared policy."""


def check(model: nn.Module, spec: ModelSpec, policy: QuantPolicy, strict: bool = True) -> dict:
    """Audit the model and assert the policy was honoured.

    Returns the audit report. With ``strict=True`` (the default) a mismatch raises rather
    than letting a mislabelled run reach the results table.
    """
    report = audit(model, spec, policy)
    problems: list[str] = []

    n_routers = report["num_routers"]
    n_quant_routers = report["num_routers_quantized"]

    if policy.is_gold:
        if report["num_quantized_modules"] != 0:
            problems.append(
                f"gold policy but {report['num_quantized_modules']} modules are quantized"
            )
    else:
        if report["num_quantized_modules"] == 0:
            problems.append(
                f"policy {policy.name!r} requested INT{policy.bits} but nothing was quantized. "
                "The _default rule matched no modules - check the transformers version "
                "and whether experts are fused into 3D parameter stacks."
            )

        if policy.name == "mixed":
            if n_quant_routers != 0:
                problems.append(
                    f"mixed policy must leave every router in high precision but "
                    f"{n_quant_routers}/{n_routers} are quantized"
                )
        elif n_quant_routers != n_routers:
            # uniform, placebo and attention all quantize the routers. For placebo this
            # is the whole point of the control: it protects a random parameter-matched
            # set *instead of* the routers. A placebo that accidentally spared a router
            # is a second `mixed` run wearing a different label, and would appear to
            # confirm whatever `mixed` showed.
            problems.append(
                f"{policy.name} policy must quantize all {n_routers} routers but only "
                f"{n_quant_routers} are quantized. Without this it is not distinguishable "
                "from 'mixed' and the comparison is vacuous."
            )

        if report["protected_but_quantized"]:
            problems.append(
                "modules the policy protects were quantized anyway: "
                f"{report['protected_but_quantized'][:5]}"
            )

    if problems and strict:
        raise VerificationError(
            "Loaded model does not match its policy:\n  - " + "\n  - ".join(problems)
        )
    report["problems"] = problems
    return report


def check_bit_width(model: nn.Module, policy: QuantPolicy, sample: int = 4) -> dict:
    """Confirm the requested precision was actually applied, via distinct-value counts.

    Independent of torchao's own bookkeeping: we look at the dequantized values and count
    how many distinct levels each output channel uses.
    """
    if policy.is_gold:
        return {"checked": 0, "note": "gold policy, nothing to check"}

    allowed = 2**policy.bits
    candidates = [
        fqn
        for fqn, module in model.named_modules()
        if isinstance(getattr(module, "weight", None), (torch.Tensor, nn.Parameter))
        and is_quantized(module.weight)
    ]
    if not candidates:
        return {"checked": 0, "allowed_levels": allowed, "samples": [], "violations": [],
                "passed": False, "note": "policy is not gold but no quantized weight found"}

    # Spread the sample across the depth of the model. Taking the first few would only
    # ever inspect layer 0 and would miss a failure isolated to later layers.
    modules = dict(model.named_modules())
    step = max(1, len(candidates) // sample)
    picked = candidates[::step][:sample]

    checked = [
        {"module": fqn, "levels": effective_levels(modules[fqn].weight), "allowed": allowed}
        for fqn in picked
    ]
    violations = [c for c in checked if c["levels"] > allowed]
    return {
        "checked": len(checked),
        "allowed_levels": allowed,
        "samples": checked,
        "violations": violations,
        "passed": not violations,
    }
