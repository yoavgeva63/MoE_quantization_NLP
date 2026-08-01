"""Verification gates.

These exist because every failure mode in this project is silent. A model whose routers
were never quantized still runs and still reports a plausible perplexity, so `uniform`
and `mixed` would be the same experiment reported under two names.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from moequant.quantize import build_policy
from moequant.verify import (
    VerificationError,
    audit,
    check,
    check_bit_width,
    effective_levels,
    is_quantized,
    weight_kind,
)


class FakeQuantized(torch.Tensor):
    """Stands in for torchao's quantized tensor subclass."""

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)


def fake_quantize_module(module: nn.Module, bits: int | None = None) -> None:
    """Replace a module's weight with a subclass instance, as torchao effectively does."""
    weight = module.weight.data.clone()
    if bits is not None:
        # Per-output-channel symmetric round trip, so distinct-value counts are realistic.
        levels = 2 ** (bits - 1) - 1
        scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / levels
        weight = torch.round(weight / scale).clamp(-levels - 1, levels) * scale
    del module._parameters["weight"]
    module.weight = FakeQuantized(weight)


# -- weight classification -------------------------------------------------------------


def test_plain_weight_is_not_quantized(tiny_model):
    weight = tiny_model.model.layers[0].mlp.gate.weight
    assert weight_kind(weight) == "Tensor"
    assert not is_quantized(weight)


def test_subclass_weight_is_detected(tiny_model):
    gate = tiny_model.model.layers[0].mlp.gate
    fake_quantize_module(gate)
    assert weight_kind(gate.weight) == "FakeQuantized"
    assert is_quantized(gate.weight)


# -- bit-width evidence ----------------------------------------------------------------


def test_unquantized_weight_has_many_levels(tiny_model):
    """A full-precision row uses essentially as many distinct values as it has entries."""
    gate = tiny_model.model.layers[0].mlp.gate
    assert effective_levels(gate.weight) > 8


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
def test_quantized_weight_respects_its_bit_budget(tiny_model, bits):
    """The check that catches a bit-width silently not being applied."""
    torch.manual_seed(0)
    linear = nn.Linear(512, 64, bias=False)
    fake_quantize_module(linear, bits=bits)
    assert effective_levels(linear.weight) <= 2**bits


def test_bit_width_check_flags_violation(tiny_model, tiny_spec):
    """An 8-bit weight labelled as INT3 must be caught."""
    gate = tiny_model.model.layers[0].mlp.gate
    fake_quantize_module(gate, bits=8)
    policy = build_policy("uniform", tiny_spec, bits=3)
    result = check_bit_width(tiny_model, policy)
    assert not result["passed"]


def test_bit_width_check_skipped_for_gold(tiny_model, tiny_spec):
    policy = build_policy("gold", tiny_spec, bits=None)
    assert check_bit_width(tiny_model, policy)["checked"] == 0


# -- policy conformance ----------------------------------------------------------------


def apply_policy(model, policy, skip=()):
    """Quantize what `policy` asks for, the way a well-behaved backend would.

    Honouring the policy's own skip patterns matters: an embedding or lm_head left
    quantized is a real failure the audit is supposed to report, so a helper that
    ignored them would make the gate untestable.
    """
    patterns = [re.compile(p) for p in policy.skip_patterns]
    exact = set(policy.skip_exact)
    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear) or fqn in skip:
            continue
        if fqn in exact or any(p.fullmatch(fqn) for p in patterns):
            continue
        fake_quantize_module(module, bits=policy.bits or 4)


def _quantize_all_linears(model, skip=()):
    """Quantize every Linear regardless of policy, to simulate a backend gone wrong."""
    for fqn, module in model.named_modules():
        if isinstance(module, nn.Linear) and fqn not in skip:
            fake_quantize_module(module, bits=4)


def test_gold_audit_reports_nothing_quantized(tiny_model, tiny_spec):
    policy = build_policy("gold", tiny_spec, bits=None)
    report = check(tiny_model, tiny_spec, policy)
    assert report["num_quantized_modules"] == 0
    assert report["num_routers_quantized"] == 0


def test_gold_rejects_a_quantized_model(tiny_model, tiny_spec):
    fake_quantize_module(tiny_model.model.layers[0].mlp.gate)
    policy = build_policy("gold", tiny_spec, bits=None)
    with pytest.raises(VerificationError, match="gold policy but"):
        check(tiny_model, tiny_spec, policy)


def test_uniform_requires_every_router_quantized(tiny_model, tiny_spec):
    """The exact silent failure that would make `uniform` identical to `mixed`."""
    routers = [f for f, _ in tiny_model.named_modules() if tiny_spec.is_router(f)]
    _quantize_all_linears(tiny_model, skip=set(routers))

    policy = build_policy("uniform", tiny_spec, bits=4)
    with pytest.raises(VerificationError, match="must quantize all"):
        check(tiny_model, tiny_spec, policy)


def test_uniform_passes_when_routers_are_quantized(tiny_model, tiny_spec):
    policy = build_policy("uniform", tiny_spec, bits=4)
    apply_policy(tiny_model, policy)
    report = check(tiny_model, tiny_spec, policy)
    assert report["num_routers_quantized"] == report["num_routers"] > 0


def test_mixed_requires_routers_untouched(tiny_model, tiny_spec):
    _quantize_all_linears(tiny_model)  # includes routers, which `mixed` forbids
    policy = build_policy("mixed", tiny_spec, bits=4)
    with pytest.raises(VerificationError, match="must leave every router"):
        check(tiny_model, tiny_spec, policy)


def test_mixed_passes_when_routers_are_spared(tiny_model, tiny_spec):
    policy = build_policy("mixed", tiny_spec, bits=4)
    apply_policy(tiny_model, policy)
    report = check(tiny_model, tiny_spec, policy)
    assert report["num_routers_quantized"] == 0
    assert report["num_quantized_modules"] > 0


def test_protected_module_that_slips_through_is_reported(tiny_model, tiny_spec):
    """lm_head is in ALWAYS_SKIP, so finding it quantized means precedence misfired."""
    policy = build_policy("uniform", tiny_spec, bits=4)
    apply_policy(tiny_model, policy)
    fake_quantize_module(tiny_model.lm_head, bits=4)
    with pytest.raises(VerificationError, match="protects were quantized anyway"):
        check(tiny_model, tiny_spec, policy)


def test_placebo_must_still_quantize_the_routers(tiny_model, tiny_spec):
    """The control is only meaningful if it leaves the routers quantized.

    A placebo that spared a router would be a second `mixed` run under another name and
    would appear to confirm whatever `mixed` showed.
    """
    victims = tuple(
        f for f, m in tiny_model.named_modules()
        if isinstance(m, nn.Linear) and not tiny_spec.is_router(f) and "proj" in f
    )[:2]
    policy = build_policy("placebo", tiny_spec, bits=4, placebo_modules=victims)

    routers = {f for f, _ in tiny_model.named_modules() if tiny_spec.is_router(f)}
    apply_policy(tiny_model, policy, skip=routers)  # the bug: routers spared

    with pytest.raises(VerificationError, match="must quantize all"):
        check(tiny_model, tiny_spec, policy)


def test_placebo_passes_when_it_protects_non_routers(tiny_model, tiny_spec):
    victims = tuple(
        f for f, m in tiny_model.named_modules()
        if isinstance(m, nn.Linear) and not tiny_spec.is_router(f) and "proj" in f
    )[:2]
    policy = build_policy("placebo", tiny_spec, bits=4, placebo_modules=victims)
    apply_policy(tiny_model, policy)

    report = check(tiny_model, tiny_spec, policy)
    assert report["num_routers_quantized"] == report["num_routers"] > 0
    assert not report["protected_but_quantized"]
    for fqn in victims:
        assert not report["routers_quantized"].get(fqn, False)


def test_quantizing_nothing_is_rejected(tiny_model, tiny_spec):
    """If the _default rule matched no modules, the run must not proceed."""
    policy = build_policy("uniform", tiny_spec, bits=4)
    with pytest.raises(VerificationError, match="nothing was quantized"):
        check(tiny_model, tiny_spec, policy)


def test_non_strict_collects_problems_instead_of_raising(tiny_model, tiny_spec):
    policy = build_policy("uniform", tiny_spec, bits=4)
    report = check(tiny_model, tiny_spec, policy, strict=False)
    assert report["problems"]


def test_audit_counts_parameters(tiny_model, tiny_spec):
    _quantize_all_linears(tiny_model)
    policy = build_policy("uniform", tiny_spec, bits=4)
    report = audit(tiny_model, tiny_spec, policy)
    assert report["quantized_params"] > 0
    assert "FakeQuantized" in report["weight_types"]
