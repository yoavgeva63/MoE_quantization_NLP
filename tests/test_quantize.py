"""Policy construction: the three Part 1 configs must differ in exactly one respect."""

from __future__ import annotations

import re

import pytest

from moequant.quantize import (
    ALWAYS_SKIP,
    build_policy,
    quantizable_modules,
    sample_placebo_modules,
)


def _skipped(policy, fqns):
    patterns = [re.compile(p) for p in policy.skip_patterns]
    exact = set(policy.skip_exact)
    return {f for f in fqns if f in exact or any(p.fullmatch(f) for p in patterns)}


def test_gold_quantizes_nothing(tiny_spec):
    policy = build_policy("gold", tiny_spec, bits=None)
    assert policy.is_gold
    assert policy.bits is None
    assert policy.skip_patterns == ()


def test_gold_ignores_bits(tiny_spec):
    assert build_policy("gold", tiny_spec, bits=4).is_gold


def test_non_gold_requires_bits(tiny_spec):
    with pytest.raises(ValueError, match="requires a bit-width"):
        build_policy("uniform", tiny_spec, bits=None)


def test_uniform_does_not_skip_routers(tiny_model, tiny_spec):
    policy = build_policy("uniform", tiny_spec, bits=4)
    fqns = [f for f, _ in quantizable_modules(tiny_model, tiny_spec)]
    routers = [f for f in fqns if tiny_spec.is_router(f)]
    assert routers, "test model has no routers"
    assert not (_skipped(policy, fqns) & set(routers))


def test_mixed_skips_every_router(tiny_model, tiny_spec):
    policy = build_policy("mixed", tiny_spec, bits=4)
    fqns = [f for f, _ in quantizable_modules(tiny_model, tiny_spec)]
    routers = {f for f in fqns if tiny_spec.is_router(f)}
    assert routers <= _skipped(policy, fqns)


def test_uniform_and_mixed_differ_only_by_routers(tiny_model, tiny_spec):
    """The central design property: any measured difference is attributable to routers."""
    fqns = [f for f, _ in quantizable_modules(tiny_model, tiny_spec)]
    uniform = _skipped(build_policy("uniform", tiny_spec, bits=4), fqns)
    mixed = _skipped(build_policy("mixed", tiny_spec, bits=4), fqns)

    difference = mixed ^ uniform
    assert difference, "policies are identical; the comparison would be vacuous"
    assert all(tiny_spec.is_router(f) for f in difference), (
        f"policies differ by non-router modules: {sorted(difference - set(fqns))}"
    )


def test_mixed_honours_extra_protect_patterns(tiny_model_shared, tiny_spec_shared):
    """A spec that protects more than its routers must have that respected.

    No real spec does this now - see test_real_specs_are_wellformed - but the policy
    builder should not quietly drop patterns a future architecture depends on.
    """
    policy = build_policy("mixed", tiny_spec_shared, bits=4)
    patterns = [re.compile(p) for p in policy.skip_patterns]
    assert any(p.fullmatch("model.layers.0.mlp.shared_expert_gate") for p in patterns)


def test_embeddings_and_head_always_skipped(tiny_spec):
    """Quantizing the head or embeddings would confound the router comparison."""
    for name in ("uniform", "mixed"):
        policy = build_policy(name, tiny_spec, bits=4)
        patterns = [re.compile(p) for p in policy.skip_patterns]
        assert any(p.fullmatch("lm_head") for p in patterns)
        assert any(p.fullmatch("model.embed_tokens") for p in patterns)


def test_quantizable_excludes_always_skip(tiny_model, tiny_spec):
    fqns = [f for f, _ in quantizable_modules(tiny_model, tiny_spec)]
    skips = [re.compile(p) for p in ALWAYS_SKIP]
    assert not [f for f in fqns if any(s.fullmatch(f) for s in skips)]


def test_attention_policy_skips_attention(tiny_model, tiny_spec):
    policy = build_policy("attention", tiny_spec, bits=4)
    fqns = [f for f, _ in quantizable_modules(tiny_model, tiny_spec)]
    skipped = _skipped(policy, fqns)
    assert any("self_attn" in f for f in skipped)


def test_placebo_requires_modules(tiny_spec):
    with pytest.raises(ValueError, match="needs an explicit module list"):
        build_policy("placebo", tiny_spec, bits=4)


def test_placebo_matches_router_budget(tiny_model, tiny_spec):
    """The control is only meaningful if it protects a comparable parameter count."""
    modules = sample_placebo_modules(tiny_model, tiny_spec, seed=0, tolerance=0.5)
    assert modules

    sizes = dict(quantizable_modules(tiny_model, tiny_spec))
    budget = sum(n for f, n in sizes.items() if tiny_spec.is_router(f))
    chosen = sum(sizes[f] for f in modules)
    assert 0.5 * budget <= chosen <= 1.5 * budget


def test_placebo_excludes_routers(tiny_model, tiny_spec):
    modules = sample_placebo_modules(tiny_model, tiny_spec, seed=0, tolerance=0.5)
    assert not any(tiny_spec.is_router(f) for f in modules)


def test_placebo_is_deterministic(tiny_model, tiny_spec):
    a = sample_placebo_modules(tiny_model, tiny_spec, seed=7, tolerance=0.5)
    b = sample_placebo_modules(tiny_model, tiny_spec, seed=7, tolerance=0.5)
    assert a == b


def _heterogeneous_model(tiny_spec):
    """A model whose candidate modules vary in size, so greedy accumulation can overrun.

    The tiny fixture has one uniform module size, which hides the overshoot: with a
    single granularity you either land in range or cannot. Mixed sizes are the case
    where adding a large module on top of several small ones blows the upper bound.
    """
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(64, 16, bias=False)  # router: 1024 params

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Block()
            # 4032 is larger than the tolerance band is wide, so accumulating a few
            # 640s and then adding one of these overshoots the upper bound.
            self.big = nn.Linear(64, 63, bias=False)  # 4032
            self.small_a = nn.Linear(64, 10, bias=False)  # 640
            self.small_b = nn.Linear(64, 10, bias=False)  # 640

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Mlp() for _ in range(6))

    return Model()


@pytest.mark.parametrize("seed", range(12))
def test_placebo_never_returns_an_out_of_tolerance_set(tiny_spec, seed):
    """Whenever sampling succeeds, the protected budget really is matched."""
    model = _heterogeneous_model(tiny_spec)
    spec = tiny_spec.__class__(
        key="het", model_id="het/synthetic",
        router_pattern=r".*\.mlp\.gate", protect_patterns=(r".*\.mlp\.gate",),
    )

    modules = sample_placebo_modules(model, spec, seed=seed, tolerance=0.25)

    sizes = dict(quantizable_modules(model, spec))
    budget = sum(n for f, n in sizes.items() if spec.is_router(f))
    chosen = sum(sizes[f] for f in modules)
    assert 0.75 * budget <= chosen <= 1.25 * budget
    assert not any(spec.is_router(f) for f in modules)


def test_unknown_policy_rejected(tiny_spec):
    with pytest.raises(ValueError, match="Unknown policy"):
        build_policy("nonsense", tiny_spec, bits=4)


def test_unsupported_bits_rejected(tiny_spec):
    from moequant.quantize import _weight_dtype

    with pytest.raises(ValueError, match="not supported"):
        _weight_dtype(7)
