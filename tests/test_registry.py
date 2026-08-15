"""Router targeting: the single most consequential thing to get right.

If the pattern matches `gate_proj` we quantize the wrong thing; if it misses `gate` we
quantize nothing and `uniform` silently equals `mixed`.
"""

from __future__ import annotations

import pytest

from moequant.registry import (
    MODEL_SPECS,
    find_routers,
    layer_index,
    parameter_census,
    resolve_topology,
)


def test_finds_exactly_one_router_per_layer(tiny_model, tiny_spec, tiny_config):
    routers = find_routers(tiny_model, tiny_spec)
    assert len(routers) == tiny_config.num_hidden_layers
    assert list(routers) == [
        f"model.layers.{i}.mlp.gate" for i in range(tiny_config.num_hidden_layers)
    ]


def test_router_pattern_excludes_expert_gate_proj(tiny_model, tiny_spec):
    routers = find_routers(tiny_model, tiny_spec)
    assert not any("gate_proj" in fqn for fqn in routers)


def test_router_pattern_excludes_shared_expert_gate(tiny_model_shared, tiny_spec_shared):
    """`shared_expert_gate` is gate-like but is not a router; it must not be hooked."""
    routers = find_routers(tiny_model_shared, tiny_spec_shared)
    assert not any("shared_expert_gate" in fqn for fqn in routers)
    assert all(fqn.endswith(".mlp.gate") for fqn in routers)


def test_protect_patterns_can_cover_more_than_routers(tiny_model_shared, tiny_spec_shared):
    """The mechanism supports protecting extra gates, even though no real spec does.

    Qwen's `shared_expert_gate` was protected at one point. It is not a router - it emits
    one scalar weighting the shared expert rather than choosing among experts - so
    protecting it made `mixed` mean something different on Qwen than on OLMoE. The real
    specs now protect routers only; this keeps the capability under test for future
    architectures that genuinely need it.
    """
    import re

    protect = [re.compile(p) for p in tiny_spec_shared.protect_patterns]
    fqn = "model.layers.0.mlp.shared_expert_gate"
    assert any(p.fullmatch(fqn) for p in protect)


def test_missing_routers_raises(tiny_model):
    from moequant.registry import ModelSpec

    bogus = ModelSpec(
        key="bogus", model_id="x", router_pattern=r".*\.router", protect_patterns=()
    )
    with pytest.raises(RuntimeError, match="No router modules matched"):
        find_routers(tiny_model, bogus)


def test_layer_index_parsing():
    assert layer_index("model.layers.7.mlp.gate") == 7
    assert layer_index("model.layers.13.mlp.gate") == 13
    assert layer_index("no_layers_here") == -1


def test_topology_matches_shapes(tiny_model, tiny_spec, tiny_config):
    topology = resolve_topology(tiny_model, tiny_spec)
    assert topology["num_experts"] == tiny_config.num_experts
    assert topology["top_k"] == tiny_config.num_experts_per_tok
    assert topology["hidden_size"] == tiny_config.hidden_size
    assert topology["num_router_layers"] == tiny_config.num_hidden_layers


def test_topology_rejects_config_shape_mismatch(tiny_model, tiny_spec):
    """A config that disagrees with the real weight shape must fail loudly."""
    tiny_model.config.num_experts = 999
    with pytest.raises(RuntimeError, match="experts but router weight"):
        resolve_topology(tiny_model, tiny_spec)


def test_router_fraction_is_small(tiny_model, tiny_spec):
    census = parameter_census(tiny_model, tiny_spec)
    assert census["total_params"] > 0
    assert 0 < census["router_params"] < census["total_params"]
    # The whole premise: routers are a tiny slice of the model.
    assert census["router_fraction"] < 0.05


def test_real_specs_are_wellformed():
    for key, spec in MODEL_SPECS.items():
        assert spec.key == key
        assert spec.protect_patterns, f"{key} protects nothing"
        # `mixed` must protect the routers and nothing besides. Anything extra stops the
        # comparison against `uniform` from isolating the router, and stops `mixed`
        # meaning the same thing from one architecture to the next.
        assert spec.protect_patterns == (spec.router_pattern,), (
            f"{key} protects more than its routers: {spec.protect_patterns}"
        )


@pytest.mark.parametrize(
    ("fqn", "expected"),
    [
        ("model.layers.0.mlp.gate", True),
        ("model.layers.11.mlp.gate", True),
        ("model.layers.0.mlp.gate_proj", False),
        ("model.layers.0.mlp.shared_expert_gate", False),
        ("model.layers.0.mlp.experts.3.gate_proj", False),
        ("model.layers.0.self_attn.q_proj", False),
        ("lm_head", False),
    ],
)
def test_router_pattern_cases(fqn, expected):
    """Exhaustive check of the naming collisions that actually occur in these models."""
    spec = MODEL_SPECS["qwen"]
    assert spec.is_router(fqn) is expected
