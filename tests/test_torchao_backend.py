"""Integration against the real torchao quantizer.

The rest of the suite stands in a `FakeQuantized` tensor subclass for torchao, which
keeps the tests fast and dependency-free but leaves the single most failure-prone part of
the pipeline unexercised: whether `FqnToConfig` actually resolves our skip patterns the
way we assume, and whether `verify.py` recognises the tensor subclass torchao really
produces. Those are exactly the things that would fail silently three hours into a
cluster job.

These tests skip when torchao is absent, so `pytest` still runs anywhere, but they should
pass on the cluster before any GPU time is spent. Run them explicitly with:

    pytest tests/test_torchao_backend.py -v
"""

from __future__ import annotations

import copy

import pytest
import torch

torchao = pytest.importorskip("torchao", reason="torchao not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from torchao.quantization import quantize_  # noqa: E402

from moequant import verify  # noqa: E402
from moequant.capture import capture_routing  # noqa: E402
from moequant.metrics import compare_routing  # noqa: E402
from moequant.quantize import build_policy, to_torchao_config  # noqa: E402
from moequant.registry import resolve_topology  # noqa: E402
from tests.conftest import TinyConfig, TinyMoE  # noqa: E402


def _model():
    torch.manual_seed(0)
    return TinyMoE(TinyConfig(hidden=64, num_experts=8, layers=2))


def _quantized(model, spec, policy_name, bits):
    """Apply a policy through the genuine torchao path."""
    model = copy.deepcopy(model)
    policy = build_policy(policy_name, spec, bits)
    config = to_torchao_config(policy)
    # FqnToConfig carries its own targeting, so torchao requires filter_fn=None.
    quantize_(model, config.quant_type, filter_fn=None)
    return model, policy


# -- policy targeting ------------------------------------------------------------------


def test_uniform_really_quantizes_every_router(tiny_spec):
    model, policy = _quantized(_model(), tiny_spec, "uniform", 4)
    report = verify.check(model, tiny_spec, policy, strict=True)
    assert report["num_routers_quantized"] == report["num_routers"] > 0
    assert not report["protected_but_quantized"]


def test_mixed_really_spares_every_router(tiny_spec):
    """The regex skip must be honoured by torchao, not just by our own bookkeeping."""
    model, policy = _quantized(_model(), tiny_spec, "mixed", 4)
    report = verify.check(model, tiny_spec, policy, strict=True)
    assert report["num_routers_quantized"] == 0
    assert report["num_quantized_modules"] > 0
    assert not report["protected_but_quantized"]


def test_mixed_leaves_exactly_the_routers_extra(tiny_spec):
    """mixed differs from uniform in precisely the router modules and nothing else."""
    uniform, _ = _quantized(_model(), tiny_spec, "uniform", 4)
    mixed, _ = _quantized(_model(), tiny_spec, "mixed", 4)

    def plain(model):
        return {
            fqn
            for fqn, m in model.named_modules()
            if isinstance(getattr(m, "weight", None), (torch.Tensor, torch.nn.Parameter))
            and not verify.is_quantized(m.weight)
        }

    extra = plain(mixed) - plain(uniform)
    assert extra == {f for f in extra if tiny_spec.is_router(f)}
    assert extra, "mixed must spare at least one module that uniform quantized"


def test_embeddings_and_head_are_never_quantized(tiny_spec):
    """ALWAYS_SKIP must hold under the real backend, or PPL gains a confound."""
    model, _ = _quantized(_model(), tiny_spec, "uniform", 4)
    assert not verify.is_quantized(model.lm_head.weight)
    assert not verify.is_quantized(model.model.embed_tokens.weight)


# -- bit-width evidence ------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
def test_bit_width_check_agrees_with_the_real_quantizer(tiny_spec, bits):
    """`effective_levels` must read a genuine torchao tensor, not just our stand-in."""
    model, policy = _quantized(_model(), tiny_spec, "uniform", bits)
    result = verify.check_bit_width(model, policy)
    assert result["checked"] > 0
    assert result["passed"], result["violations"]
    assert all(s["levels"] <= 2**bits for s in result["samples"])


def test_real_quantized_weight_is_detected_as_quantized(tiny_spec):
    model, _ = _quantized(_model(), tiny_spec, "uniform", 4)
    gate = model.model.layers[0].mlp.gate
    assert verify.is_quantized(gate.weight)
    assert verify.weight_kind(gate.weight) != "Tensor"
    assert verify.dequantize(gate.weight).dtype == torch.float32


# -- end to end ----------------------------------------------------------------------------


def test_quantized_model_still_runs_and_drifts_monotonically(tiny_spec):
    """Routing drift must grow as precision falls; a flat curve means nothing applied."""
    base = _model()
    batches = [{"input_ids": torch.randint(0, 64, (1, 32))} for _ in range(4)]
    gold = capture_routing(base, tiny_spec, batches, "cpu", progress=False)
    top_k = resolve_topology(base, tiny_spec)["top_k"]

    drift = {}
    for bits in (8, 4, 2):
        model, _ = _quantized(base, tiny_spec, "uniform", bits)
        cand = capture_routing(model, tiny_spec, batches, "cpu", progress=False)
        pooled = compare_routing(gold["logits"], cand["logits"], top_k, n_boot=25)["pooled"]
        drift[bits] = pooled["kl"]["mean"]

    assert drift[8] <= drift[4] <= drift[2]
    assert drift[2] > 0.0, "INT2 must visibly perturb routing"


def test_protecting_routers_reduces_drift_under_the_real_backend(tiny_spec):
    """Part 1's hypothesis, on a synthetic model, through the real quantizer.

    This asserts direction only. Whether the gap survives on a real checkpoint with
    confidence intervals is the actual experiment, not something to bake into a test.
    """
    base = _model()
    batches = [{"input_ids": torch.randint(0, 64, (1, 32))} for _ in range(4)]
    gold = capture_routing(base, tiny_spec, batches, "cpu", progress=False)
    top_k = resolve_topology(base, tiny_spec)["top_k"]

    drift = {}
    for policy_name in ("uniform", "mixed"):
        model, _ = _quantized(base, tiny_spec, policy_name, 2)
        cand = capture_routing(model, tiny_spec, batches, "cpu", progress=False)
        pooled = compare_routing(gold["logits"], cand["logits"], top_k, n_boot=25)["pooled"]
        drift[policy_name] = pooled["kl"]["mean"]

    assert drift["mixed"] < drift["uniform"]
