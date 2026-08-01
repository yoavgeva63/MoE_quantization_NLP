"""Hook capture and data slicing.

The alignment property tested here is load-bearing: gold and candidate runs must produce
row-for-row comparable captures, or every metric downstream is meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moequant.capture import RouterCapture, capture_routing
from moequant.data import ppl_batches, routing_batches


# -- capture ------------------------------------------------------------------------


def _batches(n=4, seq=8):
    torch.manual_seed(0)
    return [{"input_ids": torch.randint(0, 64, (1, seq))} for _ in range(n)]


def test_capture_shapes(tiny_model, tiny_spec, tiny_config):
    batches = _batches(n=4, seq=8)
    out = capture_routing(tiny_model, tiny_spec, batches, "cpu", progress=False)
    logits = out["logits"]

    assert set(logits) == set(range(tiny_config.num_hidden_layers))
    for tensor in logits.values():
        assert tensor.shape == (4 * 8, tiny_config.num_experts)


def test_capture_is_deterministic(tiny_model, tiny_spec):
    batches = _batches()
    a = capture_routing(tiny_model, tiny_spec, batches, "cpu", progress=False)["logits"]
    b = capture_routing(tiny_model, tiny_spec, batches, "cpu", progress=False)["logits"]
    for layer in a:
        assert torch.equal(a[layer], b[layer])


def test_capture_aligns_across_models(tiny_model, tiny_spec, tiny_config):
    """Two different models on the same batches give comparable, aligned captures."""
    from tests.conftest import TinyMoE

    torch.manual_seed(1)
    other = TinyMoE(tiny_config)
    batches = _batches()

    gold = capture_routing(tiny_model, tiny_spec, batches, "cpu", progress=False)["logits"]
    cand = capture_routing(other, tiny_spec, batches, "cpu", progress=False)["logits"]

    assert set(gold) == set(cand)
    for layer in gold:
        assert gold[layer].shape == cand[layer].shape
    # Different weights must actually produce different routing, else the test is vacuous.
    assert not torch.allclose(gold[0], cand[0])


def test_capture_inputs_are_strided(tiny_model, tiny_spec, tiny_config):
    batches = _batches(n=2, seq=8)
    out = capture_routing(
        tiny_model, tiny_spec, batches, "cpu", capture_inputs=True, input_stride=4, progress=False
    )
    inputs = out["inputs"]
    # 8 tokens per batch, every 4th kept, two batches.
    assert inputs[0].shape == (2 * 2, tiny_config.hidden_size)


def test_capture_can_skip_inputs(tiny_model, tiny_spec):
    out = capture_routing(
        tiny_model, tiny_spec, _batches(), "cpu", capture_inputs=False, progress=False
    )
    assert "inputs" not in out


def test_router_weights_returned(tiny_model, tiny_spec, tiny_config):
    out = capture_routing(tiny_model, tiny_spec, _batches(), "cpu", progress=False)
    weights = out["router_weights"]
    assert set(weights) == set(range(tiny_config.num_hidden_layers))
    assert weights[0].shape == (tiny_config.num_experts, tiny_config.hidden_size)


def test_hooks_are_removed(tiny_model, tiny_spec):
    with RouterCapture(tiny_model, tiny_spec) as cap:
        routers = cap.routers
        assert all(m._forward_hooks for m in routers.values())
    assert not any(m._forward_hooks for m in routers.values())
    assert not any(m._forward_pre_hooks for m in routers.values())


def test_capture_without_forward_raises(tiny_model, tiny_spec):
    with RouterCapture(tiny_model, tiny_spec) as cap:
        pass
    with pytest.raises(RuntimeError, match="No router logits captured"):
        cap.logits()


# -- data ---------------------------------------------------------------------------


def test_routing_batches_are_disjoint_and_seeded():
    tokens = torch.arange(10_000)
    a = routing_batches(tokens, num_sequences=8, seq_len=16, seed=42)
    b = routing_batches(tokens, num_sequences=8, seq_len=16, seed=42)

    for x, y in zip(a.batches, b.batches):
        assert torch.equal(x["input_ids"], y["input_ids"])

    seen = torch.cat([x["input_ids"].flatten() for x in a.batches])
    assert seen.numel() == seen.unique().numel(), "sequences overlap"


def test_routing_batches_differ_by_seed():
    tokens = torch.arange(10_000)
    a = routing_batches(tokens, num_sequences=8, seq_len=16, seed=1)
    b = routing_batches(tokens, num_sequences=8, seq_len=16, seed=2)
    same = all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a.batches, b.batches))
    assert not same


def test_groups_align_with_captured_rows():
    """One group id per token position, matching what the router will emit."""
    tokens = torch.arange(10_000)
    batch = routing_batches(tokens, num_sequences=8, seq_len=16, seed=0)
    assert batch.groups.shape == (8 * 16,)
    assert np.array_equal(np.unique(batch.groups), np.arange(8))


def test_routing_batches_reject_oversized_request():
    with pytest.raises(ValueError):
        routing_batches(torch.arange(100), num_sequences=50, seq_len=16)


def test_ppl_windows_are_disjoint_by_default():
    tokens = torch.arange(1000)
    windows = ppl_batches(tokens, seq_len=100, stride=100)
    covered = torch.cat([w["input_ids"].flatten() for w in windows])
    assert covered.numel() == covered.unique().numel()


def test_ppl_overlap_is_masked_out():
    """With stride < seq_len, overlapping context must not be scored twice."""
    tokens = torch.arange(1000)
    windows = ppl_batches(tokens, seq_len=100, stride=50)
    assert (windows[0]["labels"] != -100).all(), "first window scores everything"
    masked = (windows[1]["labels"] == -100).sum().item()
    assert masked == 50


def test_ppl_rejects_bad_stride():
    with pytest.raises(ValueError, match="stride must be"):
        ppl_batches(torch.arange(100), seq_len=10, stride=20)
