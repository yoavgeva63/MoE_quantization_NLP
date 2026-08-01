"""Metric correctness, including the identity cases the rubric asks us to demonstrate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moequant.metrics import (
    bootstrap_ci,
    compare_routing,
    effective_weights,
    per_token_js,
    per_token_kl,
    per_token_selection,
    topk_mask,
    usage_stats,
)


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(256, 8)


# -- identity cases -----------------------------------------------------------------


def test_kl_of_identical_is_zero(logits):
    assert torch.allclose(per_token_kl(logits, logits), torch.zeros(len(logits)), atol=1e-6)


def test_js_of_identical_is_zero(logits):
    assert torch.allclose(per_token_js(logits, logits), torch.zeros(len(logits)), atol=1e-6)


def test_selection_of_identical_is_perfect(logits):
    result = per_token_selection(logits, logits, k=2)
    assert result["jaccard_distance"].max().item() == pytest.approx(0.0, abs=1e-6)
    assert result["top1_error"].sum().item() == 0
    assert result["set_mismatch"].sum().item() == 0
    assert result["weight_l1"].max().item() == pytest.approx(0.0, abs=1e-6)


# -- basic properties ---------------------------------------------------------------


def test_kl_is_nonnegative(logits):
    other = torch.randn_like(logits)
    assert per_token_kl(logits, other).min().item() >= -1e-6


def test_js_is_bounded_by_log2(logits):
    other = torch.randn_like(logits) * 10
    assert per_token_js(logits, other).max().item() <= np.log(2) + 1e-6


def test_js_is_symmetric(logits):
    other = torch.randn_like(logits)
    assert torch.allclose(per_token_js(logits, other), per_token_js(other, logits), atol=1e-6)


def test_kl_is_not_symmetric(logits):
    other = torch.randn_like(logits)
    assert not torch.allclose(per_token_kl(logits, other), per_token_kl(other, logits), atol=1e-4)


def test_kl_grows_with_perturbation(logits):
    small = per_token_kl(logits, logits + 0.01 * torch.randn_like(logits)).mean()
    large = per_token_kl(logits, logits + 1.00 * torch.randn_like(logits)).mean()
    assert large > small


# -- top-k mechanics ----------------------------------------------------------------


def test_topk_mask_selects_exactly_k(logits):
    for k in (1, 2, 4):
        assert (topk_mask(logits, k).sum(-1) == k).all()


def test_effective_weights_sum_to_one(logits):
    weights = effective_weights(logits, k=3)
    assert torch.allclose(weights.sum(-1), torch.ones(len(logits)), atol=1e-5)
    assert (weights > 0).sum(-1).max().item() <= 3


def test_full_swap_gives_jaccard_one():
    """Disjoint expert sets must give the maximum Jaccard distance."""
    gold = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    cand = torch.tensor([[0.0, 0.0, 10.0, 9.0]])
    result = per_token_selection(gold, cand, k=2)
    assert result["jaccard_distance"].item() == pytest.approx(1.0)
    assert result["top1_error"].item() == 1.0


def test_partial_swap_gives_partial_jaccard():
    """One shared expert out of two: intersection 1, union 3."""
    gold = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    cand = torch.tensor([[10.0, 0.0, 9.0, 0.0]])
    result = per_token_selection(gold, cand, k=2)
    assert result["jaccard_distance"].item() == pytest.approx(1 - 1 / 3)
    assert result["top1_error"].item() == 0.0  # top-1 unchanged


# -- expert load --------------------------------------------------------------------


def test_uniform_routing_maximises_entropy():
    torch.manual_seed(0)
    balanced = torch.randn(4096, 8)
    stats = usage_stats(balanced, k=2)
    assert stats["marginal_entropy_normalized"] > 0.98
    assert stats["dead_experts"] == 0


def test_collapsed_routing_has_low_entropy():
    """All tokens forced to the same two experts: the collapse case we are watching for."""
    collapsed = torch.full((512, 8), -10.0)
    collapsed[:, 0] = 10.0
    collapsed[:, 1] = 9.0
    stats = usage_stats(collapsed, k=2)
    assert stats["marginal_entropy_normalized"] < 0.4
    assert stats["dead_experts"] == 6
    assert stats["unused_experts"] == 6


# -- bootstrap ----------------------------------------------------------------------


def test_bootstrap_brackets_the_mean():
    values = torch.randn(1000) + 5.0
    ci = bootstrap_ci(values, n_boot=200, seed=0)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["mean"] == pytest.approx(5.0, abs=0.2)


def test_bootstrap_on_constant_has_zero_width():
    ci = bootstrap_ci(torch.full((100,), 3.0), n_boot=100, seed=0)
    assert ci["ci_low"] == pytest.approx(3.0)
    assert ci["ci_high"] == pytest.approx(3.0)


def test_grouped_bootstrap_is_wider_than_naive():
    """Correlated tokens within a sequence must not produce artificially tight intervals."""
    rng = np.random.default_rng(0)
    n_groups, per_group = 20, 50
    # Strong between-sequence variation, negligible within-sequence variation.
    offsets = rng.normal(0, 3, n_groups).repeat(per_group)
    values = torch.tensor(offsets + rng.normal(0, 0.01, n_groups * per_group))
    groups = np.repeat(np.arange(n_groups), per_group)

    naive = bootstrap_ci(values, None, n_boot=400, seed=0)
    grouped = bootstrap_ci(values, groups, n_boot=400, seed=0)
    assert (grouped["ci_high"] - grouped["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_bootstrap_handles_empty():
    ci = bootstrap_ci(torch.tensor([]))
    assert ci["n"] == 0


# -- whole-model comparison ----------------------------------------------------------


def test_compare_routing_self_is_zero():
    """The gold self-check that every real run performs."""
    torch.manual_seed(0)
    logits = {i: torch.randn(128, 8) for i in range(3)}
    result = compare_routing(logits, logits, k=2, n_boot=50)
    assert result["pooled"]["kl"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert result["pooled"]["top1_error"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert set(result["per_layer"]) == {"0", "1", "2"}


def test_compare_routing_detects_difference():
    torch.manual_seed(0)
    gold = {i: torch.randn(128, 8) for i in range(3)}
    cand = {i: v + torch.randn_like(v) for i, v in gold.items()}
    result = compare_routing(gold, cand, k=2, n_boot=50)
    assert result["pooled"]["kl"]["mean"] > 0.0
    assert result["pooled"]["top1_error"]["mean"] > 0.0


def test_misaligned_shapes_raise():
    with pytest.raises(ValueError, match="align token-for-token"):
        per_token_kl(torch.randn(10, 8), torch.randn(11, 8))
