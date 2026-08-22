"""Expert-collapse metrics, checked on load distributions whose answer is known.

The point of this module is that the pooled aggregation hides per-layer imbalance, so the
tests that matter are the ones showing a metric responds to concentration and that the
per-layer path sees what the pooled path cancels.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from moequant.collapse import (
    Reconstruction,
    distribution_stats,
    expert_counts,
    gini,
    gold_counts_at_stride,
    tier1_report,
    tier1_summary,
    tier2_report,
    top_decile_count,
)
from moequant.metrics import usage_stats

NUM_EXPERTS = 8


# -- Gini ------------------------------------------------------------------------------


def test_gini_of_uniform_load_is_zero():
    assert gini(np.full(NUM_EXPERTS, 1.0 / NUM_EXPERTS)) == pytest.approx(0.0)


def test_gini_of_single_expert_approaches_one():
    load = np.zeros(NUM_EXPERTS)
    load[0] = 1.0
    # The discrete maximum for n bins is (n-1)/n, not 1.
    assert gini(load) == pytest.approx((NUM_EXPERTS - 1) / NUM_EXPERTS)


def test_gini_grows_with_concentration():
    mild = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    severe = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert gini(severe) > gini(mild) > 0.0


def test_gini_is_scale_invariant():
    load = np.array([4.0, 3.0, 2.0, 1.0])
    assert gini(load) == pytest.approx(gini(load * 100))


def test_gini_of_empty_is_nan():
    assert math.isnan(gini(np.array([])))


# -- the top-decile convention ----------------------------------------------------------


def test_top_decile_rounds_to_the_nearest_expert():
    """64 experts is 6.4, and the nearest whole expert is 6."""
    assert top_decile_count(64) == 6
    assert top_decile_count(60) == 6


def test_top_decile_never_reports_zero_experts():
    assert top_decile_count(4) == 1


# -- distribution_stats -----------------------------------------------------------------


def test_balanced_counts_give_the_balanced_reference_values():
    stats = distribution_stats(np.full(NUM_EXPERTS, 100.0))
    assert stats["gini"] == pytest.approx(0.0)
    assert stats["max_share"] == pytest.approx(1.0)
    assert stats["kl_vs_uniform"] == pytest.approx(0.0)
    assert stats["marginal_entropy_normalized"] == pytest.approx(1.0)
    assert stats["unused_experts"] == 0
    assert stats["dead_experts"] == 0
    # Under perfect balance the busiest tenth holds exactly its share of the slots.
    assert stats["top_decile_share"] == pytest.approx(top_decile_count(NUM_EXPERTS) / NUM_EXPERTS)


def test_kl_against_uniform_is_the_entropy_deficit():
    counts = np.array([50.0, 20.0, 20.0, 10.0])
    stats = distribution_stats(counts)
    load = counts / counts.sum()
    entropy = -(load * np.log(load)).sum()
    assert stats["kl_vs_uniform"] == pytest.approx(math.log(counts.size) - entropy)


def test_kl_against_uniform_is_unbounded_where_entropy_saturates():
    """The reason for reporting KL: entropy is maximal at balance, so it is insensitive
    exactly where the null hypothesis sits.

    On a bank of 64 experts, one expert going from twice to eight times its fair share
    moves normalized entropy by 0.03 - indistinguishable from the 0.9975 -> 0.9968 the
    pooled table reports - while KL against uniform moves by more than an order of
    magnitude.
    """
    mild = distribution_stats(np.array([200.0] + [100.0] * 63))
    severe = distribution_stats(np.array([800.0] + [100.0] * 63))
    entropy_gap = mild["marginal_entropy_normalized"] - severe["marginal_entropy_normalized"]
    kl_ratio = severe["kl_vs_uniform"] / mild["kl_vs_uniform"]
    assert entropy_gap < 0.05
    assert kl_ratio > 10.0


def test_unused_and_dead_counts_match_metrics_usage_stats():
    """Tier 2 must use the same thresholds as the Tier 1 fields it sits beside."""
    collapsed = torch.full((512, NUM_EXPERTS), -10.0)
    collapsed[:, 0] = 10.0
    collapsed[:, 1] = 9.0
    reference = usage_stats(collapsed, k=2)
    stats = distribution_stats(expert_counts(collapsed, k=2))
    assert stats["unused_experts"] == reference["unused_experts"]
    assert stats["dead_experts"] == reference["dead_experts"]
    assert stats["max_share"] == pytest.approx(reference["max_over_mean_load"], rel=1e-5)
    assert stats["marginal_entropy_normalized"] == pytest.approx(
        reference["marginal_entropy_normalized"], rel=1e-5
    )


def test_all_counts_zero_is_reported_rather_than_dividing_by_zero():
    stats = distribution_stats(np.zeros(NUM_EXPERTS))
    assert stats["unused_experts"] == NUM_EXPERTS
    assert math.isnan(stats["gini"])


def test_expert_counts_sum_to_k_per_token():
    torch.manual_seed(0)
    logits = torch.randn(64, NUM_EXPERTS)
    assert expert_counts(logits, k=3).sum().item() == pytest.approx(64 * 3)


# -- Tier 1 aggregation -----------------------------------------------------------------


def _usage(dead: int, unused: int, entropy: float, load: float) -> dict:
    return {
        "marginal_entropy": entropy * math.log(NUM_EXPERTS),
        "marginal_entropy_normalized": entropy,
        "mean_token_entropy": 1.0,
        "dead_experts": dead,
        "unused_experts": unused,
        "max_over_mean_load": load,
        "num_experts": NUM_EXPERTS,
        "num_tokens": 1024,
    }


def _routing(*per_layer_usage: dict) -> dict:
    return {
        "per_layer": {
            str(i): {"cand_usage": usage, "gold_usage": _usage(0, 0, 1.0, 1.0)}
            for i, usage in enumerate(per_layer_usage)
        }
    }


def test_tier1_counts_slots_across_layers():
    report = tier1_report(_routing(_usage(2, 1, 0.9, 3.0), _usage(3, 0, 0.8, 5.0)))
    summary = report["summary"]
    assert summary["num_layers"] == 2
    assert summary["expert_slots"] == 2 * NUM_EXPERTS
    assert summary["dead_slots"] == 5
    assert summary["unused_slots"] == 1
    assert summary["layers_with_unused"] == 1
    assert summary["dead_fraction"] == pytest.approx(5 / 16)


def test_tier1_reports_the_worst_layer_not_the_average():
    """The whole point: one bad layer must be visible, not averaged away."""
    report = tier1_report(
        _routing(*([_usage(0, 0, 0.99, 1.1)] * 15 + [_usage(9, 2, 0.55, 9.0)]))
    )
    summary = report["summary"]
    assert summary["worst_layer_entropy"] == pytest.approx(0.55)
    assert summary["worst_layer_entropy_layer"] == 15
    assert summary["worst_max_over_mean_load"] == pytest.approx(9.0)
    assert summary["max_dead_in_a_layer"] == 9
    # The mean is close to balanced even though one layer has collapsed.
    assert summary["mean_layer_entropy"] > 0.95


def test_tier1_per_layer_rows_are_ordered_numerically():
    """String keys in the JSON must not sort layer 10 before layer 2."""
    report = tier1_report(_routing(*([_usage(0, 0, 0.9, 1.0)] * 12)))
    assert [row["layer"] for row in report["per_layer"]] == list(range(12))


def test_tier1_summary_of_nothing_is_empty():
    assert tier1_summary([]) == {}


def test_tier1_declares_its_resolution():
    """Every number has to carry which token resolution it came from."""
    assert tier1_report(_routing(_usage(0, 0, 0.9, 1.0)))["resolution"] == "full"


# -- Tier 2 aggregation -----------------------------------------------------------------


def test_tier2_summary_aggregates_over_layers():
    counts = {
        0: torch.full((NUM_EXPERTS,), 100.0),
        1: torch.tensor([700.0] + [10.0] * (NUM_EXPERTS - 1)),
    }
    report = tier2_report(counts, stride=8)
    summary = report["summary"]
    assert summary["num_layers"] == 2
    assert summary["gini_max"] > summary["gini_mean"] > 0.0
    assert summary["gini_worst_layer"] == 1
    assert summary["kl_vs_uniform_max"] > summary["kl_vs_uniform_mean"]
    assert report["stride"] == 8
    assert report["resolution"] == "strided"


def test_tier2_records_strided_zero_counts_separately():
    """Strided zero counts are reported but labelled, never mixed into the Tier 1 column."""
    counts = {0: torch.tensor([100.0, 100.0] + [0.0] * (NUM_EXPERTS - 2))}
    summary = tier2_report(counts, stride=8)["summary"]
    assert summary["strided_unused_slots"] == NUM_EXPERTS - 2
    assert "unused_slots" not in summary


def test_tier2_of_nothing_is_empty():
    assert tier2_report({}, stride=8)["per_layer"] == []


# -- the reconstruction ----------------------------------------------------------------


def test_reconstruction_counts_match_a_direct_forward():
    torch.manual_seed(0)
    inputs = {0: torch.randn(128, 16)}
    weights = {0: torch.randn(NUM_EXPERTS, 16)}
    recon = Reconstruction(inputs=inputs, weights=weights, top_k=2, stride=8)
    direct = expert_counts(torch.nn.functional.linear(inputs[0], weights[0]), 2)
    assert torch.equal(recon.counts(0), direct)


def test_reconstruction_uses_only_layers_present_in_both():
    inputs = {0: torch.randn(4, 16), 1: torch.randn(4, 16)}
    weights = {1: torch.randn(NUM_EXPERTS, 16)}
    assert Reconstruction(inputs, weights, top_k=2, stride=8).layers == [1]


def test_reconstruction_survives_fp16_storage():
    """Inputs are stored as fp16; the selection must still be the fp32 one."""
    torch.manual_seed(0)
    inputs = torch.randn(256, 16)
    weights = torch.randn(NUM_EXPERTS, 16)
    exact = expert_counts(torch.nn.functional.linear(inputs, weights), 2)
    stored = Reconstruction({0: inputs.half()}, {0: weights}, top_k=2, stride=8).counts(0)
    assert (stored - exact).abs().sum().item() / exact.sum().item() < 0.02


def test_gold_counts_are_taken_on_the_captured_positions_only():
    """Comparing 1-in-8 candidates against all tokens would confound stride with effect."""
    torch.manual_seed(0)
    gold_logits = {0: torch.randn(64, NUM_EXPERTS)}
    mask = np.zeros(64, dtype=bool)
    mask[::8] = True
    counts = gold_counts_at_stride(gold_logits, mask, top_k=2)
    assert counts[0].sum().item() == pytest.approx(8 * 2)
