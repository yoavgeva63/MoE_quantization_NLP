"""The Part 2 decomposition, checked on cases where the answer is known in advance."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moequant.attribution import (
    Cells,
    attribute,
    check_reconstruction,
    logits,
    strided_positions,
)

LAYERS = (0, 1)
NUM_SEQUENCES = 16
SEQ_LEN = 32
STRIDE = 4
HIDDEN = 12
EXPERTS = 8
TOP_K = 2


def _groups() -> np.ndarray:
    return np.repeat(np.arange(NUM_SEQUENCES), SEQ_LEN)


def _cells(
    weight_noise: float = 0.0,
    activation_noise: float = 0.0,
    seed: int = 0,
) -> Cells:
    """Synthetic cells where each mechanism's magnitude is set independently."""
    torch.manual_seed(seed)
    rows = NUM_SEQUENCES * SEQ_LEN // STRIDE
    x_gold = {layer: torch.randn(rows, HIDDEN) for layer in LAYERS}
    w_gold = {layer: torch.randn(EXPERTS, HIDDEN) for layer in LAYERS}
    return Cells(
        x_gold=x_gold,
        x_quant={
            layer: value + activation_noise * torch.randn_like(value)
            for layer, value in x_gold.items()
        },
        w_gold=w_gold,
        w_quant={
            layer: value + weight_noise * torch.randn_like(value)
            for layer, value in w_gold.items()
        },
        groups=_groups(),
        top_k=TOP_K,
        stride=STRIDE,
    )


# -- stride bookkeeping --------------------------------------------------------------


def test_strided_positions_counts_rows_per_sequence():
    mask = strided_positions(_groups(), STRIDE)
    assert mask.sum() == NUM_SEQUENCES * SEQ_LEN // STRIDE


def test_strided_positions_restarts_each_sequence():
    """The capture strides within a sequence, so every sequence keeps its first token."""
    groups = np.array([0, 0, 0, 1, 1, 1])
    assert strided_positions(groups, 2).tolist() == [True, False, True, True, False, True]


def test_strided_positions_handles_ragged_sequences():
    groups = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    mask = strided_positions(groups, 3)
    assert mask.tolist() == [True, False, False, True, False, False, True, False]


def test_strided_positions_on_empty():
    assert strided_positions(np.array([]), 4).size == 0


# -- identity cases ------------------------------------------------------------------


def test_no_quantization_error_gives_no_drift():
    """With both copies identical, all three cells must report exactly zero."""
    report = attribute(_cells(), n_boot=20)
    for cell in ("weights_only", "activations_only", "both"):
        assert report["pooled"][cell]["top1_error"]["mean"] == pytest.approx(0.0, abs=1e-9)
        assert report["pooled"][cell]["kl"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_weight_error_alone_isolates_the_weight_cell():
    report = attribute(_cells(weight_noise=0.5), n_boot=20)
    pooled = report["pooled"]
    assert pooled["weights_only"]["top1_error"]["mean"] > 0.0
    assert pooled["activations_only"]["top1_error"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert report["shares"]["weights_of_mechanisms"]["mean"] == pytest.approx(1.0)


def test_activation_drift_alone_isolates_the_activation_cell():
    report = attribute(_cells(activation_noise=0.5), n_boot=20)
    pooled = report["pooled"]
    assert pooled["activations_only"]["top1_error"]["mean"] > 0.0
    assert pooled["weights_only"]["top1_error"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert report["shares"]["weights_of_mechanisms"]["mean"] == pytest.approx(0.0)


# -- the property Part 2 exists to measure --------------------------------------------


def test_dominant_mechanism_drives_the_share():
    """The share must follow whichever source was given the larger perturbation."""
    weight_heavy = attribute(_cells(weight_noise=0.5, activation_noise=0.02), n_boot=20)
    activation_heavy = attribute(_cells(weight_noise=0.02, activation_noise=0.5), n_boot=20)
    assert weight_heavy["shares"]["weights_of_mechanisms"]["mean"] > 0.8
    assert activation_heavy["shares"]["weights_of_mechanisms"]["mean"] < 0.2


def test_both_cell_exceeds_either_mechanism_alone():
    report = attribute(_cells(weight_noise=0.3, activation_noise=0.3), n_boot=20)
    pooled = report["pooled"]
    assert pooled["both"]["top1_error"]["mean"] >= pooled["weights_only"]["top1_error"]["mean"]
    assert pooled["both"]["top1_error"]["mean"] >= pooled["activations_only"]["top1_error"]["mean"]


def test_shares_have_intervals_and_report_interaction():
    report = attribute(_cells(weight_noise=0.3, activation_noise=0.3), n_boot=50)
    share = report["shares"]["weights_of_both"]
    assert share["ci_low"] <= share["mean"] <= share["ci_high"]
    # Non-additivity is expected; the residual must be reported, not hidden.
    assert "interaction" in report["shares"]


def test_report_records_its_own_provenance():
    report = attribute(_cells(weight_noise=0.1), n_boot=20)
    assert report["layers"] == list(LAYERS)
    assert report["num_sequences"] == NUM_SEQUENCES
    assert report["num_tokens"] == NUM_SEQUENCES * SEQ_LEN // STRIDE
    assert report["top_k"] == TOP_K


# -- the gate on the reconstruction ---------------------------------------------------


def test_reconstruction_passes_against_matching_logits():
    cells = _cells(weight_noise=0.2)
    mask = strided_positions(cells.groups, cells.stride)
    recorded = {}
    for layer in cells.layers:
        full = torch.zeros(cells.groups.size, EXPERTS)
        full[mask] = logits(cells.x_gold[layer], cells.w_gold[layer])
        recorded[layer] = full
    check = check_reconstruction(cells, recorded)
    assert check["passed"]
    assert check["top1_agreement"] == pytest.approx(1.0)


def test_reconstruction_fails_against_unrelated_logits():
    """A stale or misaligned gold capture must be caught, not averaged into the result."""
    cells = _cells()
    torch.manual_seed(1)
    recorded = {layer: torch.randn(cells.groups.size, EXPERTS) for layer in cells.layers}
    check = check_reconstruction(cells, recorded)
    assert not check["passed"]


def test_reconstruction_rejects_mismatched_token_counts():
    """Gold logits from a different run must be refused rather than silently truncated."""
    cells = _cells()
    recorded = {layer: torch.randn(cells.groups.size + 1, EXPERTS) for layer in cells.layers}
    with pytest.raises(ValueError, match="do not align"):
        check_reconstruction(cells, recorded)


def test_reconstruction_rejects_wrong_stride():
    cells = _cells()
    mask = strided_positions(cells.groups, cells.stride)
    recorded = {}
    for layer in cells.layers:
        full = torch.zeros(cells.groups.size, EXPERTS)
        full[mask] = logits(cells.x_gold[layer], cells.w_gold[layer])
        recorded[layer] = full
    # A stride that does not match the capture selects the wrong rows, so the rebuilt
    # cell has a different token count than the recorded logits it is compared against.
    wrong = Cells(**{**cells.__dict__, "stride": cells.stride * 2})
    with pytest.raises(ValueError, match="do not align"):
        check_reconstruction(wrong, recorded)
