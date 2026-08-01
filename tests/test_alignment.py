"""Guards against silently comparing two runs that did not see the same tokens.

Everything downstream compares gold and candidate row by row, but the only structural
check in the metric code is that the two logit tensors have the same shape. Any two runs
with the same sequence count and length satisfy that, so a changed seed, corpus, or
tokenizer would produce a full set of confident, meaningless numbers. These tests cover
the two places that can drift: the routing capture and the output-distribution reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from moequant.data import fingerprint_batches, ppl_batches, routing_batches
from moequant.evaluate import OutputReference, evaluate_lm
from moequant.runner import _check_alignment


# -- routing token fingerprint ----------------------------------------------------------


def _stream(n: int = 20_000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5000, (n,), generator=g)


def test_same_seed_gives_the_same_fingerprint():
    tokens = _stream()
    a = routing_batches(tokens, num_sequences=8, seq_len=32, seed=42)
    b = routing_batches(tokens, num_sequences=8, seq_len=32, seed=42)
    assert a.fingerprint == b.fingerprint


def test_different_seed_changes_the_fingerprint_but_not_the_shape():
    """The exact case the shape check cannot catch."""
    tokens = _stream()
    a = routing_batches(tokens, num_sequences=8, seq_len=32, seed=42)
    b = routing_batches(tokens, num_sequences=8, seq_len=32, seed=7)

    assert a.batches[0]["input_ids"].shape == b.batches[0]["input_ids"].shape
    assert len(a.batches) == len(b.batches)
    assert a.fingerprint != b.fingerprint


def test_different_corpus_changes_the_fingerprint():
    a = routing_batches(_stream(seed=0), num_sequences=8, seq_len=32, seed=42)
    b = routing_batches(_stream(seed=1), num_sequences=8, seq_len=32, seed=42)
    assert a.fingerprint != b.fingerprint


def test_fingerprint_is_order_sensitive():
    batches = [{"input_ids": torch.tensor([[1, 2, 3]])}, {"input_ids": torch.tensor([[4, 5, 6]])}]
    assert fingerprint_batches(batches) != fingerprint_batches(batches[::-1])


# -- the runner's gate --------------------------------------------------------------------


class _Cfg:
    gold_dir = Path("results/tiny/gold")


def _topology(**over):
    base = {"num_experts": 8, "top_k": 2, "num_router_layers": 3}
    base.update(over)
    return base


def test_alignment_gate_accepts_a_matching_run():
    routing = routing_batches(_stream(), num_sequences=8, seq_len=32, seed=42)
    gold = {"fingerprint": routing.fingerprint, "topology": _topology()}
    _check_alignment(gold, routing, _topology(), _Cfg())


def test_alignment_gate_rejects_a_different_seed():
    tokens = _stream()
    gold_routing = routing_batches(tokens, num_sequences=8, seq_len=32, seed=42)
    cand_routing = routing_batches(tokens, num_sequences=8, seq_len=32, seed=7)
    gold = {"fingerprint": gold_routing.fingerprint, "topology": _topology()}

    with pytest.raises(ValueError, match="do not match the gold run"):
        _check_alignment(gold, cand_routing, _topology(), _Cfg())


def test_alignment_gate_rejects_legacy_artifacts_without_a_fingerprint():
    routing = routing_batches(_stream(), num_sequences=8, seq_len=32, seed=42)
    with pytest.raises(ValueError, match="predate the alignment check"):
        _check_alignment({"topology": _topology()}, routing, _topology(), _Cfg())


def test_alignment_gate_rejects_a_topology_change():
    routing = routing_batches(_stream(), num_sequences=8, seq_len=32, seed=42)
    gold = {"fingerprint": routing.fingerprint, "topology": _topology(top_k=2)}
    with pytest.raises(ValueError, match="Topology differs"):
        _check_alignment(gold, routing, _topology(top_k=4), _Cfg())


# -- output reference alignment ------------------------------------------------------------


class _ToyLM(torch.nn.Module):
    """Minimal causal LM: embedding straight to a vocabulary projection."""

    def __init__(self, vocab: int = 40, hidden: int = 8):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _windows(n_windows: int, seq_len: int = 12, vocab: int = 40):
    tokens = torch.arange(n_windows * seq_len) % vocab
    return ppl_batches(tokens, seq_len=seq_len, max_windows=n_windows)


def test_output_reference_scores_every_gold_position_when_configs_match():
    model = _ToyLM()
    batches = _windows(4)

    _, reference = evaluate_lm(
        model, batches, "cpu", build_reference=True, top_m=5, position_stride=3, progress=False
    )
    assert reference is not None and reference.positions.numel() > 0

    metrics, _ = evaluate_lm(
        model, batches, "cpu", reference=reference, top_m=5,
        position_stride=3, progress=False,
    )
    assert metrics["output_positions"] == reference.positions.numel()
    # Scoring a model against its own reference must show perfect agreement.
    assert metrics["top1_agreement"] == pytest.approx(1.0)
    assert metrics["output_kl_topm"] == pytest.approx(0.0, abs=1e-6)


def test_output_reference_rejects_a_shorter_candidate_run():
    """Fewer ppl windows than gold: positions would silently go unscored."""
    model = _ToyLM()
    _, reference = evaluate_lm(
        model, _windows(4), "cpu", build_reference=True, top_m=5,
        position_stride=3, progress=False,
    )

    with pytest.raises(ValueError, match="Output reference misaligned"):
        evaluate_lm(
            model, _windows(2), "cpu", reference=reference, top_m=5,
            position_stride=3, progress=False,
        )


def test_output_reference_rejects_a_different_sequence_length():
    model = _ToyLM()
    _, reference = evaluate_lm(
        model, _windows(4, seq_len=12), "cpu", build_reference=True, top_m=5,
        position_stride=3, progress=False,
    )

    with pytest.raises(ValueError, match="Output reference misaligned"):
        evaluate_lm(
            model, _windows(4, seq_len=8), "cpu", reference=reference, top_m=5,
            position_stride=3, progress=False,
        )


def test_reference_roundtrips_through_a_dict():
    ref = OutputReference(
        positions=torch.tensor([0, 5, 9]),
        token_ids=torch.tensor([[1, 2], [3, 4], [5, 6]]),
        log_probs=torch.zeros(3, 2),
    )
    back = OutputReference.from_dict(ref.to_dict())
    assert torch.equal(back.positions, ref.positions)
    assert torch.equal(back.token_ids, ref.token_ids)


# -- bootstrap grouping ----------------------------------------------------------------


def test_bootstrap_rejects_mismatched_group_length():
    from moequant.metrics import bootstrap_ci

    with pytest.raises(ValueError, match="every token needs a sequence id"):
        bootstrap_ci(torch.zeros(10), groups=np.zeros(4), n_boot=5)
