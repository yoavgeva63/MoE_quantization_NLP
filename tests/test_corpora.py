"""Corpus selection, and the guard that stops one corpus overwriting another's results.

Two failure modes live here, both of which cost real GPU hours rather than a test run.
A C4 shard is ~45k documents, so loading it without truncation produces a token stream two
orders of magnitude longer than the sweep needs; and run directories are keyed by model
and policy but not by corpus, so a second corpus pointed at the same results root
overwrites the first, gold artifacts included.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from moequant.config import ExperimentConfig
from moequant.data import CORPORA, load_token_stream
from moequant.runner import METRICS, _check_results_corpus

# -- corpus definitions ----------------------------------------------------------------


def test_every_corpus_declares_a_text_field():
    for name, cfg in CORPORA.items():
        assert "field" in cfg, f"{name} has no text field"
        assert "split" in cfg, f"{name} has no split"


def test_c4_shard_is_keyed_by_its_split():
    """A bare data_files string lands in `train`, and split="validation" then fails."""
    cfg = CORPORA["c4"]
    assert isinstance(cfg["data_files"], dict)
    assert cfg["split"] in cfg["data_files"]


# -- document truncation ---------------------------------------------------------------


class _FakeDataset:
    """Stands in for a `datasets.Dataset`, and records what was materialised.

    `selected` is the point of the fixture: truncating after the text column has been
    read would still work and still exhaust memory on a real shard.
    """

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.selected: list[int] | None = None

    def __len__(self) -> int:
        return len(self.texts)

    def select(self, indices):
        self.selected = list(indices)
        picked = _FakeDataset([self.texts[i] for i in self.selected])
        picked.selected = self.selected
        return picked

    def __getitem__(self, field: str) -> list[str]:
        assert field == "text"
        return self.texts


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = torch.tensor([[float(len(word)) for word in text.split()]], dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids)


@pytest.fixture
def fake_datasets(monkeypatch):
    """Install a stub `datasets` module and hand back the dataset it will serve."""
    dataset = _FakeDataset([f"doc {i} body text" for i in range(50)])
    module = types.ModuleType("datasets")
    module.load_dataset = lambda **kwargs: dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return dataset


def test_max_documents_truncates_before_reading_the_text_column(fake_datasets):
    load_token_stream(_FakeTokenizer(), "c4", max_documents=10)
    assert fake_datasets.selected == list(range(10))


def test_max_documents_none_keeps_everything(fake_datasets):
    load_token_stream(_FakeTokenizer(), "c4")
    assert fake_datasets.selected is None


def test_max_documents_above_the_corpus_size_is_clamped(fake_datasets):
    """`select` raises on out-of-range indices, so the cap has to be clamped."""
    load_token_stream(_FakeTokenizer(), "c4", max_documents=10_000)
    assert fake_datasets.selected == list(range(50))


def test_unknown_corpus_is_rejected():
    with pytest.raises(KeyError, match="Unknown corpus"):
        load_token_stream(_FakeTokenizer(), "not-a-corpus")


# -- the results-tree guard ------------------------------------------------------------


def _write_run(root, model: str, run: str, corpus: str) -> None:
    run_dir = root / model / run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / METRICS).write_text(json.dumps({"dataset": {"corpus": corpus}}))


def _cfg(root, corpus: str) -> ExperimentConfig:
    return ExperimentConfig(model_key="olmoe", corpus=corpus, results_dir=str(root))


def test_guard_allows_an_empty_results_tree(tmp_path):
    _check_results_corpus(_cfg(tmp_path, "c4"))


def test_guard_allows_the_same_corpus(tmp_path):
    _write_run(tmp_path, "olmoe", "gold", "wikitext2")
    _check_results_corpus(_cfg(tmp_path, "wikitext2"))


def test_guard_rejects_a_different_corpus(tmp_path):
    _write_run(tmp_path, "olmoe", "gold", "wikitext2")
    with pytest.raises(ValueError, match="would overwrite"):
        _check_results_corpus(_cfg(tmp_path, "c4"))


def test_guard_ignores_other_models(tmp_path):
    """Each model has its own subtree; qwen's corpus says nothing about olmoe's."""
    _write_run(tmp_path, "qwen", "gold", "wikitext2")
    _check_results_corpus(_cfg(tmp_path, "c4"))


def test_guard_tolerates_an_unreadable_metrics_file(tmp_path):
    """A half-written file from a preempted job must not block the restart."""
    run_dir = tmp_path / "olmoe" / "gold"
    run_dir.mkdir(parents=True)
    (run_dir / METRICS).write_text("{not json")
    _check_results_corpus(_cfg(tmp_path, "c4"))


def test_guard_tolerates_metrics_without_a_corpus(tmp_path):
    _write_run(tmp_path, "olmoe", "gold", "wikitext2")
    (tmp_path / "olmoe" / "gold" / METRICS).write_text(json.dumps({"dataset": {}}))
    _check_results_corpus(_cfg(tmp_path, "c4"))
