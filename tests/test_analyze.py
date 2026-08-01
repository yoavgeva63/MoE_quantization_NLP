"""Figure and decision-gate construction in scripts/analyze.py.

The bit-width axis is the part worth pinning down. Each policy is drawn as its own
series, so if the x positions are derived per policy rather than from the full set of
bit-widths, a single missing run - an INT2 job that ran out of memory, say - slides that
policy's remaining points onto the wrong ticks. The resulting figure looks entirely
normal and reverses the apparent trend.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def analyze():
    spec = importlib.util.spec_from_file_location(
        "analyze_mod", REPO_ROOT / "scripts" / "analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_mod"] = module
    spec.loader.exec_module(module)
    return module


def _run(policy, bits, kl):
    return {
        "policy": policy,
        "bits": bits,
        "lm": {"perplexity": 10.0 + (bits or 0)},
        "routing": {"pooled": {"kl": {"mean": kl, "ci_low": kl - 0.01, "ci_high": kl + 0.01}}},
    }


class _RecordingAxis:
    """Captures what would have been drawn, so we can assert on coordinates."""

    def __init__(self):
        self.series = {}
        self.xticks = None
        self.xticklabels = None

    def errorbar(self, x, y, label=None, **kw):
        self.series[label] = list(zip(x, y))

    def set_xticks(self, ticks):
        self.xticks = list(ticks)

    def set_xticklabels(self, labels):
        self.xticklabels = list(labels)

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _draw(analyze, runs, monkeypatch):
    ax = _RecordingAxis()
    monkeypatch.setattr(analyze.plt, "subplots", lambda **kw: (_Fig(), ax))
    monkeypatch.setattr(analyze.plt, "close", lambda *a, **k: None)
    analyze.plot_metric(
        runs, lambda r: r["routing"]["pooled"]["kl"], "kl", "title", _Pdf(), show_gold=False
    )
    return ax


class _Fig:
    def tight_layout(self):
        pass


class _Pdf:
    def savefig(self, fig):
        pass


def test_all_policies_complete_share_the_axis(analyze, monkeypatch):
    runs = [_run("uniform", b, b * 0.1) for b in (8, 4, 2)]
    runs += [_run("mixed", b, b * 0.05) for b in (8, 4, 2)]

    ax = _draw(analyze, runs, monkeypatch)
    assert ax.xticklabels == ["INT8", "INT4", "INT2"]
    assert ax.series["uniform"] == [(0, 0.8), (1, 0.4), (2, 0.2)]
    assert ax.series["mixed"] == [(0, 0.4), (1, 0.2), (2, 0.1)]


def test_missing_run_does_not_shift_a_policy_onto_the_wrong_precision(analyze, monkeypatch):
    """The regression: mixed has no INT8 run, so its INT4 point must stay at x=1."""
    runs = [_run("uniform", b, b * 0.1) for b in (8, 4, 2)]
    runs += [_run("mixed", b, b * 0.05) for b in (4, 2)]  # INT8 mixed run missing

    ax = _draw(analyze, runs, monkeypatch)

    assert ax.xticklabels == ["INT8", "INT4", "INT2"]
    assert ax.series["uniform"] == [(0, 0.8), (1, 0.4), (2, 0.2)]
    assert ax.series["mixed"] == [(1, 0.2), (2, 0.1)]

    # Both policies' points at any shared x must describe the same bit-width.
    for x, _ in ax.series["mixed"]:
        assert x in dict(ax.series["uniform"])


def test_axis_labels_are_not_overwritten_by_the_last_policy(analyze, monkeypatch):
    runs = [_run("uniform", b, 0.1) for b in (8, 4, 2)]
    runs += [_run("mixed", 4, 0.05)]  # single point, drawn last

    ax = _draw(analyze, runs, monkeypatch)
    assert ax.xticklabels == ["INT8", "INT4", "INT2"]
    assert ax.series["mixed"] == [(1, 0.05)]


# -- decision gate ---------------------------------------------------------------------


def _gated(mixed_kl, uniform_kl, spread=0.01):
    def mk(policy, kl):
        run = _run(policy, 4, kl)
        run["routing"]["pooled"]["kl"] = {
            "mean": kl, "ci_low": kl - spread, "ci_high": kl + spread
        }
        run["routing"]["pooled"]["top1_error"] = {
            "mean": kl, "ci_low": kl - spread, "ci_high": kl + spread
        }
        return run

    return [mk("uniform", uniform_kl), mk("mixed", mixed_kl)]


def test_gate_declares_a_win_only_when_intervals_separate(analyze):
    lines = "\n".join(analyze.decision_gate(_gated(mixed_kl=0.10, uniform_kl=0.50)))
    assert "MIXED WINS" in lines
    assert "real advantage for router" in lines


def test_gate_reports_no_separation_when_intervals_overlap(analyze):
    lines = "\n".join(analyze.decision_gate(_gated(mixed_kl=0.49, uniform_kl=0.50)))
    assert "no separation" in lines
    assert "MIXED WINS" not in lines
    assert "proceed to the Part 2 attribution" in lines


def test_gate_calls_out_mixed_losing(analyze):
    lines = "\n".join(analyze.decision_gate(_gated(mixed_kl=0.50, uniform_kl=0.10)))
    assert "MIXED LOSES" in lines
