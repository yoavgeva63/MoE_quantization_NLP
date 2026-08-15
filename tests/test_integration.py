"""End-to-end exercise of the data path without downloading a real checkpoint.

Two tiny models stand in for gold and a quantized candidate. Everything between the
forward pass and the final report - capture, comparison, JSON serialisation, plotting,
and the decision gate - runs for real, so a shape or key mismatch surfaces here rather
than three hours into a cluster job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from moequant.capture import capture_routing
from moequant.config import ExperimentConfig, environment
from moequant.metrics import compare_routing
from moequant.quantize import build_policy
from moequant.registry import parameter_census, resolve_topology
from moequant.runner import (
    _json_default,
    _max_memory,
    _memory_report,
    _print_memory,
    _reset_memory_stats,
)
from moequant.verify import audit
from tests.conftest import TinyMoE
from tests.test_verify import fake_quantize_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def _perturbed_copy(model: TinyMoE, config, bits: int, quantize_routers: bool) -> TinyMoE:
    """A 'quantized' twin, optionally sparing the routers as `mixed` would."""
    import copy

    from torch import nn

    twin = copy.deepcopy(model)
    for fqn, module in twin.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if fqn in ("lm_head",):
            continue
        if fqn.endswith(".mlp.gate") and not quantize_routers:
            continue
        fake_quantize_module(module, bits=bits)
    return twin


def _build_metrics(model, twin, spec, policy, batches, config) -> dict:
    gold = capture_routing(model, spec, batches, "cpu", progress=False)
    cand = capture_routing(twin, spec, batches, "cpu", progress=False)
    topology = resolve_topology(model, spec)

    return {
        "config": {"model_key": "tiny"},
        "environment": environment(),
        "model_id": spec.model_id,
        "policy": policy.name,
        "bits": policy.bits,
        "audit": audit(twin, spec, policy),
        "bit_width_check": {"passed": True, "checked": 3},
        "topology": topology,
        "parameter_census": parameter_census(model, spec),
        "dataset": {"corpus": "synthetic", "total_tokens": 1024},
        "lm": {"perplexity": 12.0 + policy.bits, "top1_agreement": 0.9, "output_kl_topm": 0.01},
        "routing": compare_routing(
            gold["logits"], cand["logits"], topology["top_k"], n_boot=25
        ),
    }


def test_full_pipeline_and_analysis(tmp_path, tiny_model, tiny_spec, tiny_config):
    torch.manual_seed(0)
    batches = [{"input_ids": torch.randint(0, 64, (1, 16))} for _ in range(6)]
    results = tmp_path / "results" / "tiny"

    # -- gold ---------------------------------------------------------------------
    gold_policy = build_policy("gold", tiny_spec, None)
    gold_capture = capture_routing(tiny_model, tiny_spec, batches, "cpu", progress=False)
    topology = resolve_topology(tiny_model, tiny_spec)
    gold_routing = compare_routing(
        gold_capture["logits"], gold_capture["logits"], topology["top_k"], n_boot=25
    )

    # The gold self-check the real runner performs.
    assert abs(gold_routing["pooled"]["kl"]["mean"]) < 1e-9
    assert abs(gold_routing["pooled"]["top1_error"]["mean"]) < 1e-9

    gold_metrics = {
        "config": {"model_key": "tiny"},
        "environment": environment(),
        "policy": "gold",
        "bits": None,
        "audit": audit(tiny_model, tiny_spec, gold_policy),
        "bit_width_check": {"passed": True, "checked": 0},
        "topology": topology,
        "parameter_census": parameter_census(tiny_model, tiny_spec),
        "dataset": {"corpus": "synthetic", "total_tokens": 1024},
        "lm": {"perplexity": 11.0},
        "routing": gold_routing,
        "self_check": {"kl": 0.0, "top1_error": 0.0, "passed": True},
    }
    (results / "gold").mkdir(parents=True)
    (results / "gold" / "metrics.json").write_text(
        json.dumps(gold_metrics, indent=2, default=_json_default)
    )

    # -- uniform and mixed at two bit-widths ----------------------------------------
    for bits in (8, 4):
        for name, quantize_routers in (("uniform", True), ("mixed", False)):
            twin = _perturbed_copy(tiny_model, tiny_config, bits, quantize_routers)
            policy = build_policy(name, tiny_spec, bits)
            payload = _build_metrics(tiny_model, twin, tiny_spec, policy, batches, tiny_config)

            run_dir = results / f"{name}_int{bits}"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2, default=_json_default)
            )

    # -- the substantive check ------------------------------------------------------
    mixed = json.loads((results / "mixed_int4" / "metrics.json").read_text())
    uniform = json.loads((results / "uniform_int4" / "metrics.json").read_text())

    assert mixed["audit"]["num_routers_quantized"] == 0
    assert uniform["audit"]["num_routers_quantized"] == topology["num_router_layers"]

    # Protecting the routers helps, but does not eliminate routing drift.
    mixed_kl = mixed["routing"]["pooled"]["kl"]["mean"]
    uniform_kl = uniform["routing"]["pooled"]["kl"]["mean"]
    assert 0.0 < mixed_kl < uniform_kl

    # That non-zero floor is the entire premise of Part 2, reproduced here in miniature.
    # `mixed` uses bit-identical router weights, so every remaining flip comes from the
    # hidden states arriving at the router having already drifted through quantized
    # attention and expert layers upstream. No amount of router precision removes it.
    assert mixed["routing"]["per_layer"]["0"]["kl"]["mean"] > 0.0

    # -- analysis script ------------------------------------------------------------
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "analyze.py"), "--results-dir", str(results)],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{REPO_ROOT / '.deps'}:{REPO_ROOT / 'src'}:{REPO_ROOT}",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "MPLBACKEND": "Agg",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert (results / "figures.pdf").exists()

    summary = (results / "summary.md").read_text()
    assert "Part 1 decision gate" in summary
    assert "Correctness gates" in summary
    # In this synthetic setup mixed is genuinely perfect, so the gate must say so.
    assert "MIXED WINS" in summary


def test_config_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model_key: olmoe\nrouting_sequences: 32\nseed: 7\n")

    cfg = ExperimentConfig.from_yaml(path, policy="mixed", bits=4)
    assert cfg.model_key == "olmoe"
    assert cfg.routing_sequences == 32
    assert cfg.seed == 7
    assert cfg.run_name == "mixed_int4"
    assert cfg.run_dir.as_posix().endswith("results/olmoe/mixed_int4")
    assert cfg.gold_dir.as_posix().endswith("results/olmoe/gold")


def test_gold_run_name_ignores_bits(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("model_key: olmoe\n")
    assert ExperimentConfig.from_yaml(path, policy="gold").run_name == "gold"


def test_environment_records_versions():
    env = environment()
    assert env["torch"]
    assert "python" in env
    assert "cuda_available" in env


def test_memory_instrumentation_never_raises():
    """A diagnostic must not be able to fail a run.

    `reset_peak_memory_stats` bypasses lazy init and throws on a device whose allocator is
    not up yet, which killed a whole sweep once: gold died before loading and the other six
    runs then failed for want of gold artifacts.
    """
    model = torch.nn.Linear(4, 4)
    for device in ("cpu", "cuda"):
        _reset_memory_stats(device)
        report = _memory_report(model, device)
        _print_memory(report)
        assert set(report) >= {"device_map", "offloaded_modules", "per_device"}


def test_memory_report_is_json_serialisable():
    """It lands in metrics.json, so it has to survive the same encoder as everything else."""
    report = _memory_report(torch.nn.Linear(4, 4), "cpu")
    assert json.loads(json.dumps(report, default=_json_default)) is not None


def test_max_memory_is_none_off_gpu():
    """No CUDA means no budget to impose; `from_pretrained` should see its default."""
    assert _max_memory(ExperimentConfig(model_key="olmoe", device="cpu")) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_max_memory_reserves_headroom_on_every_device():
    """Every visible GPU must be offered, and none of it offered in full.

    Handing accelerate the whole card is what let a quantized model land entirely on GPU 0
    with nothing left for the logits. The budget has to be a strict fraction.
    """
    cfg = ExperimentConfig(model_key="olmoe", device="cuda", gpu_mem_fraction=0.75)
    budget = _max_memory(cfg)

    assert set(budget) == set(range(torch.cuda.device_count()))
    for index, allowed in budget.items():
        total = torch.cuda.get_device_properties(index).total_memory
        assert 0 < allowed < total
        assert allowed == pytest.approx(total * 0.75, rel=1e-6)
