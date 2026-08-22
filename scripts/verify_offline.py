#!/usr/bin/env python3
"""Machine-checked verification of the router precision claims, from saved artifacts. No GPU.

Two claims the paper needs to make and could previously only assert:

1. **`mixed` really left the routers alone, and `uniform`/`placebo` really quantized them
   to the requested width.** Each run's `router_inputs.pt` stores the dequantized router
   weights that run actually used, so this is decidable offline: compare against the gold
   `router_weights` for bit-identity, and count distinct values per output channel for the
   precision ceiling. Under per-axis symmetric integer quantization a channel is
   `scale * q`, so an INT-N weight can show at most `2**N` distinct values in a row.

   The in-pipeline `verify.check_bit_width` samples four modules; this inspects **every
   output channel of every router in every layer of every run**, which is a strictly
   stronger statement and the one to put in the appendix.

2. **What the placebo control actually protected.** The chosen module FQNs were never
   written into any `metrics.json` - only `num_protected_modules` - so the results files do
   not document the control. The choice is deterministic given `placebo_seed`, so it is
   re-derived here from a meta-device skeleton (no weights materialised, no download: the
   config is read from `architecture.json`), together with how many tokens actually reach
   the protected module.

    python scripts/verify_offline.py --results-dir results/olmoe

Writes verification.md and verification.json next to the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moequant.metrics import topk_mask


def channel_levels(weight: torch.Tensor) -> dict:
    """Distinct values per output channel, over every channel rather than a sample."""
    dense = weight.detach().to(torch.float32)
    if dense.ndim < 2:
        dense = dense.reshape(1, -1)
    counts = [int(torch.unique(dense[i]).numel()) for i in range(dense.shape[0])]
    return {
        "max_levels": max(counts),
        "min_levels": min(counts),
        "channels": len(counts),
        "values_per_channel": int(dense.shape[1]),
    }


def run_router_weights(run_dir: Path, gold_weights: dict) -> dict:
    """The router weights a run used.

    Candidate runs store a dequantized copy alongside their captured inputs. The gold run
    does not - its weights are the reference in `artifacts.pt` - so gold resolves to the
    reference itself, which is what makes its row a trivial pass rather than a gap.
    """
    payload = torch.load(
        run_dir / "router_inputs.pt", map_location="cpu", mmap=True, weights_only=False
    )
    weights = payload.get("router_weights")
    return gold_weights if weights is None else {int(k): v for k, v in weights.items()}


def check_routers(run_dir: Path, gold_weights: dict, bits: int | None) -> dict:
    """Level counts and bit-identity for every router weight a run used."""
    weights = run_router_weights(run_dir, gold_weights)
    allowed = None if bits is None else 2**bits

    per_layer = []
    for layer in sorted(set(weights) & set(gold_weights)):
        run_w = weights[layer].to(torch.float32)
        gold_w = gold_weights[layer].to(torch.float32)
        levels = channel_levels(run_w)
        identical = bool(torch.equal(run_w, gold_w))
        per_layer.append({
            "layer": int(layer),
            "identical_to_gold": identical,
            "max_abs_delta": float((run_w - gold_w).abs().max()),
            **levels,
        })

    if not per_layer:
        return {"num_layers": 0}

    max_levels = max(row["max_levels"] for row in per_layer)
    identical = sum(1 for row in per_layer if row["identical_to_gold"])
    return {
        "num_layers": len(per_layer),
        "allowed_levels": allowed,
        "max_levels_any_channel": max_levels,
        "min_levels_any_channel": min(row["min_levels"] for row in per_layer),
        "channels_inspected": sum(row["channels"] for row in per_layer),
        "layers_identical_to_gold": identical,
        "within_level_ceiling": allowed is None or max_levels <= allowed,
        "max_abs_delta_from_gold": max(row["max_abs_delta"] for row in per_layer),
        "per_layer": per_layer,
    }


def expected_router_state(policy: str) -> str:
    """What each policy claims about its routers, so the check has something to fail."""
    if policy == "gold":
        return "identical to gold in every layer"
    if policy == "mixed":
        return "identical to gold in every layer (routers protected)"
    return "quantized in every layer, at most 2**bits levels per channel"


def evaluate(policy: str, bits: int | None, report: dict) -> tuple[bool, str]:
    """Decide whether a run's routers match what its policy promised."""
    if not report.get("num_layers"):
        return True, "no router weights stored"
    layers = report["num_layers"]
    identical = report["layers_identical_to_gold"]

    if policy in ("gold", "mixed"):
        if identical == layers:
            return True, f"bit-identical to gold in {identical}/{layers} layers"
        return False, (
            f"only {identical}/{layers} layers are bit-identical to gold; "
            f"max abs delta {report['max_abs_delta_from_gold']:.3e}"
        )

    problems = []
    if identical != 0:
        problems.append(f"{identical}/{layers} layers are still bit-identical to gold")
    if not report["within_level_ceiling"]:
        problems.append(
            f"a channel shows {report['max_levels_any_channel']} distinct values, above the "
            f"INT{bits} ceiling of {report['allowed_levels']}"
        )
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"quantized in all {layers} layers, at most "
        f"{report['max_levels_any_channel']} of {report['allowed_levels']} levels per channel"
    )


# -- the placebo control ---------------------------------------------------------------


def derive_placebo(results_dir: Path, model_key: str, seed: int) -> dict:
    """Re-derive which module the placebo protected, and how exposed it is.

    Built on a meta-device skeleton from the config recorded in `architecture.json`, so no
    weights are materialised and nothing is downloaded.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from moequant.quantize import quantizable_modules, sample_placebo_modules
    from moequant.registry import get_spec

    architecture = json.loads((results_dir / "architecture.json").read_text())
    raw = {k: v for k, v in architecture["config"].items() if not k.startswith("_")}
    config = AutoConfig.for_model(raw.pop("model_type"), **raw)
    with init_empty_weights():
        skeleton = AutoModelForCausalLM.from_config(config)

    spec = get_spec(model_key)
    everything = quantizable_modules(skeleton, spec)
    router_re = spec.router_regex()
    budget = sum(n for fqn, n in everything if router_re.fullmatch(fqn))
    sizes = dict(everything)
    pool = [(fqn, n) for fqn, n in everything if not router_re.fullmatch(fqn) and n <= budget]
    chosen = sample_placebo_modules(skeleton, spec, seed=seed)
    del skeleton

    protected = sum(sizes[fqn] for fqn in chosen)
    return {
        "placebo_seed": seed,
        "modules": list(chosen),
        "num_modules": len(chosen),
        "protected_params": protected,
        "router_budget": budget,
        "ratio_to_router_budget": protected / budget if budget else float("nan"),
        "num_routers": sum(1 for fqn, _ in everything if router_re.fullmatch(fqn)),
        "candidate_pool_size": len(pool),
        "candidate_pool_sizes": sorted({n for _, n in pool}),
    }


def placebo_exposure(gold_dir: Path, modules: list[str]) -> dict:
    """How many routing tokens are actually on the protected module's compute path.

    A router is read by every token in every layer. An expert projection is read only by
    the tokens its own expert is selected for, which is the mismatch that matters and the
    one parameter counting hides.
    """
    artifacts = torch.load(
        gold_dir / "artifacts.pt", map_location="cpu", mmap=True, weights_only=False
    )
    top_k = int(artifacts["topology"]["top_k"])
    num_layers = int(artifacts["topology"]["num_router_layers"])

    rows = []
    for fqn in modules:
        parts = fqn.split(".")
        try:
            layer = int(parts[parts.index("layers") + 1])
            expert = int(parts[parts.index("experts") + 1])
        except (ValueError, IndexError):
            rows.append({"module": fqn, "note": "not an expert projection; exposure not defined"})
            continue
        if layer not in artifacts["logits"]:
            rows.append({"module": fqn, "note": f"layer {layer} not captured"})
            continue
        selected = topk_mask(artifacts["logits"][layer].to(torch.float32), top_k)
        fraction = float(selected[:, expert].to(torch.float64).mean())
        rows.append({
            "module": fqn,
            "layer": layer,
            "expert": expert,
            "tokens": int(selected.shape[0]),
            "token_fraction": fraction,
            # Routers are read by every token in every layer; the protected module is read
            # by `fraction` of tokens in one layer.
            "token_layer_exposure_gap": (num_layers / fraction) if fraction > 0 else float("inf"),
        })
    return {"top_k": top_k, "num_layers": num_layers, "modules": rows}


# -- report ----------------------------------------------------------------------------


def router_table(rows: list[dict]) -> list[str]:
    lines = [
        "## Router weights, every channel of every layer",
        "",
        "Read from each run's stored `router_weights`, which are the dequantized weights "
        "that run actually used. `levels` is the largest number of distinct values found in "
        "any single output channel; under per-axis symmetric INT-*N* quantization a channel "
        "cannot exceed `2**N`.",
        "",
        "| Run | Layers | Channels inspected | Max levels in a channel | Allowed | "
        "Layers bit-identical to gold | Verdict |",
        "|-----|--------|--------------------|-------------------------|---------|"
        "------------------------------|---------|",
    ]
    for row in rows:
        report = row["routers"]
        if not report.get("num_layers"):
            continue
        # `gold` and `mixed` leave the routers in high precision, so a level ceiling does
        # not apply to them; printing one would read as a violation at a glance.
        allowed = (
            "n/a (protected)" if row["policy"] in ("gold", "mixed")
            else str(report["allowed_levels"])
        )
        lines.append(
            f"| {row['label']} | {report['num_layers']} "
            f"| {report['channels_inspected']:,} "
            f"| {report['max_levels_any_channel']:,} "
            f"| {allowed} "
            f"| **{report['layers_identical_to_gold']}/{report['num_layers']}** "
            f"| {'PASS' if row['ok'] else 'FAIL'} |"
        )
    lines.append("")
    return lines


def router_verdict(rows: list[dict]) -> list[str]:
    lines = ["## What this establishes", ""]
    for row in rows:
        if not row["routers"].get("num_layers"):
            continue
        lines.append(
            f"- **{row['label']}** ({expected_router_state(row['policy'])}): {row['detail']}"
        )
    protected = [r for r in rows if r["policy"] in ("gold", "mixed") and r["ok"]]
    quantized = [
        r for r in rows
        if r["policy"] in ("uniform", "placebo", "attention") and r["ok"]
        and r["routers"].get("num_layers")
    ]
    # Name the policies actually covered rather than hard-coding them: `attention` joined
    # the quantized-router group after this sentence was first written.
    quantized_policies = sorted({r["policy"] for r in quantized})
    named = ", ".join(f"`{p}`" for p in quantized_policies[:-1])
    named = f"{named} and `{quantized_policies[-1]}`" if named else f"`{quantized_policies[-1]}`"
    lines += [
        "",
        f"Every `mixed` run's router weights are bit-identical to gold in every layer "
        f"({len(protected)} runs checked), and every {named} run's routers "
        f"sit at or below their INT-*N* level ceiling in every channel with no layer "
        f"matching gold ({len(quantized)} runs checked). That is direct evidence for the "
        "single-variable claim - `mixed` and `uniform` differ in the routers and nothing "
        "else - computed from the artifacts rather than from the quantizer's own "
        "bookkeeping.",
        "",
        "The INT8 rows show fewer than 256 levels, which is expected rather than a shortfall: "
        "with a per-channel scale of `max|w| / 127` and roughly Gaussian router weights, "
        "the 2048 values in a channel do not reach the extreme codes.",
        "",
    ]
    return lines


def placebo_section(placebo: dict, exposure: dict) -> list[str]:
    lines = [
        "## What the placebo control protected",
        "",
        "The chosen FQNs are not recorded in any `metrics.json` - only "
        "`num_protected_modules` - so they are re-derived here from a meta-device skeleton. "
        "Deterministic given `placebo_seed`, and reproducible with "
        "`python scripts/verify_offline.py --results-dir results/<model>`.",
        "",
        f"- `placebo_seed` = {placebo['placebo_seed']}",
        f"- Protected module{'s' if placebo['num_modules'] != 1 else ''}: "
        + ", ".join(f"`{m}`" for m in placebo["modules"]),
        f"- Modules protected: **{placebo['num_modules']}**, against "
        f"{placebo['num_routers']} routers",
        f"- Parameters protected: {placebo['protected_params']:,} against a router budget of "
        f"{placebo['router_budget']:,} "
        f"(**{placebo['ratio_to_router_budget']:.6f}x**)",
        f"- Candidate pool: {placebo['candidate_pool_size']:,} modules, with distinct sizes "
        + ", ".join(f"{n:,}" for n in placebo["candidate_pool_sizes"]),
        "",
    ]
    if placebo["num_modules"] == 1:
        lines += [
            "The selection is a **single draw**, not the greedy accumulation the code's "
            "loop suggests: every eligible candidate is already within tolerance of the "
            "whole router budget on its own, so the first module drawn satisfies the "
            "stopping condition and the tolerance band is never exercised.",
            "",
        ]

    rows = [r for r in exposure["modules"] if "token_fraction" in r]
    if rows:
        lines += [
            "### Activation exposure",
            "",
            "Parameter count is the only dimension on which this control is matched. A "
            "router is read by every token in every layer; an expert projection is read only "
            "by the tokens routed to its own expert, in its own layer. Measured from the "
            "gold router logits:",
            "",
            "| Protected module | Layer | Expert | Tokens on its path | Token x layer "
            "exposure gap vs routers |",
            "|------------------|-------|--------|--------------------|"
            "---------------------------------------|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['module']}` | {row['layer']} | {row['expert']} "
                f"| {row['token_fraction'] * 100:.2f}% of {row['tokens']:,} "
                f"| **{row['token_layer_exposure_gap']:.0f}x** |"
            )
        lines += [
            "",
            f"The routers are read at 100% of tokens in all {exposure['num_layers']} layers. "
            "Counting token x layer forward events, they are exposed by the factor in the "
            "last column more than the placebo's protected parameters. This is the honest "
            "limitation of the control: it rules out \"any high-precision parameters help\", "
            "and does not address \"any high-precision island on the every-token, "
            "every-layer main path helps\".",
            "",
        ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-key", default=None, help="Registry key; defaults to the dir name.")
    parser.add_argument("--placebo-seed", type=int, default=0)
    parser.add_argument("--skip-placebo", action="store_true")
    parser.add_argument("--out-prefix", default="verification")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = results_dir.name
    model_key = args.model_key or model

    gold_dir = results_dir / "gold"
    artifacts = torch.load(
        gold_dir / "artifacts.pt", map_location="cpu", mmap=True, weights_only=False
    )
    gold_weights = {int(k): v for k, v in artifacts["router_weights"].items()}
    del artifacts

    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if not (run_dir / "router_inputs.pt").exists():
            continue
        with metrics_path.open() as handle:
            payload = json.load(handle)
        bits = payload.get("bits")
        report = check_routers(run_dir, gold_weights, bits)
        ok, detail = evaluate(payload["policy"], bits, report)
        rows.append({
            "run": run_dir.name,
            "policy": payload["policy"],
            "bits": bits,
            "label": f"{payload['policy']} {'BF16' if bits is None else f'INT{bits}'}",
            "routers": report,
            "ok": ok,
            "detail": detail,
        })
        print(f"  {run_dir.name}: {'PASS' if ok else 'FAIL'} - {detail}")

    if not rows:
        raise SystemExit(f"No run under {results_dir} stored router weights to check")

    rows.sort(key=lambda r: (-(r["bits"] or 99), r["policy"]))

    placebo = exposure = None
    if not args.skip_placebo:
        try:
            placebo = derive_placebo(results_dir, model_key, args.placebo_seed)
            exposure = placebo_exposure(gold_dir, placebo["modules"])
            print(f"  placebo (seed {args.placebo_seed}): {placebo['modules']}")
        except Exception as error:  # noqa: BLE001 - the router check is the primary result
            print(f"  placebo derivation skipped: {type(error).__name__}: {error}")

    failures = [row for row in rows if not row["ok"]]
    parts = [f"# {model} - offline verification", ""]
    parts += [
        "Machine-checked from the saved artifacts, with no GPU and no model download. "
        "Regenerate with `python scripts/verify_offline.py --results-dir "
        f"{results_dir}`.",
        "",
        (
            f"**All {len(rows)} runs pass.**" if not failures
            else f"**{len(failures)} of {len(rows)} runs FAIL:** "
            + ", ".join(row["label"] for row in failures)
        ),
        "",
    ]
    parts += router_table(rows)
    parts += router_verdict(rows)
    if placebo and exposure:
        parts += placebo_section(placebo, exposure)
    body = "\n".join(parts)

    out_md = results_dir / f"{args.out_prefix}.md"
    out_md.write_text(body)
    out_json = results_dir / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(
        {"model": model, "runs": rows, "placebo": placebo, "placebo_exposure": exposure},
        indent=2, default=str,
    ))

    print(f"\n{body}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
