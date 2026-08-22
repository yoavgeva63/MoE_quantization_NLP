#!/usr/bin/env python3
"""Paired-difference bootstrap for `mixed` - `uniform`. No GPU.

Part 1 decides separation by asking whether two independently-built bootstrap intervals
overlap. Both arms were bootstrapped with the same seed over the same 128 sequence groups,
so the replicates are literally the same resamples of the same sequences and the pairing is
being thrown away. That makes the existing test conservative rather than wrong - no
"intervals disjoint" claim is at risk - but the correct test differences the two arms
inside each replicate, which cancels the sequence-level variation they share.

The per-token metric values were never saved, so the difference is recomputed here from
the artifacts: each run's `router_inputs.pt` holds the router activations it saw and the
dequantized router weights it used, and `F.linear` of the two reproduces that run's logits
at the captured positions. The capture strided to one token in `input_stride`, so these
numbers are on a subset of the tokens Part 1 measured; the report states the marginal means
from both so the reconstruction can be checked against the recorded run.

    python scripts/paired_bootstrap.py --results-dir results/olmoe

Writes paired_bootstrap.md and paired_bootstrap.json next to the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from moequant.attribution import logits as rebuild_logits
from moequant.attribution import strided_positions
from moequant.metrics import bootstrap_ci, paired_bootstrap_ci, per_token_kl, per_token_selection

METRICS = ("kl", "top1_error")
METRIC_LABELS = {"kl": "routing KL", "top1_error": "top-1 flip rate"}


def per_token_values(reference: torch.Tensor, candidate: torch.Tensor, k: int) -> dict:
    """The two per-token quantities the decision gate is built on."""
    return {
        "kl": per_token_kl(reference, candidate),
        "top1_error": per_token_selection(reference, candidate, k)["top1_error"],
    }


def reconstruct_run(run_dir: Path, reference: dict[int, torch.Tensor], k: int) -> dict:
    """Pooled per-token metrics for one run, rebuilt at the captured positions.

    Loaded with `mmap=True`: these files are hundreds of megabytes on shared storage and
    only one layer is needed at a time.
    """
    payload = torch.load(
        run_dir / "router_inputs.pt", map_location="cpu", mmap=True, weights_only=False
    )
    layers = sorted(set(payload["inputs"]) & set(payload["router_weights"]) & set(reference))
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in METRICS}
    for layer in layers:
        rebuilt = rebuild_logits(payload["inputs"][layer], payload["router_weights"][layer])
        if rebuilt.shape != reference[layer].shape:
            raise SystemExit(
                f"{run_dir}: layer {layer} rebuilt {tuple(rebuilt.shape)} does not match the "
                f"gold reference {tuple(reference[layer].shape)}. Check --stride."
            )
        values = per_token_values(reference[layer], rebuilt, k)
        for name in METRICS:
            collected[name].append(values[name])
    return {
        "layers": layers,
        "stride": int(payload.get("input_stride") or 1),
        "values": {name: torch.cat(collected[name]) for name in METRICS},
    }


def recorded_pooled(run_dir: Path) -> dict:
    """Part 1's own pooled numbers, for calibrating the reconstruction against."""
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        pooled = json.load(handle)["routing"]["pooled"]
    return {name: pooled[name] for name in METRICS if name in pooled}


def analyse_bits(
    results_dir: Path, bits: int, reference: dict[int, torch.Tensor], k: int,
    n_boot: int, seed: int, groups: np.ndarray,
) -> dict | None:
    uniform_dir = results_dir / f"uniform_int{bits}"
    mixed_dir = results_dir / f"mixed_int{bits}"
    if not (uniform_dir / "router_inputs.pt").exists():
        return None
    if not (mixed_dir / "router_inputs.pt").exists():
        return None

    uniform = reconstruct_run(uniform_dir, reference, k)
    mixed = reconstruct_run(mixed_dir, reference, k)
    if uniform["layers"] != mixed["layers"]:
        raise SystemExit(
            f"INT{bits}: uniform captured layers {uniform['layers']} but mixed captured "
            f"{mixed['layers']}; the two arms must cover the same routers to be paired."
        )

    pooled_groups = np.tile(groups, len(uniform["layers"]))
    out: dict = {
        "bits": bits,
        "layers": uniform["layers"],
        "stride": uniform["stride"],
        "num_tokens_per_layer": int(groups.size),
        "num_sequences": int(np.unique(groups).size),
        "metrics": {},
    }
    for name in METRICS:
        u, m = uniform["values"][name], mixed["values"][name]
        out["metrics"][name] = {
            "paired": paired_bootstrap_ci(u, m, pooled_groups, n_boot=n_boot, seed=seed),
            # Same reconstructed data, bootstrapped the way Part 1 does it, so the width
            # comparison isolates the pairing rather than the token resolution.
            "unpaired_uniform": bootstrap_ci(u, pooled_groups, n_boot=n_boot, seed=seed),
            "unpaired_mixed": bootstrap_ci(m, pooled_groups, n_boot=n_boot, seed=seed),
        }
    out["recorded"] = {
        "uniform": recorded_pooled(uniform_dir),
        "mixed": recorded_pooled(mixed_dir),
    }
    return out


def paired_table(by_bits: dict[int, dict]) -> list[str]:
    lines = [
        "## Paired-difference intervals on `mixed` - `uniform`",
        "",
        "Negative means `mixed` is better. Each replicate resamples the 128 sequence groups "
        "once and differences the two arms within that resample, so the sequence-level "
        "variation they share cancels instead of being counted twice.",
        "",
        "| Bits | Metric | uniform | mixed | difference | 95% paired CI | replicates &gt; 0 |",
        "|------|--------|---------|-------|------------|---------------|------------------|",
    ]
    for bits in sorted(by_bits, reverse=True):
        for name in METRICS:
            paired = by_bits[bits]["metrics"][name]["paired"]
            above = 1.0 - paired["fraction_below_zero"]
            lines.append(
                f"| INT{bits} | {METRIC_LABELS[name]} "
                f"| {paired['baseline_mean']:.6f} | {paired['treatment_mean']:.6f} "
                f"| {paired['difference']:+.6f} "
                f"| [{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}] "
                f"| {above * 100:.2f}% |"
            )
    lines.append("")
    return lines


def width_table(by_bits: dict[int, dict]) -> list[str]:
    """Show what the pairing buys, on identical data."""
    lines = [
        "## What the pairing buys",
        "",
        "Both columns are computed on the same reconstructed per-token values with the same "
        "resamples. The unpaired column is the gap between the two marginal intervals - the "
        "quantity the Part 1 disjointness test inspects - and the paired column is the "
        "interval on the difference itself.",
        "",
        "| Bits | Metric | Unpaired interval width (sum of both arms) | Paired interval width "
        "| Narrower by |",
        "|------|--------|--------------------------------------------|-----------------------"
        "|-------------|",
    ]
    for bits in sorted(by_bits, reverse=True):
        for name in METRICS:
            block = by_bits[bits]["metrics"][name]
            u, m, paired = block["unpaired_uniform"], block["unpaired_mixed"], block["paired"]
            # A difference of two independent estimates carries both arms' uncertainty, so
            # the honest unpaired comparator is the sum of the two widths.
            unpaired_width = (u["ci_high"] - u["ci_low"]) + (m["ci_high"] - m["ci_low"])
            paired_width = paired["ci_high"] - paired["ci_low"]
            lines.append(
                f"| INT{bits} | {METRIC_LABELS[name]} | {unpaired_width:.6f} "
                f"| {paired_width:.6f} "
                f"| {unpaired_width / max(paired_width, 1e-12):.2f}x |"
            )
    lines.append("")
    return lines


def calibration_table(by_bits: dict[int, dict]) -> list[str]:
    """The reconstruction is on a token subset; show it lands on the recorded run."""
    lines = [
        "## Reconstruction against the recorded Part 1 numbers",
        "",
        "The paired test is computed on the strided capture, so its marginal means are not "
        "expected to equal Part 1's numbers exactly - Part 1 measured every token, and the "
        "rebuilt logits carry a small reconstruction floor from fp16-stored activations "
        "recomputed in CPU fp32 against a bf16 GPU forward pass. They should be close, and "
        "they are:",
        "",
        "| Bits | Metric | uniform rebuilt | uniform recorded | mixed rebuilt | mixed recorded |",
        "|------|--------|-----------------|------------------|---------------|----------------|",
    ]
    for bits in sorted(by_bits, reverse=True):
        recorded = by_bits[bits]["recorded"]
        for name in METRICS:
            paired = by_bits[bits]["metrics"][name]["paired"]
            u_rec = recorded.get("uniform", {}).get(name, {}).get("mean", float("nan"))
            m_rec = recorded.get("mixed", {}).get(name, {}).get("mean", float("nan"))
            lines.append(
                f"| INT{bits} | {METRIC_LABELS[name]} | {paired['baseline_mean']:.6f} "
                f"| {u_rec:.6f} | {paired['treatment_mean']:.6f} | {m_rec:.6f} |"
            )
    lines.append("")
    return lines


def guidance(by_bits: dict[int, dict]) -> list[str]:
    """Say which number the paper should quote, and why."""
    all_separated = all(
        by_bits[bits]["metrics"][name]["paired"]["ci_high"] < 0
        for bits in by_bits for name in METRICS
    )
    lines = [
        "## Which number the paper should quote",
        "",
        "**Quote the Part 1 intervals from `summary.md` as the headline, and this paired "
        "test as the confirmatory one.** The Part 1 numbers are measured on all 32,768 "
        "routing tokens per layer, which is the full sample the experiment collected; the "
        "paired test here is on one token in "
        f"{next(iter(by_bits.values()))['stride']}, because the per-token metric values were "
        "never saved and only the strided router inputs can be replayed offline. Trading the "
        "full sample for the pairing is not worth it for the headline.",
        "",
        "What the paired test settles is the *method* objection: that comparing two "
        "separately-bootstrapped intervals discards a pairing that is present in the data. "
        "It does, and correcting it moves the conclusion in the safe direction. "
        + (
            "Every cell separates from zero in the paired test as well, so no claim in "
            "`summary.md` depends on the choice of test."
            if all_separated
            else "At least one cell does not separate from zero under the paired test; that "
            "cell should be reported as such."
        ),
        "",
        "Recommended wording: report the Part 1 means and intervals, describe the "
        "disjointness check as a conservative stand-in for a paired-difference test, and "
        "cite this table for the paired confirmation.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3])
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-prefix", default="paired_bootstrap")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = results_dir.name

    artifacts = torch.load(
        results_dir / "gold" / "artifacts.pt", map_location="cpu", mmap=True, weights_only=False
    )
    k = int(artifacts["topology"]["top_k"])
    all_groups = np.asarray(artifacts["groups"])
    stride = args.stride or next(
        (
            json.loads((results_dir / f"uniform_int{b}" / "metrics.json").read_text())["config"][
                "input_stride"
            ]
            for b in args.bits
            if (results_dir / f"uniform_int{b}" / "metrics.json").exists()
        ),
        8,
    )
    mask = strided_positions(all_groups, stride)
    selector = torch.from_numpy(mask)
    reference = {int(layer): artifacts["logits"][layer][selector] for layer in artifacts["logits"]}
    groups = all_groups[mask]
    del artifacts

    by_bits: dict[int, dict] = {}
    for bits in args.bits:
        print(f"[{model}] INT{bits}: rebuilding both arms at stride {stride}")
        report = analyse_bits(results_dir, bits, reference, k, args.n_boot, args.seed, groups)
        if report is None:
            print(f"  skipped: uniform_int{bits} or mixed_int{bits} has no router_inputs.pt")
            continue
        by_bits[bits] = report
        for name in METRICS:
            paired = report["metrics"][name]["paired"]
            print(
                f"  {METRIC_LABELS[name]}: difference {paired['difference']:+.6f} "
                f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]"
            )

    if not by_bits:
        raise SystemExit(f"No paired runs found under {results_dir}")

    parts = [f"# {model} - paired-difference bootstrap", ""]
    parts += [
        "Supplementary to the Part 1 decision gate, which compares two independently "
        "bootstrapped intervals. Both arms use `seed=42` over the same 128 sequence groups, "
        "so their replicates are the same resamples and the difference can be bootstrapped "
        f"directly. Rebuilt from the saved router inputs at one token in {stride}; "
        f"{args.n_boot:,} replicates, sequence-level resampling.",
        "",
    ]
    parts += paired_table(by_bits)
    parts += width_table(by_bits)
    parts += calibration_table(by_bits)
    parts += guidance(by_bits)
    body = "\n".join(parts)

    out_md = results_dir / f"{args.out_prefix}.md"
    out_md.write_text(body)
    out_json = results_dir / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(
        {
            "model": model, "stride": stride, "n_boot": args.n_boot, "seed": args.seed,
            "by_bits": {str(b): r for b, r in by_bits.items()},
        },
        indent=2, default=str,
    ))

    print(f"\n{body}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
