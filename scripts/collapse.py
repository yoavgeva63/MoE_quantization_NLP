#!/usr/bin/env python3
"""Per-layer expert-collapse tables from results already on disk. No GPU.

The headline expert-usage entropy in `summary.md` is pooled across layers, which cancels
independent per-layer imbalances and reads as a null even where a run really has collapsed.
This script reports per layer instead, at two token resolutions:

* dead and unused expert counts from each run's `metrics.json`, computed on all 32,768
  routing tokens, which is the only resolution at which a zero count is meaningful;
* Gini, max expert share, top-decile load share and KL against uniform, rebuilt from the
  gold `artifacts.pt` and each run's `router_inputs.pt`, which the capture strided to
  1 token in 8.

    python scripts/collapse.py --results-dir results/olmoe

Writes collapse.md, collapse.json and collapse.pdf next to the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from moequant.attribution import strided_positions  # noqa: E402
from moequant.collapse import (  # noqa: E402
    gold_counts_at_stride,
    load_reconstruction,
    tier1_report,
    tier2_report,
)

COLORS = {
    "gold": "#4C72B0",
    "uniform": "#C44E52",
    "mixed": "#55A868",
    "placebo": "#8172B2",
    "attention": "#CCB974",
}
POLICY_ORDER = {"gold": 0, "uniform": 1, "mixed": 2, "placebo": 3, "attention": 4}


def load_runs(results_dir: Path) -> list[dict]:
    """Every run under a model's results directory, with its own name attached."""
    runs = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        with path.open() as handle:
            payload = json.load(handle)
        payload["_run"] = path.parent.name
        payload["_dir"] = path.parent
        runs.append(payload)
    if not runs:
        raise SystemExit(f"No metrics.json found under {results_dir}")
    return runs


def _label(run: dict) -> str:
    bits = run.get("bits")
    return f"{run['policy']} {'BF16' if bits is None else f'INT{bits}'}"


def _sort_key(run: dict) -> tuple[int, int]:
    return (-(run.get("bits") or 99), POLICY_ORDER.get(run["policy"], 9))


# -- Tier 1 -----------------------------------------------------------------------------


def tier1_for_runs(runs: list[dict]) -> dict[str, dict]:
    """Per-layer collapse report for every run, from fields the runs already computed."""
    reports = {}
    for run in sorted(runs, key=_sort_key):
        routing = run.get("routing")
        if not routing:
            continue
        # The gold run's own `cand_usage` is its unquantized load, which is the baseline
        # every candidate should be read against.
        reports[run["_run"]] = {
            "policy": run["policy"],
            "bits": run.get("bits"),
            "label": _label(run),
            **tier1_report(routing, key="cand_usage"),
        }
    return reports


def tier1_table(reports: dict[str, dict]) -> list[str]:
    lines = [
        "## Per-layer expert collapse (full resolution, 32,768 tokens per layer)",
        "",
        "Dead means an expert receiving less than a tenth of its fair share of tokens in "
        "that layer; unused means exactly zero selections. Counts are summed over layers "
        "and expressed against the total expert *slots* (layers x experts), since expert "
        "*j* in one layer is a different expert from expert *j* in another.",
        "",
        "| Run | Dead slots | Unused slots | Layers with an unused expert | Worst-layer "
        "norm. entropy | Worst-layer max/mean load |",
        "|-----|------------|--------------|------------------------------|"
        "--------------------------|---------------------------|",
    ]
    for report in reports.values():
        summary = report["summary"]
        if not summary:
            continue
        lines.append(
            f"| {report['label']} "
            f"| {summary['dead_slots']} / {summary['expert_slots']} "
            f"({summary['dead_fraction'] * 100:.1f}%) "
            f"| {summary['unused_slots']} / {summary['expert_slots']} "
            f"({summary['unused_fraction'] * 100:.2f}%) "
            f"| {summary['layers_with_unused']} of {summary['num_layers']} "
            f"| {summary['worst_layer_entropy']:.4f} (layer "
            f"{summary['worst_layer_entropy_layer']}) "
            f"| {summary['worst_max_over_mean_load']:.2f} (layer "
            f"{summary['worst_max_over_mean_load_layer']}) |"
        )
    lines.append("")
    return lines


def pooled_contrast(runs: list[dict], reports: dict[str, dict]) -> list[str]:
    """Show what the pooled headline number reports next to the per-layer worst case."""
    lines = [
        "## Why the pooled number missed this",
        "",
        "Both columns are the same normalized usage entropy on the same run. The left one "
        "is what `summary.md` quotes; it concatenates every layer's tokens into a single "
        "histogram before measuring, so imbalances pointing in different directions across "
        "layers cancel.",
        "",
        "| Run | Pooled (reported) | Worst layer | Mean over layers |",
        "|-----|-------------------|-------------|------------------|",
    ]
    by_run = {run["_run"]: run for run in runs}
    for name, report in reports.items():
        summary = report["summary"]
        pooled = (
            by_run[name].get("routing", {}).get("pooled", {}).get("cand_usage", {})
            .get("marginal_entropy_normalized")
        )
        if not summary or pooled is None:
            continue
        lines.append(
            f"| {report['label']} | {pooled:.4f} | {summary['worst_layer_entropy']:.4f} "
            f"| {summary['mean_layer_entropy']:.4f} |"
        )
    lines.append("")
    return lines


# -- Tier 2 -----------------------------------------------------------------------------


def tier2_for_runs(results_dir: Path, runs: list[dict], stride_hint: int) -> dict[str, dict]:
    """Rebuild per-expert counts for every run that saved its router inputs.

    Gold is measured on the same strided positions as the candidates so the comparison is
    not confounded by resolution.
    """
    gold_run = next((r for r in runs if r["policy"] == "gold"), None)
    if gold_run is None:
        return {}

    artifacts = torch.load(
        Path(gold_run["_dir"]) / "artifacts.pt", map_location="cpu", mmap=True, weights_only=False
    )
    top_k = int(artifacts["topology"]["top_k"])
    groups = np.asarray(artifacts["groups"])

    reports: dict[str, dict] = {}
    mask = strided_positions(groups, stride_hint)
    reports[gold_run["_run"]] = {
        "policy": "gold",
        "bits": None,
        "label": _label(gold_run),
        **tier2_report(gold_counts_at_stride(artifacts["logits"], mask, top_k), stride_hint),
    }
    del artifacts

    for run in sorted(runs, key=_sort_key):
        if run["policy"] == "gold":
            continue
        if not (Path(run["_dir"]) / "router_inputs.pt").exists():
            continue
        recon = load_reconstruction(run["_dir"], top_k)
        counts = {layer: recon.counts(layer) for layer in recon.layers}
        reports[run["_run"]] = {
            "policy": run["policy"],
            "bits": run.get("bits"),
            "label": _label(run),
            **tier2_report(counts, recon.stride),
        }
        del recon, counts
    return reports


def tier2_table(reports: dict[str, dict]) -> list[str]:
    lines = [
        "## Load-distribution shape (strided capture, 1 token in 8)",
        "",
        "These are the sensitive instruments. Normalized entropy is maximal at balance, so "
        "its gradient there is zero; Gini and KL against uniform are not. `max share` is "
        "the busiest expert's load as a multiple of fair share, `top-10%` the fraction of "
        "routing slots the busiest tenth of experts absorbs (0.100 under perfect balance).",
        "",
        "Gold is measured on the same strided positions, so every row is comparable. Dead "
        "and unused counts are **not** reported here - at 1 token in 8 the zero counts are "
        "inflated, which is exactly why the table above uses the full-resolution fields.",
        "",
        "| Run | Gini (mean / max) | Max share | Top-10% share | KL(load &#124;&#124; "
        "uniform) mean |",
        "|-----|-------------------|-----------|---------------|"
        "----------------------------|",
    ]
    for report in reports.values():
        summary = report["summary"]
        if not summary:
            continue
        lines.append(
            f"| {report['label']} "
            f"| {summary['gini_mean']:.3f} / {summary['gini_max']:.3f} "
            f"| {summary['max_share_max']:.2f}x "
            f"| {summary['top_decile_share_mean']:.3f} "
            f"| {summary['kl_vs_uniform_mean']:.4f} |"
        )
    lines.append("")
    return lines


def resolution_caveat(
    tier1: dict[str, dict], tier2: dict[str, dict], stride: int
) -> list[str]:
    """Quantify the stride's effect on zero counts, so the split is not just asserted."""
    lines = [
        "## Why dead counts and ratio metrics come from different resolutions",
        "",
        f"The candidate router inputs were captured every {stride}th token, so the "
        "reconstruction sees one token in "
        f"{stride}. Ratio statistics are unaffected, but a smaller sample makes low counts "
        "hit zero for reasons that have nothing to do with quantization. Measured on the "
        "same runs:",
        "",
        "| Run | Dead slots, full resolution | Dead slots, strided | Unused slots, full "
        "| Unused slots, strided |",
        "|-----|-----------------------------|---------------------|--------------------"
        "|-----------------------|",
    ]
    for name, report in tier1.items():
        summary = report["summary"]
        strided = tier2.get(name, {}).get("summary")
        if not summary or not strided:
            continue
        lines.append(
            f"| {report['label']} | {summary['dead_slots']} | {strided['strided_dead_slots']} "
            f"| {summary['unused_slots']} | {strided['strided_unused_slots']} |"
        )
    lines += [
        "",
        "The inflation is systematic and affects gold as much as the candidates, so the "
        "full-resolution fields are the ones quoted for dead and unused experts and the "
        "reconstruction is used only for the shape statistics.",
        "",
    ]
    return lines


# -- verdict ----------------------------------------------------------------------------


def verdict(model: str, tier1: dict[str, dict], tier2: dict[str, dict]) -> list[str]:
    """State plainly whether collapse happened and whether protecting the routers helped.

    "Collapse" is read off the full-resolution fields: experts pushed below a tenth of
    fair share, and in the severe case experts receiving nothing at all.
    """
    lines = [f"## Did collapse occur on {model}?", ""]
    bit_widths = sorted(
        {r["bits"] for r in tier1.values() if r["bits"] is not None}, reverse=True
    )
    gold = next((r["summary"] for r in tier1.values() if r["policy"] == "gold"), {})
    if gold:
        lines += [
            f"Baseline: the unquantized model already leaves {gold['dead_slots']} of "
            f"{gold['expert_slots']} expert slots below a tenth of fair share "
            f"({gold['dead_fraction'] * 100:.1f}%), with "
            f"{gold['unused_slots']} receiving nothing. Natural routers are not balanced, "
            "so a candidate is only collapsing if it is worse than this.",
            "",
        ]

    for bits in bit_widths:
        by_policy = {
            r["policy"]: r for r in tier1.values() if r["bits"] == bits
        }
        if not {"uniform", "mixed"} <= set(by_policy):
            continue
        u, m = by_policy["uniform"]["summary"], by_policy["mixed"]["summary"]
        u2 = next(
            (v["summary"] for k, v in tier2.items()
             if v["policy"] == "uniform" and v["bits"] == bits), {}
        )
        m2 = next(
            (v["summary"] for k, v in tier2.items()
             if v["policy"] == "mixed" and v["bits"] == bits), {}
        )

        collapsed = (
            u["dead_fraction"] > 2 * max(gold.get("dead_fraction", 0.0), 0.01)
            or u["layers_with_unused"] > gold.get("layers_with_unused", 0)
        )
        mitigated = collapsed and (
            m["dead_slots"] < u["dead_slots"] and m["layers_with_unused"] <= u["layers_with_unused"]
        )

        headline = (
            "**collapse**" if collapsed else "no collapse beyond the unquantized baseline"
        )
        lines.append(f"- **INT{bits}: {headline} under `uniform`.** ")
        lines.append(
            f"  Dead slots {u['dead_slots']}/{u['expert_slots']} "
            f"({u['dead_fraction'] * 100:.1f}%) under `uniform` vs "
            f"{m['dead_slots']}/{m['expert_slots']} ({m['dead_fraction'] * 100:.1f}%) under "
            f"`mixed`; layers containing a fully unused expert "
            f"{u['layers_with_unused']} vs {m['layers_with_unused']}; worst-layer entropy "
            f"{u['worst_layer_entropy']:.4f} vs {m['worst_layer_entropy']:.4f}; worst-layer "
            f"max/mean load {u['worst_max_over_mean_load']:.2f} vs "
            f"{m['worst_max_over_mean_load']:.2f}."
        )
        if u2 and m2:
            lines.append(
                f"  Load-distribution KL against uniform (strided) "
                f"{u2['kl_vs_uniform_mean']:.4f} under `uniform` vs "
                f"{m2['kl_vs_uniform_mean']:.4f} under `mixed`, a "
                f"{u2['kl_vs_uniform_mean'] / max(m2['kl_vs_uniform_mean'], 1e-12):.2f}x "
                f"reduction; Gini {u2['gini_mean']:.3f} vs {m2['gini_mean']:.3f}."
            )
        if collapsed:
            lines.append(
                "  Router protection **mitigated** it." if mitigated
                else "  Router protection did **not** mitigate it."
            )
        else:
            lines.append(
                "  Nothing to mitigate at this precision; the comparison is a null and the "
                "boundary condition is worth reporting as such."
            )
    lines.append("")
    return lines


# -- figures ----------------------------------------------------------------------------


def plot_dead_by_layer(tier1: dict[str, dict], pdf: PdfPages, model: str) -> None:
    bit_widths = sorted({r["bits"] for r in tier1.values() if r["bits"] is not None}, reverse=True)
    for bits in bit_widths:
        selected = {
            r["policy"]: r for r in tier1.values() if r["bits"] in (bits, None)
        }
        if len(selected) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, report in sorted(selected.items(), key=lambda kv: POLICY_ORDER.get(kv[0], 9)):
            rows = report["per_layer"]
            ax.plot(
                [r["layer"] for r in rows],
                [r["dead_experts"] for r in rows],
                marker="o", label=report["label"], color=COLORS.get(policy),
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Experts below a tenth of fair share")
        ax.set_title(f"{model}: starved experts by layer at INT{bits}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_entropy_by_layer(tier1: dict[str, dict], pdf: PdfPages, model: str) -> None:
    bit_widths = sorted({r["bits"] for r in tier1.values() if r["bits"] is not None}, reverse=True)
    for bits in bit_widths:
        selected = {r["policy"]: r for r in tier1.values() if r["bits"] in (bits, None)}
        if len(selected) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, report in sorted(selected.items(), key=lambda kv: POLICY_ORDER.get(kv[0], 9)):
            rows = report["per_layer"]
            ax.plot(
                [r["layer"] for r in rows],
                [r["marginal_entropy_normalized"] for r in rows],
                marker="o", label=report["label"], color=COLORS.get(policy),
            )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized expert-usage entropy")
        ax.set_title(f"{model}: per-layer usage entropy at INT{bits}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_gini(tier2: dict[str, dict], pdf: PdfPages, model: str) -> None:
    if not tier2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bit_widths = sorted({r["bits"] for r in tier2.values() if r["bits"] is not None}, reverse=True)
    position = {b: i for i, b in enumerate(bit_widths)}
    plotted = False
    for policy in ("uniform", "mixed", "placebo", "attention"):
        points = [
            (position[r["bits"]], r["summary"]["gini_mean"])
            for r in tier2.values()
            if r["policy"] == policy and r["bits"] in position and r["summary"]
        ]
        if not points:
            continue
        points.sort()
        ax.plot(*zip(*points), marker="o", label=policy, color=COLORS.get(policy), linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    gold = next((r["summary"]["gini_mean"] for r in tier2.values() if r["policy"] == "gold"), None)
    if gold is not None:
        ax.axhline(gold, linestyle="--", color=COLORS["gold"], label="gold (BF16)")
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([f"INT{b}" for b in bit_widths])
    ax.set_xlabel("Expert precision")
    ax.set_ylabel("Mean per-layer Gini of expert load")
    ax.set_title(f"{model}: expert-load concentration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Capture stride for the gold logits. Defaults to the runs' input_stride.",
    )
    parser.add_argument(
        "--skip-reconstruction", action="store_true",
        help="Tier 1 only. Use when the .pt artifacts are unavailable or too slow to read.",
    )
    parser.add_argument("--out-prefix", default="collapse")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = results_dir.name
    runs = load_runs(results_dir)

    stride = args.stride or next(
        (r["config"]["input_stride"] for r in runs if r.get("config", {}).get("input_stride")), 8
    )

    print(f"[{model}] Tier 1: reading per-layer usage from {len(runs)} runs")
    tier1 = tier1_for_runs(runs)

    tier2: dict[str, dict] = {}
    if not args.skip_reconstruction:
        print(f"[{model}] Tier 2: rebuilding per-expert counts at stride {stride}")
        tier2 = tier2_for_runs(results_dir, runs, stride)

    parts = [f"# {model} - expert collapse", ""]
    parts += [
        "Per-layer expert-load balance for every run, from data already on disk. The "
        "pooled entropy in `summary.md` concatenates all layers into one histogram before "
        "measuring, which cancels independent per-layer imbalances; everything below is "
        "resolved per layer.",
        "",
    ]
    parts += verdict(model, tier1, tier2)
    parts += tier1_table(tier1)
    parts += pooled_contrast(runs, tier1)
    if tier2:
        parts += tier2_table(tier2)
        parts += resolution_caveat(tier1, tier2, stride)
    body = "\n".join(parts)

    out_md = results_dir / f"{args.out_prefix}.md"
    out_md.write_text(body)

    out_json = results_dir / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(
        {
            "model": model,
            "stride": stride,
            "full_resolution": tier1,
            "strided_reconstruction": tier2,
            "notes": {
                "dead_and_unused": "from metrics.json, all 32,768 routing tokens per layer",
                "shape_statistics": f"rebuilt from artifacts, 1 token in {stride}",
            },
        },
        indent=2, default=str,
    ))

    out_pdf = results_dir / f"{args.out_prefix}.pdf"
    with PdfPages(out_pdf) as pdf:
        plot_dead_by_layer(tier1, pdf, model)
        plot_entropy_by_layer(tier1, pdf, model)
        plot_gini(tier2, pdf, model)

    print(f"\n{body}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
