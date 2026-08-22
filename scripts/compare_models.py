#!/usr/bin/env python3
"""Cross-model figures and tables: both architectures on one axis.

`analyze.py` reports one model at a time, which is the right unit for the decision gate
but the wrong unit for the paper. The claim is that router protection helps on *MoE
models*, not on OLMoE, so the headline figure has to show both architectures together and
let the reader see that the trend repeats.

    python scripts/compare_models.py --results-dir results/olmoe results/qwen

Writes headline.pdf, comparison.pdf and comparison.md into the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

# The placebo lands on top of uniform - that is the control working - so it is drawn first
# as a wide translucent underlay. Plotted as a normal line it would simply hide uniform,
# and the reader would see one curve where the point is that two curves coincide.
POLICY_STYLE = {
    "placebo": {
        "color": "#8172B2", "marker": "", "linewidth": 6.5, "alpha": 0.5, "zorder": 1,
        "label": "placebo (random 0.02% protected)",
    },
    "uniform": {
        "color": "#C44E52", "marker": "o", "linewidth": 1.8, "zorder": 3,
        "label": "uniform (router quantized)",
    },
    "mixed": {
        "color": "#55A868", "marker": "s", "linewidth": 1.8, "zorder": 3,
        "label": "mixed (router in BF16)",
    },
}
MODEL_LINESTYLE = {0: "-", 1: "--", 2: ":"}

# Display names, since "olmoe" in a directory listing is not what belongs in a caption.
PRETTY = {
    "olmoe": "OLMoE-1B-7B",
    "qwen": "Qwen1.5-MoE-A2.7B",
}


def load_model(results_dir: Path) -> dict:
    runs = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        with path.open() as handle:
            runs.append(json.load(handle))
    if not runs:
        raise SystemExit(f"No metrics.json found under {results_dir}")
    return {"name": results_dir.name, "label": PRETTY.get(results_dir.name, results_dir.name),
            "runs": runs}


def _series(runs: list[dict], policy: str, getter):
    rows = []
    for run in runs:
        if run.get("policy") != policy or run.get("bits") is None:
            continue
        try:
            value = getter(run)
        except (KeyError, TypeError):
            continue
        if value is None:
            continue
        if isinstance(value, dict):
            rows.append((run["bits"], value["mean"], value["ci_low"], value["ci_high"]))
        else:
            rows.append((run["bits"], value, value, value))
    rows.sort(key=lambda row: -row[0])
    return rows


def plot_panel(ax, models: list[dict], getter, ylabel: str, logy: bool = False) -> None:
    all_bits = sorted(
        {run["bits"] for model in models for run in model["runs"] if run.get("bits")},
        reverse=True,
    )
    position = {bits: index for index, bits in enumerate(all_bits)}

    for model_index, model in enumerate(models):
        for policy, style in POLICY_STYLE.items():
            rows = _series(model["runs"], policy, getter)
            if not rows:
                continue
            bits = [position[row[0]] for row in rows]
            mean = [row[1] for row in rows]
            yerr = [
                [row[1] - row[2] for row in rows],
                [row[3] - row[1] for row in rows],
            ]
            ax.errorbar(
                bits, mean, yerr=yerr, capsize=3,
                color=style["color"], marker=style["marker"],
                linewidth=style.get("linewidth", 1.8),
                alpha=style.get("alpha", 1.0),
                zorder=style.get("zorder", 2),
                linestyle=MODEL_LINESTYLE.get(model_index, "-"),
                label=f"{model['label']} - {policy}",
            )
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([f"INT{bits}" for bits in all_bits])
    ax.set_xlabel("Expert precision")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def headline_figure(models: list[dict], out_pdf: Path) -> None:
    """The two-panel figure for the front of the paper: routing drift and expert changes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_panel(axes[0], models, lambda r: r["routing"]["pooled"]["kl"],
               "Routing KL(gold || candidate)", logy=True)
    axes[0].set_title("Routing drift")
    plot_panel(axes[1], models, lambda r: r["routing"]["pooled"]["top1_error"],
               "Top-1 expert flip rate")
    axes[1].set_title("Top-1 expert changes")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Protecting the router alone, across two MoE architectures")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_pdf)
    plt.close(fig)


def detail_figures(models: list[dict], out_pdf: Path) -> None:
    panels = [
        (lambda r: r["lm"]["perplexity"], "WikiText-2 perplexity", True),
        (lambda r: r["routing"]["pooled"]["jaccard_distance"], "Top-k Jaccard distance", False),
        (lambda r: r["lm"].get("output_kl_topm"), "Output KL (top-M)", False),
        (
            lambda r: r["routing"]["pooled"]["cand_usage"]["marginal_entropy_normalized"],
            "Normalized expert-usage entropy",
            False,
        ),
    ]
    with PdfPages(out_pdf) as pdf:
        for getter, ylabel, logy in panels:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plot_panel(ax, models, getter, ylabel, logy=logy)
            ax.legend(fontsize=7)
            ax.set_title(ylabel)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def comparison_table(models: list[dict]) -> list[str]:
    lines = [
        "# Cross-model comparison",
        "",
        "| Model | Policy | Bits | PPL | Routing KL | Top-1 flip |",
        "|-------|--------|------|-----|------------|------------|",
    ]
    order = {"gold": 0, "uniform": 1, "mixed": 2, "placebo": 3, "attention": 4}
    for model in models:
        for run in sorted(
            model["runs"], key=lambda r: (-(r.get("bits") or 99), order.get(r["policy"], 9))
        ):
            pooled = run.get("routing", {}).get("pooled", {})
            kl = pooled.get("kl", {}).get("mean")
            top1 = pooled.get("top1_error", {}).get("mean")
            bits = run.get("bits")
            lines.append(
                f"| {model['label']} | {run['policy']} "
                f"| {'BF16' if bits is None else f'INT{bits}'} "
                f"| {run['lm']['perplexity']:.4f} "
                f"| {'-' if kl is None else f'{kl:.6f}'} "
                f"| {'-' if top1 is None else f'{top1 * 100:.2f}%'} |"
            )
    lines.append("")
    return lines


def reproduction_gap(models: list[dict]) -> list[str]:
    """Does the effect replicate across architectures at every bit-width?"""
    lines = ["## Does the effect replicate?", ""]
    for model in models:
        for bits in sorted(
            {run["bits"] for run in model["runs"] if run.get("bits")}, reverse=True
        ):
            by_policy = {
                run["policy"]: run for run in model["runs"] if run.get("bits") == bits
            }
            if not {"uniform", "mixed"} <= set(by_policy):
                continue
            uniform = by_policy["uniform"]["routing"]["pooled"]["top1_error"]
            mixed = by_policy["mixed"]["routing"]["pooled"]["top1_error"]
            reduction = 1.0 - mixed["mean"] / uniform["mean"] if uniform["mean"] else float("nan")
            separated = mixed["ci_high"] < uniform["ci_low"] or uniform["ci_high"] < mixed["ci_low"]
            placebo = by_policy.get("placebo")
            placebo_note = ""
            if placebo is not None:
                placebo_flip = placebo["routing"]["pooled"]["top1_error"]["mean"]
                placebo_reduction = (
                    1.0 - placebo_flip / uniform["mean"] if uniform["mean"] else float("nan")
                )
                placebo_note = f", placebo {placebo_reduction * 100:+.1f}%"
            lines.append(
                f"- {model['label']} INT{bits}: mixed cuts top-1 flips by "
                f"{reduction * 100:.1f}%{placebo_note} "
                f"({'intervals disjoint' if separated else 'NOT separated'})"
            )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/paper")
    args = parser.parse_args()

    models = [load_model(Path(path)) for path in args.results_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headline_figure(models, out_dir / "headline.pdf")
    detail_figures(models, out_dir / "comparison.pdf")
    body = "\n".join(comparison_table(models) + reproduction_gap(models))
    (out_dir / "comparison.md").write_text(body)

    print(body)
    print(f"Wrote {out_dir / 'headline.pdf'}")
    print(f"Wrote {out_dir / 'comparison.pdf'}")
    print(f"Wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
