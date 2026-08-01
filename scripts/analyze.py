#!/usr/bin/env python3
"""Turn results JSON into figures, tables, and the Part 1 decision.

The decision gate is the point of this script: does `mixed` beat `uniform` by more than
the bootstrap confidence intervals? It prints a verdict per bit-width rather than leaving
the reader to eyeball overlapping error bars.

    python scripts/analyze.py --results-dir results/olmoe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

COLORS = {
    "gold": "#4C72B0",
    "uniform": "#C44E52",
    "mixed": "#55A868",
    "placebo": "#8172B2",
    "attention": "#CCB974",
}


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        with path.open() as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        runs.append(payload)
    if not runs:
        raise SystemExit(f"No metrics.json found under {results_dir}")
    return runs


def _series(runs: list[dict], policy: str, getter) -> tuple[list[int], list[float], list[float], list[float]]:
    """Extract (bits, mean, ci_low, ci_high) for one policy, sorted by descending bits."""
    rows = []
    for run in runs:
        if run["policy"] != policy or run.get("bits") is None:
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
    rows.sort(key=lambda r: -r[0])
    if not rows:
        return [], [], [], []
    bits, mean, low, high = zip(*rows)
    return list(bits), list(mean), list(low), list(high)


def _gold_value(runs: list[dict], getter):
    for run in runs:
        if run["policy"] == "gold":
            try:
                value = getter(run)
            except (KeyError, TypeError):
                return None
            return value["mean"] if isinstance(value, dict) else value
    return None


def plot_metric(runs, getter, ylabel, title, pdf, logy=False, show_gold=True):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plotted = False

    # One shared axis built from every bit-width present, so a policy with a missing run
    # (an INT2 OOM, say) keeps its remaining points on the correct ticks instead of
    # sliding left onto another policy's precision.
    all_bits = sorted({r["bits"] for r in runs if r.get("bits") is not None}, reverse=True)
    position = {b: i for i, b in enumerate(all_bits)}

    for policy in ("uniform", "mixed", "placebo", "attention"):
        bits, mean, low, high = _series(runs, policy, getter)
        if not bits:
            continue
        x = [position[b] for b in bits]
        yerr = [[m - lo for m, lo in zip(mean, low)], [hi - m for m, hi in zip(mean, high)]]
        ax.errorbar(
            x, mean, yerr=yerr, marker="o", capsize=4,
            label=policy, color=COLORS.get(policy), linewidth=2,
        )
        plotted = True

    ax.set_xticks(list(position.values()))
    ax.set_xticklabels([f"INT{b}" for b in all_bits])

    if show_gold:
        gold = _gold_value(runs, getter)
        if gold is not None:
            ax.axhline(gold, linestyle="--", color=COLORS["gold"], label="gold (BF16)")

    if not plotted:
        plt.close(fig)
        return
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Expert precision")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_layerwise(runs, bits, pdf):
    """Layer-by-layer routing KL at one bit-width."""
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for run in runs:
        if run.get("bits") != bits or run["policy"] == "gold":
            continue
        per_layer = run.get("routing", {}).get("per_layer")
        if not per_layer:
            continue
        layers = sorted(int(k) for k in per_layer)
        values = [per_layer[str(layer)]["kl"]["mean"] for layer in layers]
        ax.plot(layers, values, marker="o", label=run["policy"], color=COLORS.get(run["policy"]))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(gold || candidate)")
    ax.set_title(f"Layer-wise routing drift at INT{bits}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def decision_gate(runs: list[dict]) -> list[str]:
    """Does `mixed` beat `uniform` beyond the confidence intervals?"""
    lines = ["## Part 1 decision gate", ""]
    verdicts: list[bool] = []
    losses: list[bool] = []

    bit_widths = sorted({r["bits"] for r in runs if r.get("bits") is not None}, reverse=True)
    metrics = [
        ("routing KL", lambda r: r["routing"]["pooled"]["kl"], "lower"),
        ("top-1 flip rate", lambda r: r["routing"]["pooled"]["top1_error"], "lower"),
    ]

    for bits in bit_widths:
        by_policy = {r["policy"]: r for r in runs if r.get("bits") == bits}
        if not {"uniform", "mixed"} <= set(by_policy):
            continue
        for label, getter, direction in metrics:
            try:
                u, m = getter(by_policy["uniform"]), getter(by_policy["mixed"])
            except (KeyError, TypeError):
                continue
            # Separation is symmetric: the intervals are disjoint in either direction.
            # Testing only "mixed below uniform" would make a measurably *worse* mixed
            # result indistinguishable from a genuine tie, hiding a real finding.
            separated = m["ci_high"] < u["ci_low"] or u["ci_high"] < m["ci_low"]
            better = m["mean"] < u["mean"] if direction == "lower" else m["mean"] > u["mean"]
            verdict = (
                "MIXED WINS" if separated and better
                else "MIXED LOSES" if separated
                else "no separation"
            )
            verdicts.append(separated and better)
            losses.append(separated and not better)
            lines.append(
                f"- INT{bits} {label}: uniform {u['mean']:.6f} "
                f"[{u['ci_low']:.6f}, {u['ci_high']:.6f}] vs mixed {m['mean']:.6f} "
                f"[{m['ci_low']:.6f}, {m['ci_high']:.6f}] -> **{verdict}**"
            )

    # Perplexity has no bootstrap interval, so it is reported but not used for the verdict.
    for bits in bit_widths:
        by_policy = {r["policy"]: r for r in runs if r.get("bits") == bits}
        if {"uniform", "mixed"} <= set(by_policy):
            u = by_policy["uniform"]["lm"]["perplexity"]
            m = by_policy["mixed"]["lm"]["perplexity"]
            lines.append(f"- INT{bits} perplexity: uniform {u:.4f} vs mixed {m:.4f}")

    lines.append("")
    if any(verdicts):
        lines.append(
            "**Verdict: at least one bit-width shows a real advantage for router "
            "protection.** Run the parameter-matched placebo control before claiming it, "
            "to rule out that any protected 0.02% would do as well."
        )
    elif any(losses):
        lines.append(
            "**Verdict: router protection is measurably worse than uniform quantization "
            "at at least one bit-width.** This is a finding, not a null result: the "
            "intervals are disjoint. Check the placebo and the audit before reporting it, "
            "then proceed to the Part 2 attribution."
        )
    else:
        lines.append(
            "**Verdict: no bit-width shows separation beyond the confidence intervals.** "
            "This is the honest Part 1 answer; proceed to the Part 2 attribution to "
            "explain where the routing damage actually comes from."
        )
    lines.append("")
    return lines


def summary_table(runs: list[dict]) -> list[str]:
    lines = [
        "## Results",
        "",
        "| Policy | Bits | PPL | Routing KL | Top-1 flip | Jaccard dist | Usage entropy |",
        "|--------|------|-----|------------|------------|--------------|---------------|",
    ]
    order = {"gold": 0, "uniform": 1, "mixed": 2, "placebo": 3, "attention": 4}
    for run in sorted(runs, key=lambda r: (-(r.get("bits") or 99), order.get(r["policy"], 9))):
        pooled = run.get("routing", {}).get("pooled", {})
        kl = pooled.get("kl", {}).get("mean")
        top1 = pooled.get("top1_error", {}).get("mean")
        jac = pooled.get("jaccard_distance", {}).get("mean")
        entropy = pooled.get("cand_usage", {}).get("marginal_entropy_normalized")
        bits = run.get("bits")
        lines.append(
            f"| {run['policy']} | {'BF16' if bits is None else f'INT{bits}'} "
            f"| {run['lm']['perplexity']:.4f} "
            f"| {'-' if kl is None else f'{kl:.6f}'} "
            f"| {'-' if top1 is None else f'{top1 * 100:.2f}%'} "
            f"| {'-' if jac is None else f'{jac:.4f}'} "
            f"| {'-' if entropy is None else f'{entropy:.4f}'} |"
        )
    lines.append("")
    return lines


def sanity_section(runs: list[dict]) -> list[str]:
    lines = ["## Correctness gates", ""]
    for run in sorted(runs, key=lambda r: (r["policy"], r.get("bits") or 0)):
        audit = run.get("audit", {})
        bit_check = run.get("bit_width_check", {})
        precision = "BF16" if run.get("bits") is None else f"INT{run['bits']}"
        lines.append(
            f"- **{run['policy']} {precision}**: "
            f"{audit.get('num_quantized_modules', 0)} modules quantized, "
            f"{audit.get('num_routers_quantized', 0)}/{audit.get('num_routers', 0)} routers quantized, "
            f"bit-width check {'passed' if bit_check.get('passed', True) else 'FAILED'}"
        )
        if run.get("self_check"):
            sc = run["self_check"]
            lines.append(
                f"  - gold self-comparison: KL={sc['kl']:.2e}, top-1 error={sc['top1_error']:.2e} "
                f"({'passed' if sc['passed'] else 'FAILED'})"
            )
    census = next((r.get("parameter_census") for r in runs if r.get("parameter_census")), None)
    if census:
        lines.append(
            f"- Router parameters: {census['router_params']:,} of {census['total_params']:,} "
            f"({census['router_fraction'] * 100:.4f}% of the model)"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-pdf", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    out_pdf = Path(args.out_pdf) if args.out_pdf else results_dir / "figures.pdf"
    out_md = Path(args.out_md) if args.out_md else results_dir / "summary.md"

    with PdfPages(out_pdf) as pdf:
        plot_metric(runs, lambda r: r["lm"]["perplexity"], "WikiText-2 perplexity",
                    "Perplexity vs expert precision", pdf)
        plot_metric(runs, lambda r: r["routing"]["pooled"]["kl"], "KL(gold || candidate)",
                    "Routing drift vs expert precision", pdf, logy=True, show_gold=False)
        plot_metric(runs, lambda r: r["routing"]["pooled"]["top1_error"], "Top-1 flip rate",
                    "Top-1 expert changes vs expert precision", pdf, show_gold=False)
        plot_metric(runs, lambda r: r["routing"]["pooled"]["jaccard_distance"],
                    "Top-k Jaccard distance", "Expert-set disagreement vs precision", pdf,
                    show_gold=False)
        plot_metric(runs,
                    lambda r: r["routing"]["pooled"]["cand_usage"]["marginal_entropy_normalized"],
                    "Normalized expert-usage entropy", "Expert collapse check", pdf,
                    show_gold=False)
        plot_metric(runs, lambda r: r["lm"].get("output_kl_topm"), "Output KL (top-M)",
                    "Output distribution drift vs precision", pdf, show_gold=False)
        for bits in sorted({r["bits"] for r in runs if r.get("bits")}, reverse=True):
            plot_layerwise(runs, bits, pdf)

    parts = [f"# {results_dir.name} - Part 1 results", ""]
    parts += summary_table(runs)
    parts += decision_gate(runs)
    parts += sanity_section(runs)
    body = "\n".join(parts)
    out_md.write_text(body)

    print(body)
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
