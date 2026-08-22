#!/usr/bin/env python3
"""Part 2 attribution: how much routing drift is the router's own weights?

Reads the router inputs captured during the Part 1 sweep and rebuilds the four
combinations of gold/quantized weights and activations offline. No GPU, no model download,
no second sweep.

    python scripts/attribute.py --results-dir results/olmoe --bits 4 3

Writes attribution.json, attribution.md and attribution.pdf next to the metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from moequant.attribution import (  # noqa: E402
    attribute,
    check_reconstruction,
    load_cells,
    load_gold,
)

LABELS = {
    "weights_only": "router weights only",
    "activations_only": "upstream activations only",
    "both": "both (= uniform)",
}
COLORS = {
    "weights_only": "#C44E52",
    "activations_only": "#4C72B0",
    "both": "#8172B2",
}


def plot_mechanisms(by_bits: dict[int, dict], pdf: PdfPages, model: str) -> None:
    """Top-1 flip rate for each mechanism, side by side at every bit-width."""
    bit_widths = sorted(by_bits, reverse=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.26
    for offset, cell in enumerate(("weights_only", "activations_only", "both")):
        x = [i + (offset - 1) * width for i in range(len(bit_widths))]
        means = [by_bits[b]["pooled"][cell]["top1_error"]["mean"] for b in bit_widths]
        errs = [
            [
                by_bits[b]["pooled"][cell]["top1_error"]["mean"]
                - by_bits[b]["pooled"][cell]["top1_error"]["ci_low"]
                for b in bit_widths
            ],
            [
                by_bits[b]["pooled"][cell]["top1_error"]["ci_high"]
                - by_bits[b]["pooled"][cell]["top1_error"]["mean"]
                for b in bit_widths
            ],
        ]
        ax.bar(x, means, width, yerr=errs, capsize=4, label=LABELS[cell], color=COLORS[cell])
    ax.set_xticks(range(len(bit_widths)))
    ax.set_xticklabels([f"INT{b}" for b in bit_widths])
    ax.set_xlabel("Expert precision")
    ax.set_ylabel("Top-1 flip rate vs gold")
    ax.set_title(f"{model}: where routing drift comes from")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_layerwise(by_bits: dict[int, dict], pdf: PdfPages, model: str) -> None:
    """Per-layer mechanism split, which shows where in depth each source dominates."""
    for bits in sorted(by_bits, reverse=True):
        report = by_bits[bits]
        layers = report["layers"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for cell in ("weights_only", "activations_only", "both"):
            values = [
                report["per_layer"][cell][str(layer)]["top1_error"]["mean"] for layer in layers
            ]
            ax.plot(layers, values, marker="o", label=LABELS[cell], color=COLORS[cell])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Top-1 flip rate vs gold")
        ax.set_title(f"{model}: mechanism split by layer at INT{bits}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def cross_check(results_dir: Path, by_bits: dict[int, dict]) -> list[str]:
    """Compare the rebuilt cells against the Part 1 runs they should reproduce.

    Cell C is what `mixed` computed and cell D is close to what `uniform` computed, so
    agreement with the recorded metrics is independent evidence that the decomposition is
    measuring the real experiment. D is only close, not equal: it pairs quantized router
    weights with the activations from the `mixed` run, whereas in `uniform` the changed
    routing decisions feed back into the hidden states of later layers. That gap is the
    size of the feedback effect and is worth reporting rather than hiding.
    """
    lines = [
        "## Cross-check against Part 1",
        "",
        "Cell C is the `mixed` condition and cell D is `uniform` without routing feedback. "
        "Part 1 measured both on every token; the cells use the strided capture subset.",
        "",
        "| Bits | C (rebuilt) | mixed (Part 1) | D (rebuilt) | uniform (Part 1) |",
        "|------|-------------|----------------|-------------|------------------|",
    ]
    for bits in sorted(by_bits, reverse=True):
        pooled = by_bits[bits]["pooled"]
        recorded = {}
        for policy in ("mixed", "uniform"):
            path = results_dir / f"{policy}_int{bits}" / "metrics.json"
            if path.exists():
                with path.open() as handle:
                    payload = json.load(handle)
                recorded[policy] = payload["routing"]["pooled"]["top1_error"]["mean"]
        lines.append(
            f"| INT{bits} "
            f"| {pooled['activations_only']['top1_error']['mean'] * 100:.2f}% "
            f"| {recorded.get('mixed', float('nan')) * 100:.2f}% "
            f"| {pooled['both']['top1_error']['mean'] * 100:.2f}% "
            f"| {recorded.get('uniform', float('nan')) * 100:.2f}% |"
        )
    lines.append("")
    return lines


def markdown(model: str, by_bits: dict[int, dict], checks: dict[int, dict]) -> list[str]:
    lines = [
        f"# {model} - Part 2 attribution",
        "",
        "Routing drift split into the router's own rounded weights and the drift in the "
        "hidden states arriving at the router, rebuilt offline from the captured inputs.",
        "",
        "| Bits | Weights only | Activations only | Both (uniform) | Weight share of both | "
        "Weight share of mechanisms |",
        "|------|--------------|------------------|----------------|----------------------|"
        "----------------------------|",
    ]
    for bits in sorted(by_bits, reverse=True):
        report = by_bits[bits]
        pooled, shares = report["pooled"], report["shares"]
        lines.append(
            f"| INT{bits} "
            f"| {pooled['weights_only']['top1_error']['mean'] * 100:.2f}% "
            f"| {pooled['activations_only']['top1_error']['mean'] * 100:.2f}% "
            f"| {pooled['both']['top1_error']['mean'] * 100:.2f}% "
            f"| {shares['weights_of_both']['mean'] * 100:.1f}% "
            f"[{shares['weights_of_both']['ci_low'] * 100:.1f}, "
            f"{shares['weights_of_both']['ci_high'] * 100:.1f}] "
            f"| {shares['weights_of_mechanisms']['mean'] * 100:.1f}% "
            f"[{shares['weights_of_mechanisms']['ci_low'] * 100:.1f}, "
            f"{shares['weights_of_mechanisms']['ci_high'] * 100:.1f}] |"
        )

    lines += ["", "## Routing KL by mechanism", "",
              "| Bits | Weights only | Activations only | Both (uniform) |",
              "|------|--------------|------------------|----------------|"]
    for bits in sorted(by_bits, reverse=True):
        pooled = by_bits[bits]["pooled"]
        lines.append(
            f"| INT{bits} "
            f"| {pooled['weights_only']['kl']['mean']:.6f} "
            f"| {pooled['activations_only']['kl']['mean']:.6f} "
            f"| {pooled['both']['kl']['mean']:.6f} |"
        )

    lines += ["", "## Reconstruction check", "",
              "Cell A is rebuilt from the stored activations and weights; it must reproduce "
              "the routing decisions the gold run recorded.", ""]
    for bits in sorted(checks, reverse=True):
        check = checks[bits]
        lines.append(
            f"- INT{bits}: mean top-1 agreement with recorded gold logits "
            f"{check['top1_agreement'] * 100:.2f}%, worst layer "
            f"{check['worst_layer_agreement'] * 100:.2f}% "
            f"({'passed' if check['passed'] else 'FAILED'})"
        )

    interaction = {bits: by_bits[bits]["shares"]["interaction"]["mean"] for bits in by_bits}
    lines += [
        "",
        "## Reading these numbers",
        "",
        "The two mechanisms are not additive, so the shares of `both` do not sum to 100%. "
        "A negative residual means the mechanisms flip overlapping sets of tokens: a token "
        "that either source alone would have flipped is counted once in `both` but twice "
        "across the two single-mechanism cells. The residual is reported rather than "
        "folded into a share: "
        + ", ".join(f"INT{b} {v * 100:+.1f}%" for b, v in sorted(interaction.items(), reverse=True))
        + ".",
        "",
        "The last column is the interpretable one: of the routing flips caused by exactly "
        "one mechanism, what fraction comes from the router's own weights. Above 50% means "
        "router protection addresses the dominant source at that precision.",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="attribution")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = results_dir.name
    gold = load_gold(results_dir / "gold")

    by_bits: dict[int, dict] = {}
    checks: dict[int, dict] = {}
    for bits in args.bits:
        print(f"[{model}] INT{bits}: loading cells")
        cells = load_cells(results_dir, bits, gold=gold)
        check = check_reconstruction(cells, gold[0]["logits"])
        print(
            f"  reconstruction: top-1 agreement {check['top1_agreement'] * 100:.2f}% "
            f"(worst layer {check['worst_layer_agreement'] * 100:.2f}%)"
        )
        if not check["passed"]:
            raise SystemExit(
                f"Reconstruction check failed for INT{bits}: cell A does not reproduce the "
                f"gold run's routing decisions (worst layer "
                f"{check['worst_layer_agreement'] * 100:.2f}% < "
                f"{check['threshold'] * 100:.0f}%). The decomposition would be meaningless."
            )
        report = attribute(cells, n_boot=args.n_boot, seed=args.seed)
        pooled = report["pooled"]
        print(
            f"  top-1 flips: weights {pooled['weights_only']['top1_error']['mean'] * 100:.2f}% | "
            f"activations {pooled['activations_only']['top1_error']['mean'] * 100:.2f}% | "
            f"both {pooled['both']['top1_error']['mean'] * 100:.2f}%"
        )
        by_bits[bits] = report
        checks[bits] = check

    out_json = results_dir / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(
        {"model": model, "checks": checks,
         "attribution": {str(b): r for b, r in by_bits.items()}},
        indent=2,
    ))

    out_pdf = results_dir / f"{args.out_prefix}.pdf"
    with PdfPages(out_pdf) as pdf:
        plot_mechanisms(by_bits, pdf, model)
        plot_layerwise(by_bits, pdf, model)

    body = "\n".join(markdown(model, by_bits, checks) + cross_check(results_dir, by_bits))
    out_md = results_dir / f"{args.out_prefix}.md"
    out_md.write_text(body)

    print(f"\n{body}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
