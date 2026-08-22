"""Part 2: split routing drift into router-weight error and upstream activation drift.

Part 1 measures how much routing changes when a model is quantized. It cannot say *why*.
Two mechanisms are folded together in every `uniform` number: the router's own weights
were rounded, and the hidden states arriving at the router were already damaged by
quantized layers upstream. Protecting the router removes only the first.

The captured router inputs make the two separable without touching a GPU. With gold and
quantized copies of both the weights and the activations, all four combinations are one
matrix multiply each:

    A = X_gold  @ W_gold^T     reference; reproduces the gold run's logits
    B = X_gold  @ W_quant^T    router weight error alone
    C = X_quant @ W_gold^T     upstream activation drift alone (the `mixed` condition)
    D = X_quant @ W_quant^T    both mechanisms together

Every metric is computed against A, which makes B and C directly comparable and lets us
say what fraction of the damage in D each mechanism accounts for. The two are not
additive - routing is a top-k argmax over a softmax, so errors can cancel as easily as
compound - and the residual is reported explicitly as an interaction term, with its own
bootstrap interval, rather than quietly folded into one of the shares. Because the
residual is large and negative on both models, the shares are *not* a partition: they
cannot be quoted as "X% weights, (100-X)% activations".

Activations come from the `mixed` run rather than `uniform`: both quantize the same expert
and attention weights, but only `mixed` leaves the routers alone, so its hidden states are
the ones that isolate upstream drift from any feedback through changed routing decisions.

That choice makes C genuinely the `mixed` condition, but it means **D is not the `uniform`
run**. D pairs quantized router weights with the `mixed` run's activations, so it omits
the feedback of perturbed routing decisions on later layers' hidden states. The gap is
measurable and grows with damage (OLMoE INT3: D 50.01% vs uniform 52.52%; Qwen INT3:
68.58% vs 72.48%), and it is the size of that feedback effect rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .metrics import compare_layer, per_token_selection

CELLS = ("weights_only", "activations_only", "both")

# Cell A is rebuilt from fp16-stored activations and fp32 weights on the CPU, while the
# run itself computed logits in bf16 on a GPU. The two will not agree bitwise, so the
# gate is on decisions rather than values: anything below this means the reconstruction
# is not the same computation the model performed, and the decomposition is meaningless.
MIN_RECONSTRUCTION_AGREEMENT = 0.95


@dataclass(frozen=True)
class Cells:
    """Gold and quantized copies of both inputs to the routing computation."""

    x_gold: dict[int, torch.Tensor]
    x_quant: dict[int, torch.Tensor]
    w_gold: dict[int, torch.Tensor]
    w_quant: dict[int, torch.Tensor]
    groups: np.ndarray
    top_k: int
    stride: int

    @property
    def layers(self) -> list[int]:
        return sorted(set(self.x_gold) & set(self.x_quant) & set(self.w_gold) & set(self.w_quant))


def logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """One router's logits for stored activations and weights."""
    return torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))


def strided_positions(groups: np.ndarray, stride: int) -> np.ndarray:
    """Boolean mask selecting the token positions that the input capture kept.

    The capture strides *within* each sequence, so the mask cannot be recovered by
    slicing the concatenated token axis unless every sequence has the same length. This
    reconstructs it from the sequence ids instead, which holds either way.
    """
    groups = np.asarray(groups).ravel()
    if groups.size == 0:
        return np.zeros(0, dtype=bool)
    # Sequences are captured contiguously, so a change in the id starts a new sequence.
    boundaries = np.flatnonzero(np.diff(groups, prepend=groups[0] - 1))
    counts = np.diff(np.append(boundaries, groups.size))
    position = np.concatenate([np.arange(count) for count in counts])
    return position % stride == 0


def load_gold(gold_dir: Path) -> tuple[dict, dict]:
    """Read the gold artifacts and captured inputs, which every bit-width shares.

    These are the largest files in the results tree, so a sweep over bit-widths reads them
    once and passes them back in rather than paying for them per run.
    """
    gold_dir = Path(gold_dir)
    return (
        torch.load(gold_dir / "artifacts.pt", map_location="cpu", weights_only=False),
        torch.load(gold_dir / "router_inputs.pt", map_location="cpu", weights_only=False),
    )


def load_cells(
    results_dir: Path,
    bits: int,
    gold: tuple[dict, dict] | None = None,
    gold_dir: Path | None = None,
) -> Cells:
    """Assemble the four-cell inputs from one model's saved runs.

    Quantized router weights come from the `uniform` run, which is the only policy that
    both quantizes the routers and stores a dequantized copy of what it used. Nothing is
    re-quantized here: re-deriving the weights offline would risk measuring a different
    rounding scheme than the one the experiment actually ran.
    """
    results_dir = Path(results_dir)
    gold_dir = Path(gold_dir) if gold_dir is not None else results_dir / "gold"

    gold_artifacts, gold_inputs = gold if gold is not None else load_gold(gold_dir)
    mixed = torch.load(
        results_dir / f"mixed_int{bits}" / "router_inputs.pt",
        map_location="cpu",
        weights_only=False,
    )
    uniform = torch.load(
        results_dir / f"uniform_int{bits}" / "router_inputs.pt",
        map_location="cpu",
        weights_only=False,
    )

    w_gold = gold_artifacts["router_weights"]
    w_quant = uniform["router_weights"]
    unchanged = [layer for layer in w_gold if layer in w_quant
                 and torch.equal(w_gold[layer].float(), w_quant[layer].float())]
    if unchanged:
        raise ValueError(
            f"Router weights in uniform_int{bits} are identical to gold for layers "
            f"{unchanged[:5]}. That run did not quantize its routers, so cells B and D "
            "would repeat A and C."
        )

    return Cells(
        x_gold=gold_inputs["inputs"],
        x_quant=mixed["inputs"],
        w_gold=w_gold,
        w_quant=w_quant,
        groups=np.asarray(gold_artifacts["groups"]),
        top_k=int(gold_artifacts["topology"]["top_k"]),
        stride=int(gold_inputs.get("input_stride") or mixed.get("input_stride") or 1),
    )


def check_reconstruction(cells: Cells, gold_logits: dict[int, torch.Tensor]) -> dict:
    """Confirm cell A reproduces the routing decisions the gold run actually made.

    This is the gate on the whole decomposition. If A disagrees with the recorded gold
    logits, then the stored activations, weights or stride do not describe the same
    computation, and B/C/D are measuring something other than quantization.
    """
    mask = strided_positions(cells.groups, cells.stride)
    agreements: list[float] = []
    kls: list[float] = []
    for layer in cells.layers:
        if layer not in gold_logits:
            continue
        if gold_logits[layer].shape[0] != mask.size:
            raise ValueError(
                f"Layer {layer}: recorded gold logits have {gold_logits[layer].shape[0]} "
                f"tokens but the sequence ids describe {mask.size}. These artifacts do not "
                "align; the logits and groups must come from the same gold run."
            )
        recorded = gold_logits[layer][mask]
        rebuilt = logits(cells.x_gold[layer], cells.w_gold[layer])
        if recorded.shape != rebuilt.shape:
            raise ValueError(
                f"Layer {layer}: recorded gold logits {tuple(recorded.shape)} do not align "
                f"with the rebuilt cell {tuple(rebuilt.shape)}. Check the input stride."
            )
        selection = per_token_selection(recorded, rebuilt, cells.top_k)
        agreements.append(1.0 - float(selection["top1_error"].mean()))
        delta = torch.log_softmax(recorded.float(), -1) - torch.log_softmax(rebuilt.float(), -1)
        kls.append(float(delta.abs().mean()))

    report = {
        "top1_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "worst_layer_agreement": float(np.min(agreements)) if agreements else float("nan"),
        "mean_abs_logprob_delta": float(np.mean(kls)) if kls else float("nan"),
        "num_layers": len(agreements),
        "threshold": MIN_RECONSTRUCTION_AGREEMENT,
    }
    report["passed"] = bool(report["worst_layer_agreement"] >= MIN_RECONSTRUCTION_AGREEMENT)
    return report


def _grouped_ratio_ci(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    groups: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a ratio of two per-token means, resampling whole sequences.

    The shares are ratios of correlated quantities measured on the same tokens, so their
    interval cannot be derived from the two marginal intervals. Resampling sequences once
    and recomputing both means keeps that correlation.
    """
    num = numerator.detach().cpu().numpy().astype(np.float64).ravel()
    den = denominator.detach().cpu().numpy().astype(np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if num.size == 0 or num.size != den.size or groups.size != num.size:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    point = float(num.mean() / den.mean()) if den.mean() else float("nan")
    _, index = np.unique(groups, return_inverse=True)
    num_sums = np.bincount(index, weights=num)
    den_sums = np.bincount(index, weights=den)

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, num_sums.size, size=(n_boot, num_sums.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = num_sums[draw].sum(axis=1) / den_sums[draw].sum(axis=1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"mean": point, "ci_low": float("nan"), "ci_high": float("nan")}
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"mean": point, "ci_low": float(low), "ci_high": float(high)}


def attribute(cells: Cells, n_boot: int = 200, seed: int = 0) -> dict:
    """Run the decomposition and return per-layer, pooled, and share statistics."""
    layers = cells.layers
    if not layers:
        raise ValueError("No layer is present in all four cells")

    mask = strided_positions(cells.groups, cells.stride)
    groups = cells.groups[mask]

    built: dict[str, dict[int, torch.Tensor]] = {"reference": {}, **{c: {} for c in CELLS}}
    for layer in layers:
        x_gold, x_quant = cells.x_gold[layer], cells.x_quant[layer]
        w_gold, w_quant = cells.w_gold[layer], cells.w_quant[layer]
        built["reference"][layer] = logits(x_gold, w_gold)
        built["weights_only"][layer] = logits(x_gold, w_quant)
        built["activations_only"][layer] = logits(x_quant, w_gold)
        built["both"][layer] = logits(x_quant, w_quant)

    per_layer: dict[str, dict[str, dict]] = {cell: {} for cell in CELLS}
    pooled: dict[str, dict] = {}
    flips: dict[str, torch.Tensor] = {}

    pooled_reference = torch.cat([built["reference"][layer] for layer in layers], dim=0)
    pooled_groups = np.tile(groups, len(layers))

    for cell in CELLS:
        for layer in layers:
            per_layer[cell][str(layer)] = compare_layer(
                built["reference"][layer], built[cell][layer], cells.top_k,
                groups=groups, n_boot=n_boot, seed=seed,
            )
        candidate = torch.cat([built[cell][layer] for layer in layers], dim=0)
        pooled[cell] = compare_layer(
            pooled_reference, candidate, cells.top_k,
            groups=pooled_groups, n_boot=n_boot, seed=seed,
        )
        flips[cell] = per_token_selection(pooled_reference, candidate, cells.top_k)["top1_error"]

    shares = {
        "weights_of_both": _grouped_ratio_ci(
            flips["weights_only"], flips["both"], pooled_groups, n_boot, seed=seed
        ),
        "activations_of_both": _grouped_ratio_ci(
            flips["activations_only"], flips["both"], pooled_groups, n_boot, seed=seed
        ),
        "weights_of_mechanisms": _grouped_ratio_ci(
            flips["weights_only"],
            flips["weights_only"] + flips["activations_only"],
            pooled_groups, n_boot, seed=seed,
        ),
    }
    # The residual is an exact rearrangement of the two shares above, so it gets a real
    # interval from the same resamples rather than a point estimate standing beside two
    # numbers that have one.
    shares["interaction"] = _grouped_ratio_ci(
        flips["both"] - flips["weights_only"] - flips["activations_only"],
        flips["both"],
        pooled_groups, n_boot, seed=seed,
    )

    return {
        "top_k": cells.top_k,
        "layers": layers,
        "stride": cells.stride,
        "num_tokens": int(mask.sum()),
        "num_sequences": int(np.unique(groups).size),
        "per_layer": per_layer,
        "pooled": pooled,
        "shares": shares,
    }
