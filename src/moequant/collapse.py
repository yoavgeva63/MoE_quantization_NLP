"""Expert collapse: per-layer expert-load imbalance, measured two ways.

The proposal committed to detecting *expert collapse* - quantization starving some
experts of tokens - and to checking that router protection keeps the rare experts
working. `metrics.usage_stats` already measures the right thing, but the number the
summary tables report is `routing.pooled.cand_usage`, and `compare_routing` builds
`pooled` by concatenating tokens across every layer before measuring. Expert 5 in layer 0
and expert 5 in layer 12 are different experts, so imbalance in opposite directions
across layers cancels and the pooled figure reads as a null. This module reports per
layer instead.

Two tiers, at deliberately different token resolutions, because each field is only
trustworthy at one of them:

* **Tier 1** reads `routing.per_layer.<L>.cand_usage` straight out of a run's
  `metrics.json`. Those fields were computed on all 32,768 routing tokens, which is the
  only resolution at which a zero selection count means an expert really received
  nothing. Dead and unused counts come from here.
* **Tier 2** rebuilds per-expert selection counts from the saved artifacts, because the
  runs never stored the counts themselves and the sharper distributional statistics
  (Gini, max share, top-decile share, KL against uniform) need them. Candidate router
  inputs were captured every `input_stride`-th token, so Tier 2 sees 1 token in 8. That
  is ample for ratio statistics but it inflates zero and near-zero counts, which is why
  it must not be used for dead/unused counts.

Normalized entropy is reported for continuity with the existing tables, but it is the
wrong shape for this job: uniform load is its maximum, so its gradient is zero exactly
where the null hypothesis sits. One expert of 64 taking four times its fair share still
scores about 0.99. The Tier 2 statistics are the sensitive ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .metrics import topk_mask

# Fields of `usage_stats` that are only meaningful at full token resolution, so they are
# read from `metrics.json` rather than recomputed from the strided capture.
FULL_RESOLUTION_FIELDS = ("dead_experts", "unused_experts")

# An expert is "dead" below a tenth of its fair share. Same threshold `usage_stats` uses,
# repeated here so a Tier 2 count is directly comparable to a Tier 1 one.
DEAD_FRACTION = 0.1


def gini(load: np.ndarray) -> float:
    """Gini coefficient of an expert load vector: 0 is perfectly balanced, 1 maximally.

    Unlike normalized entropy this is sensitive at the balanced end, which is where the
    interesting comparison sits.
    """
    x = np.sort(np.asarray(load, dtype=np.float64).ravel())
    total = x.sum()
    if x.size == 0 or total <= 0:
        return float("nan")
    rank = np.arange(1, x.size + 1)
    return float((2.0 * (rank * x).sum() - (x.size + 1) * total) / (x.size * total))


def top_decile_count(num_experts: int) -> int:
    """How many experts make up the "top 10%" of a bank of this size.

    Rounded rather than truncated or raised, so 64 experts gives 6 and not 7 - the
    nearest whole expert to a tenth of the bank.
    """
    return max(1, int(round(0.1 * num_experts)))


def distribution_stats(counts: np.ndarray | torch.Tensor) -> dict:
    """Imbalance statistics for one layer's per-expert selection counts.

    `max_share` and `top_decile_share` are expressed against a perfectly balanced router:
    `max_share` is the busiest expert's load as a multiple of its fair share, and
    `top_decile_share` is the fraction of all routing slots the busiest tenth absorbs
    (0.1 under perfect balance).
    """
    if isinstance(counts, torch.Tensor):
        counts = counts.detach().cpu().numpy()
    counts = np.asarray(counts, dtype=np.float64).ravel()
    num_experts = counts.size
    total = counts.sum()
    if num_experts == 0 or total <= 0:
        return {
            "num_experts": int(num_experts),
            "num_selections": 0,
            "gini": float("nan"),
            "max_share": float("nan"),
            "top_decile_share": float("nan"),
            "kl_vs_uniform": float("nan"),
            "marginal_entropy_normalized": float("nan"),
            "dead_experts": 0,
            "unused_experts": int(num_experts),
        }

    load = counts / total
    nonzero = load[load > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    decile = top_decile_count(num_experts)
    uniform_load = 1.0 / num_experts

    return {
        "num_experts": int(num_experts),
        "num_selections": int(total),
        "gini": gini(load),
        "max_share": float(load.max() / uniform_load),
        "top_decile_share": float(np.sort(load)[::-1][:decile].sum()),
        # KL(load || uniform) = log(n) - H(load): zero when balanced, and unlike entropy
        # it grows without an upper ceiling as load concentrates.
        "kl_vs_uniform": float(math.log(num_experts) - entropy),
        "marginal_entropy_normalized": (
            entropy / math.log(num_experts) if num_experts > 1 else float("nan")
        ),
        "dead_experts": int((load < DEAD_FRACTION * uniform_load).sum()),
        "unused_experts": int((counts == 0).sum()),
    }


def expert_counts(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Per-expert top-k selection counts for one layer's router logits."""
    return topk_mask(logits.detach().to(torch.float32), k).sum(0).to(torch.float64)


# -- Tier 1: read what the runs already computed ---------------------------------------


def tier1_layers(per_layer: dict, key: str = "cand_usage") -> list[dict]:
    """Pull one usage block per layer out of a `metrics.json` routing section."""
    rows = []
    for layer in sorted(per_layer, key=int):
        usage = per_layer[layer].get(key)
        if not usage:
            continue
        rows.append({"layer": int(layer), **usage})
    return rows


def tier1_summary(rows: list[dict]) -> dict:
    """Collapse the per-layer table into the numbers a results table quotes.

    Dead and unused counts are summed over layers and expressed against the total number
    of expert *slots* (layers x experts), because an expert in one layer is not the same
    expert as the one with that index in another layer.
    """
    if not rows:
        return {}
    num_layers = len(rows)
    num_experts = int(rows[0]["num_experts"])
    slots = num_layers * num_experts
    dead = sum(int(r["dead_experts"]) for r in rows)
    unused = sum(int(r["unused_experts"]) for r in rows)
    entropies = [float(r["marginal_entropy_normalized"]) for r in rows]
    loads = [float(r["max_over_mean_load"]) for r in rows]

    worst_entropy_layer = min(rows, key=lambda r: r["marginal_entropy_normalized"])
    worst_load_layer = max(rows, key=lambda r: r["max_over_mean_load"])

    return {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "expert_slots": slots,
        "num_tokens": int(rows[0]["num_tokens"]),
        "dead_slots": dead,
        "dead_fraction": dead / slots,
        "unused_slots": unused,
        "unused_fraction": unused / slots,
        "layers_with_unused": sum(1 for r in rows if r["unused_experts"] > 0),
        "layers_with_dead": sum(1 for r in rows if r["dead_experts"] > 0),
        "max_dead_in_a_layer": max(int(r["dead_experts"]) for r in rows),
        "worst_layer_entropy": min(entropies),
        "worst_layer_entropy_layer": int(worst_entropy_layer["layer"]),
        "mean_layer_entropy": float(np.mean(entropies)),
        "worst_max_over_mean_load": max(loads),
        "worst_max_over_mean_load_layer": int(worst_load_layer["layer"]),
        "mean_max_over_mean_load": float(np.mean(loads)),
    }


def tier1_report(routing: dict, key: str = "cand_usage") -> dict:
    """Per-layer table plus summary for one run, from its `metrics.json` routing block."""
    rows = tier1_layers(routing.get("per_layer", {}), key)
    return {"resolution": "full", "per_layer": rows, "summary": tier1_summary(rows)}


# -- Tier 2: rebuild per-expert counts from the artifacts -------------------------------


@dataclass(frozen=True)
class Reconstruction:
    """One run's router logits at the captured positions, layer by layer.

    `router_weights` are the dequantized weights the run actually used, saved alongside
    its inputs, so nothing is re-quantized here and the counts describe the real run.
    """

    inputs: dict[int, torch.Tensor]
    weights: dict[int, torch.Tensor]
    top_k: int
    stride: int

    @property
    def layers(self) -> list[int]:
        return sorted(set(self.inputs) & set(self.weights))

    def counts(self, layer: int) -> torch.Tensor:
        rebuilt = torch.nn.functional.linear(
            self.inputs[layer].to(torch.float32), self.weights[layer].to(torch.float32)
        )
        return expert_counts(rebuilt, self.top_k)


def load_reconstruction(run_dir: Path, top_k: int) -> Reconstruction:
    """Memory-map a run's `router_inputs.pt`.

    These files are 270-410 MB each on shared storage, so they are mapped rather than
    read: only the layers actually visited are paged in.
    """
    payload = torch.load(
        Path(run_dir) / "router_inputs.pt", map_location="cpu", mmap=True, weights_only=False
    )
    return Reconstruction(
        inputs=payload["inputs"],
        weights=payload["router_weights"],
        top_k=top_k,
        stride=int(payload.get("input_stride") or 1),
    )


def tier2_report(counts_by_layer: dict[int, torch.Tensor], stride: int) -> dict:
    """Distributional statistics per layer, plus the aggregates worth quoting."""
    rows = [
        {"layer": int(layer), **distribution_stats(counts_by_layer[layer])}
        for layer in sorted(counts_by_layer)
    ]
    if not rows:
        return {"resolution": "strided", "stride": stride, "per_layer": [], "summary": {}}

    def across(field: str):
        return np.array([r[field] for r in rows], dtype=np.float64)

    worst = max(rows, key=lambda r: r["gini"])
    summary = {
        "num_layers": len(rows),
        "num_experts": rows[0]["num_experts"],
        "expert_slots": len(rows) * rows[0]["num_experts"],
        "tokens_per_layer": int(rows[0]["num_selections"] // max(1, rows[0]["num_experts"])),
        "gini_mean": float(across("gini").mean()),
        "gini_max": float(across("gini").max()),
        "gini_worst_layer": int(worst["layer"]),
        "max_share_mean": float(across("max_share").mean()),
        "max_share_max": float(across("max_share").max()),
        "top_decile_share_mean": float(across("top_decile_share").mean()),
        "top_decile_share_max": float(across("top_decile_share").max()),
        "kl_vs_uniform_mean": float(across("kl_vs_uniform").mean()),
        "kl_vs_uniform_max": float(across("kl_vs_uniform").max()),
        # Reported only to show how badly the stride inflates them; the trustworthy
        # counts are the Tier 1 ones.
        "strided_dead_slots": int(across("dead_experts").sum()),
        "strided_unused_slots": int(across("unused_experts").sum()),
    }
    return {"resolution": "strided", "stride": stride, "per_layer": rows, "summary": summary}


def gold_counts_at_stride(
    gold_logits: dict[int, torch.Tensor], mask: np.ndarray, top_k: int
) -> dict[int, torch.Tensor]:
    """Gold selection counts restricted to the positions the candidates captured.

    Candidate Tier 2 statistics can only be compared against a gold measured on the same
    positions; comparing 1-in-8 against all 32,768 tokens would confound the stride with
    the effect.
    """
    selector = torch.from_numpy(np.asarray(mask, dtype=bool))
    return {
        int(layer): expert_counts(gold_logits[layer][selector], top_k)
        for layer in sorted(gold_logits)
    }
