"""Routing comparison metrics.

Everything here takes raw router logits of shape ``[num_tokens, num_experts]`` and
compares a candidate model against gold on identical token positions.

Three deliberate choices worth knowing about:

* We report the **full-softmax KL** and the **effective routing weights** separately. The
  full softmax is dominated by the long tail of experts that are never selected, so on
  its own it can move a lot while the actual computation path is unchanged. The effective
  weights - post-top-k, renormalised, zero elsewhere - are what really multiply the
  expert outputs.
* We report **Jensen-Shannon** alongside KL because KL is unbounded and explodes when the
  candidate puts near-zero mass where gold had some.
* Everything aggregate comes with a **bootstrap confidence interval** over sequences.
  Quantization is deterministic, so sampling noise is the only randomness, and without
  intervals a small difference between `mixed` and `uniform` is unreadable.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def _as_float32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32)


def _check_aligned(gold: torch.Tensor, cand: torch.Tensor) -> None:
    if gold.shape != cand.shape:
        raise ValueError(
            f"Gold and candidate logits must align token-for-token, got "
            f"{tuple(gold.shape)} vs {tuple(cand.shape)}. Both runs must use the same "
            "seeded batches in the same order."
        )


# -- divergences --------------------------------------------------------------------


def per_token_kl(gold: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """KL(gold || candidate) over the full expert softmax, per token."""
    _check_aligned(gold, cand)
    log_p = torch.log_softmax(_as_float32(gold), dim=-1)
    log_q = torch.log_softmax(_as_float32(cand), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def per_token_js(gold: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence per token: symmetric and bounded by log 2."""
    _check_aligned(gold, cand)
    p = torch.softmax(_as_float32(gold), dim=-1)
    q = torch.softmax(_as_float32(cand), dim=-1)
    m = 0.5 * (p + q)
    log_m = (m + EPS).log()
    kl_pm = (p * ((p + EPS).log() - log_m)).sum(-1)
    kl_qm = (q * ((q + EPS).log() - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


# -- selection --------------------------------------------------------------------


def topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean [num_tokens, num_experts] mask of the selected experts."""
    idx = logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(-1, idx, True)


def effective_weights(logits: torch.Tensor, k: int, normalize: bool = True) -> torch.Tensor:
    """Routing weights as they actually reach the experts: top-k, renormalised, else zero."""
    probs = torch.softmax(_as_float32(logits), dim=-1)
    mask = topk_mask(probs, k)
    kept = probs * mask
    if normalize:
        kept = kept / kept.sum(-1, keepdim=True).clamp_min(EPS)
    return kept


def per_token_selection(gold: torch.Tensor, cand: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per-token top-k agreement statistics."""
    _check_aligned(gold, cand)
    gold_mask = topk_mask(_as_float32(gold), k)
    cand_mask = topk_mask(_as_float32(cand), k)

    intersection = (gold_mask & cand_mask).sum(-1).to(torch.float32)
    # |A| = |B| = k, so the union is 2k minus the overlap.
    union = (2 * k) - intersection
    jaccard_distance = 1.0 - intersection / union.clamp_min(EPS)

    gold_top1 = _as_float32(gold).argmax(-1)
    cand_top1 = _as_float32(cand).argmax(-1)

    return {
        "jaccard_distance": jaccard_distance,
        "set_mismatch": (intersection < k).to(torch.float32),
        "top1_error": (gold_top1 != cand_top1).to(torch.float32),
        "num_swapped": (k - intersection),
        "weight_l1": (
            effective_weights(gold, k) - effective_weights(cand, k)
        ).abs().sum(-1),
    }


# -- expert load --------------------------------------------------------------------


def usage_stats(logits: torch.Tensor, k: int) -> dict[str, float]:
    """Expert load distribution, the collapse detector.

    Marginal entropy alone conflates load imbalance with per-token router confidence, so
    we report per-token entropy separately.
    """
    logits = _as_float32(logits)
    num_tokens, num_experts = logits.shape
    mask = topk_mask(logits, k)

    counts = mask.sum(0).to(torch.float32)
    load = counts / counts.sum().clamp_min(EPS)
    marginal_entropy = -(load * (load + EPS).log()).sum().item()

    probs = torch.softmax(logits, dim=-1)
    token_entropy = -(probs * (probs + EPS).log()).sum(-1).mean().item()

    uniform_load = 1.0 / num_experts
    return {
        "marginal_entropy": marginal_entropy,
        "marginal_entropy_normalized": marginal_entropy / float(np.log(num_experts)),
        "mean_token_entropy": token_entropy,
        "dead_experts": int((load < 0.1 * uniform_load).sum().item()),
        "unused_experts": int((counts == 0).sum().item()),
        "max_over_mean_load": (load.max() / uniform_load).item(),
        "num_experts": num_experts,
        "num_tokens": num_tokens,
    }


# -- aggregation --------------------------------------------------------------------


def bootstrap_ci(
    values: torch.Tensor | np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Mean with a bootstrap confidence interval, resampling whole sequences.

    Tokens within a sequence are correlated, so resampling individual tokens would give
    dishonestly tight intervals. When `groups` (a sequence id per token) is supplied we
    resample sequences instead.
    """
    arr = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    arr = arr.astype(np.float64).ravel()
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    mean = float(arr.mean())
    rng = np.random.default_rng(seed)

    if groups is None:
        idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
        samples = arr[idx].mean(axis=1)
    else:
        groups = np.asarray(groups).ravel()
        if groups.size != arr.size:
            raise ValueError(
                f"groups has {groups.size} entries but values has {arr.size}; "
                "every token needs a sequence id."
            )
        # Precompute per-group sums and counts so each bootstrap draw is O(num_groups).
        # bincount does this in one pass instead of one mask per group.
        _, index = np.unique(groups, return_inverse=True)
        sums = np.bincount(index, weights=arr)
        counts = np.bincount(index).astype(np.float64)
        draw = rng.integers(0, sums.size, size=(n_boot, sums.size))
        samples = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)

    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high), "n": int(arr.size)}


def compare_layer(
    gold: torch.Tensor,
    cand: torch.Tensor,
    k: int,
    groups: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """All routing metrics for a single layer."""
    selection = per_token_selection(gold, cand, k)
    return {
        "kl": bootstrap_ci(per_token_kl(gold, cand), groups, n_boot, seed=seed),
        "js": bootstrap_ci(per_token_js(gold, cand), groups, n_boot, seed=seed),
        "jaccard_distance": bootstrap_ci(
            selection["jaccard_distance"], groups, n_boot, seed=seed
        ),
        "set_mismatch": bootstrap_ci(selection["set_mismatch"], groups, n_boot, seed=seed),
        "top1_error": bootstrap_ci(selection["top1_error"], groups, n_boot, seed=seed),
        "weight_l1": bootstrap_ci(selection["weight_l1"], groups, n_boot, seed=seed),
        "mean_experts_swapped": float(selection["num_swapped"].mean().item()),
        "gold_usage": usage_stats(gold, k),
        "cand_usage": usage_stats(cand, k),
    }


def compare_routing(
    gold_logits: dict[int, torch.Tensor],
    cand_logits: dict[int, torch.Tensor],
    k: int,
    groups: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Layer-wise and pooled routing comparison across the whole model."""
    layers = sorted(set(gold_logits) & set(cand_logits))
    if not layers:
        raise ValueError("Gold and candidate captures share no layers")

    per_layer = {
        layer: compare_layer(gold_logits[layer], cand_logits[layer], k, groups, n_boot, seed)
        for layer in layers
    }

    # Pool across layers by concatenating tokens, so the headline number weights every
    # (token, layer) routing decision equally.
    all_gold = torch.cat([gold_logits[i] for i in layers], dim=0)
    all_cand = torch.cat([cand_logits[i] for i in layers], dim=0)
    pooled_groups = np.tile(groups, len(layers)) if groups is not None else None
    pooled = compare_layer(all_gold, all_cand, k, pooled_groups, n_boot, seed)

    return {
        "top_k": k,
        "layers": layers,
        "per_layer": {str(layer): per_layer[layer] for layer in layers},
        "pooled": pooled,
    }
