"""End-to-end language modelling quality.

Perplexity is the headline number the proposal promised, but it is blunt: INT8 barely
moves it, and a model can shift its output distribution noticeably while perplexity stays
put. So we also record how often the candidate's top prediction matches gold, and a
truncated KL over gold's most likely tokens.

Storing full-vocabulary distributions is not viable (tens of thousands of positions times
a 150k vocabulary), so the gold run saves only its top-M token ids and log-probabilities
at a strided subset of positions. The candidate then gathers those same token ids and
both sides are renormalised over that shared support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

EPS = 1e-12


@dataclass
class OutputReference:
    """Gold output distribution, compressed to a top-M support."""

    positions: torch.Tensor  # [P] flat index of each sampled position
    token_ids: torch.Tensor  # [P, M] gold's most likely tokens there
    log_probs: torch.Tensor  # [P, M] gold log-probabilities for those tokens

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "token_ids": self.token_ids,
            "log_probs": self.log_probs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, torch.Tensor]) -> OutputReference:
        return cls(
            positions=payload["positions"],
            token_ids=payload["token_ids"],
            log_probs=payload["log_probs"],
        )


@torch.no_grad()
def evaluate_lm(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device | str,
    reference: OutputReference | None = None,
    top_m: int = 64,
    position_stride: int = 16,
    build_reference: bool = False,
    progress: bool = True,
) -> tuple[dict, OutputReference | None]:
    """Compute perplexity and, optionally, output-agreement against a gold reference.

    Set ``build_reference=True`` on the gold run to produce the reference that later
    candidate runs are scored against.
    """
    from tqdm.auto import tqdm

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    ref_positions: list[torch.Tensor] = []
    ref_tokens: list[torch.Tensor] = []
    ref_logprobs: list[torch.Tensor] = []

    agree_hits = 0
    agree_total = 0
    kl_sum = 0.0
    kl_count = 0

    offset = 0
    ref_cursor = 0

    for batch in tqdm(batches, desc="eval", disable=not progress):
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels")
        labels = labels.to(device) if labels is not None else input_ids

        logits = model(input_ids=input_ids).logits.float()

        # Standard causal shift: position t predicts token t+1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.reshape(-1)

        valid = flat_labels != -100
        if valid.any():
            nll = torch.nn.functional.cross_entropy(
                flat_logits[valid], flat_labels[valid], reduction="sum"
            )
            total_nll += nll.item()
            total_tokens += int(valid.sum().item())

        log_probs = torch.log_softmax(flat_logits, dim=-1)
        n_positions = flat_logits.shape[0]

        if build_reference:
            local = torch.arange(0, n_positions, position_stride, device=device)
            local = local[valid[local]]
            if local.numel():
                top = log_probs[local].topk(top_m, dim=-1)
                ref_positions.append((local + offset).cpu())
                ref_tokens.append(top.indices.cpu())
                ref_logprobs.append(top.values.cpu())

        elif reference is not None:
            # Reference positions are sorted flat indices over the concatenated batches,
            # so the ones belonging to this batch are a contiguous run ending at `end`.
            pos = reference.positions
            end = offset + n_positions
            hi = int(torch.searchsorted(pos, torch.tensor(end, dtype=pos.dtype)).item())
            sel = slice(ref_cursor, hi)
            ref_cursor = hi

            if sel.stop > sel.start:
                local = (pos[sel] - offset).to(device)
                gold_ids = reference.token_ids[sel].to(device)
                gold_lp = reference.log_probs[sel].to(device).float()

                cand_lp_full = log_probs[local]
                cand_lp = cand_lp_full.gather(-1, gold_ids)

                # Renormalise both sides over gold's top-M support before comparing.
                p = torch.softmax(gold_lp, dim=-1)
                q_log = torch.log_softmax(cand_lp, dim=-1)
                p_log = torch.log_softmax(gold_lp, dim=-1)
                kl_sum += float((p * (p_log - q_log)).sum(-1).sum().item())
                kl_count += int(local.numel())

                agree_hits += int(
                    (cand_lp_full.argmax(-1) == gold_ids[:, 0]).sum().item()
                )
                agree_total += int(local.numel())

        offset += n_positions

    if reference is not None and not build_reference:
        # Every gold position must have been consumed. A leftover means this run produced
        # fewer scoring positions than gold did - a different ppl_seq_len, corpus, or
        # max_ppl_windows - and the positions we did score belong to different tokens.
        if ref_cursor != reference.positions.numel():
            raise ValueError(
                f"Output reference misaligned: consumed {ref_cursor} of "
                f"{reference.positions.numel()} gold positions across {offset} scoring "
                "positions. The candidate run must use the same corpus, ppl_seq_len, "
                "ppl_stride and max_ppl_windows as the gold run it is scored against."
            )

    perplexity = float(torch.tensor(total_nll / max(total_tokens, 1)).exp().item())
    result: dict = {
        "perplexity": perplexity,
        "mean_nll": total_nll / max(total_tokens, 1),
        "scored_tokens": total_tokens,
    }
    if agree_total:
        result["top1_agreement"] = agree_hits / agree_total
        result["output_kl_topm"] = kl_sum / max(kl_count, 1)
        result["output_positions"] = agree_total

    built = None
    if build_reference and ref_positions:
        built = OutputReference(
            positions=torch.cat(ref_positions),
            token_ids=torch.cat(ref_tokens),
            log_probs=torch.cat(ref_logprobs),
        )
    return result, built
