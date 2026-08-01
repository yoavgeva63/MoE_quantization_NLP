"""Evaluation corpora and the seeded token subsets every config shares.

Gold and candidate runs must see *exactly* the same tokens in the same order, otherwise
captured router logits do not line up and every comparison is meaningless. Everything
here is therefore a pure function of (corpus, tokenizer, seed), with no run-to-run state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

CORPORA = {
    "wikitext2": dict(path="wikitext", name="wikitext-2-raw-v1", split="test", field="text"),
    # No `name` here on purpose: allenai/c4 rejects a config name alongside an explicit
    # data_files shard, and one validation shard is plenty for this evaluation.
    "c4": dict(
        path="allenai/c4",
        split="validation",
        field="text",
        data_files="en/c4-validation.00000-of-00008.json.gz",
    ),
}


@dataclass
class RoutingBatches:
    """Fixed-length sequences plus the sequence id of every token position.

    ``groups`` is what lets the bootstrap resample whole sequences rather than individual
    tokens, which would otherwise give dishonestly narrow confidence intervals.
    """

    batches: list[dict[str, torch.Tensor]]
    groups: np.ndarray
    seq_len: int
    num_sequences: int

    def __len__(self) -> int:
        return len(self.batches)

    @property
    def fingerprint(self) -> str:
        """Hash of the exact token ids fed to the model, in order.

        Gold and candidate runs are compared row by row, but the only structural check
        downstream is that the logit tensors have the same shape. Shapes still match if
        the seed, corpus, or tokenizer changed, so without this a mismatched pair would
        be silently compared and produce entirely plausible numbers.
        """
        return fingerprint_batches(self.batches)


def fingerprint_batches(batches: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        ids = batch["input_ids"].to(torch.int64).cpu().numpy()
        digest.update(np.ascontiguousarray(ids).tobytes())
    return digest.hexdigest()[:16]


def load_token_stream(
    tokenizer,
    corpus: str = "wikitext2",
    max_documents: int | None = None,
    cache_dir: str | None = None,
) -> torch.Tensor:
    """Tokenize a corpus into one flat 1-D tensor of token ids."""
    from datasets import load_dataset

    if corpus not in CORPORA:
        raise KeyError(f"Unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    cfg = dict(CORPORA[corpus])
    field = cfg.pop("field")

    dataset = load_dataset(**cfg, cache_dir=cache_dir)
    texts = dataset[field]
    if max_documents is not None:
        texts = texts[:max_documents]

    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt")
    return encoded.input_ids[0]


def ppl_batches(
    tokens: torch.Tensor,
    seq_len: int = 1024,
    stride: int | None = None,
    max_windows: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Sliding windows for perplexity.

    With ``stride == seq_len`` (the default) windows are disjoint and every token is
    predicted exactly once. A smaller stride gives each token more context; the overlap
    is then masked out of the loss with -100 so it is not counted twice.
    """
    stride = stride or seq_len
    if not 0 < stride <= seq_len:
        raise ValueError(f"stride must be in (0, seq_len]; got {stride} with seq_len={seq_len}")

    windows: list[dict[str, torch.Tensor]] = []
    for start in range(0, max(1, tokens.numel() - 1), stride):
        end = min(start + seq_len, tokens.numel())
        if end - start < 2:
            break
        chunk = tokens[start:end]
        labels = chunk.clone()
        # Only score the tokens this window is responsible for.
        context = (seq_len - stride) if start > 0 else 0
        if context:
            labels[:context] = -100
        windows.append(
            {"input_ids": chunk.unsqueeze(0), "labels": labels.unsqueeze(0)}
        )
        if end == tokens.numel():
            break
    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def routing_batches(
    tokens: torch.Tensor,
    num_sequences: int = 128,
    seq_len: int = 256,
    seed: int = 42,
) -> RoutingBatches:
    """Seeded, non-overlapping sequences used for every routing comparison.

    Single-sequence batches keep this padding-free, so every captured router row is a
    real token and no masking bookkeeping is needed downstream.
    """
    usable = tokens.numel() - seq_len
    if usable <= 0:
        raise ValueError(
            f"Corpus has {tokens.numel()} tokens, too few for {num_sequences}x{seq_len}"
        )

    rng = np.random.default_rng(seed)
    max_start = usable // seq_len
    if max_start < num_sequences:
        raise ValueError(
            f"Corpus supports at most {max_start} disjoint sequences of length {seq_len}, "
            f"but {num_sequences} were requested"
        )
    starts = rng.choice(max_start, size=num_sequences, replace=False) * seq_len
    starts.sort()

    batches = [
        {"input_ids": tokens[s : s + seq_len].unsqueeze(0)} for s in starts.tolist()
    ]
    groups = np.repeat(np.arange(num_sequences), seq_len)
    return RoutingBatches(
        batches=batches, groups=groups, seq_len=seq_len, num_sequences=num_sequences
    )


def describe(tokens: torch.Tensor, routing: RoutingBatches, corpus: str) -> dict:
    """Dataset statistics for the paper."""
    return {
        "corpus": corpus,
        "total_tokens": int(tokens.numel()),
        "routing_sequences": routing.num_sequences,
        "routing_seq_len": routing.seq_len,
        "routing_tokens": int(routing.num_sequences * routing.seq_len),
        "routing_fingerprint": routing.fingerprint,
    }
