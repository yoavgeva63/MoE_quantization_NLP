# Project status

Where things stand as of 2026-08-22. This is a catch-up note, not a paper draft.
`OVERVIEW.md` has the research question, `ARCHITECTURE.md` has the code, `results/` has
every number. This file says what we ran, what we think it means, and what is left.

## 1. What we ran

Two MoE models — **OLMoE-1B-7B** (16 layers, 64 experts, top-8; primary) and
**Qwen1.5-MoE-A2.7B** (24 layers, 60 experts, top-4). Post-training weight-only
quantization, per-output-channel symmetric INT-*N*, **no calibration anywhere**: scales
come from the weight tensors alone, so no data ever touches the quantizer.

Five configurations, each differing from `uniform` in exactly one respect:

- `gold` — BF16 everywhere (upper bound)
- `uniform` — INT-*N* everywhere (negative baseline)
- `mixed` — INT-*N* experts, BF16 routers (the proposal)
- `placebo` — `uniform` plus one randomly drawn non-router module matched to the router
parameter budget (control for "any high-precision weights help")
- `attention` — `uniform` but attention stays BF16 (control for "any high-precision
island on the every-token path helps")

Swept over INT8 / INT4 / INT3, except `attention`, which we only ran at INT4 and INT3.
Evaluated on the WikiText-2 test split (~288k / ~299k scored tokens) with perplexity,
routing KL, top-1 expert flip rate, top-*k* Jaccard distance, per-layer expert usage and
output-distribution drift. Routing metrics use a 32,768-token-per-layer subset (128
sequences × 256 tokens) with sequence-level bootstrap intervals.

**Part 2** is a separate offline analysis that costs no GPU time: using router inputs
captured during the runs (every 8th token), it rebuilds routing decisions under four
combinations of quantized/clean router weights and quantized/clean incoming activations,
splitting routing drift into the router's own rounded weights versus drift in what
arrives at the router. Two further offline analyses came out of the same artifacts: a
per-layer expert-collapse table and a paired-difference bootstrap (10,000 replicates).

Everything is machine-checked. `results/<model>/verification.md` reports **all 12 runs
pass** on each model, confirming from the stored weights that `mixed`'s routers are
bit-identical to gold in every layer while `uniform`/`placebo`/`attention` routers sit
exactly at their INT-*N* level ceiling.

## 2. What the conclusions are

**Router protection works, on every cell we tested.** `mixed` beats `uniform` on routing
KL and top-1 flip rate at all three bit-widths on both models — six of six cells, with
disjoint 95% intervals, confirmed by the paired bootstrap (0.00% of 10,000 replicates
favour `uniform` in any cell). The effect grows as precision drops.


| Model | Bits | Routing KL: uniform → mixed (attention) | PPL: uniform → mixed (attention) |
| ----- | ---- | --------------------------------------- | -------------------------------- |
| OLMoE | INT8 | 0.0085 → 0.0064                         | 9.39 → 9.32                      |
| OLMoE | INT4 | 0.0286 → 0.0090 (0.0237)                | 10.49 → 9.69 (9.73)              |
| OLMoE | INT3 | 0.1049 → 0.0385 (0.0777)                | 30.23 → 18.64 (14.36)            |
| Qwen  | INT8 | 0.0235 → 0.0175                         | 9.35 → 9.28                      |
| Qwen  | INT4 | 0.0861 → 0.0355 (0.0691)                | 12.01 → 11.24 (10.51)            |
| Qwen  | INT3 | 0.4901 → 0.3147 (0.3380)                | 9555 → 1236 (288)                |


Gold perplexity is 8.36 (OLMoE) and 7.97 (Qwen). Full numbers and intervals in
`results/<model>/summary.md`; everything below points at a file rather than repeating it.

**The placebo control passed everywhere** — at all three bit-widths on both models it
reproduces `uniform` to several decimals. But state the caveat plainly: it protects a
*single* expert down-projection in one layer, on the compute path for 11.4% of tokens on
OLMoE and 5.0% on Qwen, giving a token×layer exposure gap of 140× / 477× against the
routers. It rules out "any high-precision parameters help" and nothing stronger.

**The** `attention` **control is the interesting one, and it splits.** At INT4 it behaves much
like `uniform` on routing — at OLMoE INT4, KL is 0.0286 (uniform), 0.0237 (attention),
0.0090 (mixed) — recovering only 25.1% (OLMoE) and 33.6% (Qwen) of `mixed`'s routing-KL
reduction despite holding 128× the router parameter budget (3.88% of OLMoE, 2.81% of Qwen,
versus 0.030% / 0.021% for the routers) with the same every-token, every-layer exposure.
That is the evidence for router specificity, and it is an **INT4** statement: at INT3 the
two policies converge, with attention recovering 41.0% of the KL reduction on OLMoE and
86.7% on Qwen. They separate least exactly where the models are most damaged.

On **perplexity** attention is competitive or better: it wins at OLMoE INT3 (14.36 vs
18.64), Qwen INT4 (10.51 vs 11.24) and Qwen INT3, and only at OLMoE INT4 is `mixed`
marginally ahead (9.69 vs 9.73). The honest reading is that the two safeguards fix
different failure modes — protecting the router preserves routing *decisions* (KL, flip
rate), while protecting upstream weights preserves end-task *quality*. That is what Part 2
predicts, since upstream activation drift is a large term in its own right. Routers are far
more efficient per parameter, but we cannot claim router protection is the best use of a
precision budget for perplexity, and the paper should not try.

**Part 2 attribution** (`attribution.md`): of the routing flips caused by exactly one
mechanism, the router's own weights account for 43.4% / 40.3% at INT8 (OLMoE / Qwen),
62.2% / 58.0% at INT4, and 58.0% / 45.4% at INT3. So upstream activation drift dominates
at INT8 and router weights dominate at INT4, on both models. At INT3, OLMoE's 58.0% is
consistent with its INT4 result; Qwen's 45.4% is the only cell below half, but it sits in
the saturated regime we exclude below, so it should not be read as activation dominance
returning at very low precision — we cannot dismiss Qwen INT3 in Part 1 and then quote it
in Part 2. Note
the two mechanisms overlap and are **not additive** — do not quote them as an "X% /
(100−X)%" split. The analysis self-validates: the rebuilt reference cell reproduces gold's
recorded routing decisions at 99.66% (OLMoE) and 99.37% (Qwen) mean top-1 agreement,
worst layer 99.46% / 98.49%.

**Expert collapse** (`collapse.md`): the pooled entropy in `summary.md` showed nothing
because pooling across layers cancels per-layer imbalance. Per layer it is real on Qwen
and genuinely null on OLMoE. At Qwen INT3, `uniform` starves 222 of 1440 expert slots
(15.4%) with 5 layers containing a fully unused expert; `mixed` starves 55 (3.8%) with
zero fully unused experts anywhere. Note that `attention` tracks `mixed` here rather than
`uniform` — 74 slots (5.1%), also with no fully unused expert — so load balance is a third
routing-adjacent metric where the attention control sits near router protection. Router
specificity holds for KL and flip rate, not for collapse. On OLMoE nothing collapses at
any bit-width — 6/1024
dead slots under `uniform` INT3 versus 9/1024 under `mixed`, against a gold baseline of
17/1024 — which is worth reporting as an honest boundary condition rather than buried.

**Qwen INT3 is saturation, not evidence.** Both policies are destroyed (perplexity 9555
and 1236 against a gold of 7.97), so the "win" between them means little as a perplexity
result. The audit confirms this is genuine saturation and not a numerical bug. Make the
perplexity claims at INT8 and INT4; INT3 is useful for the routing and collapse metrics,
which stay in range and behave monotonically.

Known limitations, all documented in `notes/method_audit.md`: single corpus (WikiText-2
only, though the proposal said "WikiText-2/C4"); perplexity carries no confidence
interval anywhere; single seed; the placebo is n=1 draw; and the in-pipeline bit-width
check happened to sample only attention projections on the `uniform`/`mixed` runs, which
is why the offline router verification exists.

## 3. Next steps toward a finished paper

The paper is a skeleton at `paper/main.tex` (ACL format, tables and figures populated,
`\todo` markers where prose is needed) with a verified 23-entry bibliography in
`paper/custom.bib`. It compiles with no overfull boxes and no undefined references.

**The page budget is fine, and the naive reading of it is not.** The `\todo` blocks are
longer than the prose they stand for, so the build as it sits runs to 16 pages. Stub both
macros to `{}` and recompile and the whole document is 8 pages — body ~2.9, references
~0.4, appendix ~4.7. So every table, figure and caption costs under 3 body pages, leaving
roughly 5 of the 8 for prose. Nothing needs to be cut or moved to the appendix. Re-measure
this way rather than reading the page count off the noted build; `paper/README.md` has the
procedure.

In rough priority order:

1. **Write the prose against the** `\todo`**s.** This is the bulk of the remaining work, and
  there is room for it.
2. **Write the mandatory "AI Disclosure and Reflection" section.** Tomer and Yoav have to
  write this themselves; it cannot be delegated.
3. **Read the three OpenReview papers in a browser** — `RouteQuant` (`bPsPPI65hf`) above
   all, since it argues full-precision routers are *insufficient* and is the claim we most
   directly engage. OpenReview blocks non-browser clients, so this is the one task that
   cannot be automated from the cluster. `notes/prior_art.md` has the rest of the
   prior-art verdict: our novelty claim needed narrowing, because EAQuant's Table 4 does
   contain a router-precision comparison on OLMoE. That is already fixed in Related Work,
   but RouteQuant could move it again.
4. **Frame the two hedges deliberately**, since a careful reader will find them anyway:
  the attention control's perplexity result, and the placebo's exposure mismatch.

```
Optional, only if time allows — none of these are blocking, and the audit ranks them:
a layer-distributed placebo (placement-matched, needs ~10 lines of code plus a short GPU
run), attention at INT8 to complete that row, extra placebo seeds, and a single cheap
C4 robustness point (OLMoE, INT4, gold + uniform + mixed, ~35 min).
```

For LaTeX preflight there is a local toolchain:

```bash
export PATH="/home/morg/NLP_2526b/yoavgeva1/texlive/2026/bin/x86_64-linux:$PATH"
cd paper && latexmk -pdf main.tex && pdfinfo main.pdf
```

Overleaf remains authoritative for the submitted artifact; the local build is for
checking page count and overfull boxes before pasting.