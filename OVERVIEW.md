# What we are doing and why

NLP final project, Tel Aviv University. Tomer Alfandary and Yoav Geva.

## The question in plain words

A Mixture-of-Experts model works like a hospital. Every patient (token) first sees a
**receptionist** — the router, also called the gate. She glances at them and sends them
to a handful of **specialists** (the experts) out of dozens available. Only those few
specialists do any work, which is what makes MoE models cheap to run despite being huge.

**Quantization** is compressing everyone's notes to save filing space: instead of writing
numbers at full precision, you round them off. Doing this to the specialists saves an
enormous amount, because they are almost the entire hospital. But if you also round the
receptionist's notes, she may start sending patients to the wrong specialists, and a
wrong referral is far worse than slightly sloppy notes.

**Our proposal:** the receptionist is one person out of thousands of staff. Keeping her
notes at full precision costs essentially nothing. So compress every specialist, leave
her alone, and see whether the hospital still works.

**The catch we are ready for:** even a perfect receptionist misroutes if the *patient
chart arriving at her desk* was already garbled by compressed departments upstream. If
that turns out to dominate, protecting her will not help much.

## The two parts

### Part 1 — the question we promised the lecturer (this repository)

> Does keeping only the router in BF16, while quantizing all expert weights, reduce
> routing drift and perplexity loss relative to uniform quantization, with no calibration?

Three configurations, differing in exactly one respect:

| Config | Experts | Routers | Role |
|--------|---------|---------|------|
| **gold** | BF16 | BF16 | Upper bound |
| **uniform** | INT*N* | INT*N* | Negative baseline |
| **mixed** | INT*N* | BF16 | The proposed safeguard |

Run across INT8, INT4, and INT3 on two architectures (OLMoE-1B-7B and
Qwen1.5-MoE-A2.7B), measuring routing KL divergence, top-*k* expert mismatch,
expert-usage entropy, perplexity, and output-distribution drift — all with bootstrap
confidence intervals.

The expert-usage metrics are reported **per layer**, in `results/*/collapse.md`. The
pooled figure in `results/*/summary.md` concatenates every layer's tokens into one
histogram before measuring, which cancels imbalances that point in different directions
across layers; the per-layer view is what actually detects expert collapse.

**Why sweep bit-widths instead of only INT8, as the proposal said?** INT8 is nearly
lossless. The difference between `mixed` and `uniform` there will almost certainly be
smaller than the measurement noise, which would leave us unable to claim anything in
either direction. If the effect exists, it lives at 4 and 3 bits. Sweeping also gives a
*trend* rather than a single point.

### The decision gate

After the sweep, one question decides what happens next: **does `mixed` beat `uniform` by
more than the confidence intervals, at any bit-width?** `scripts/analyze.py` prints the
verdict rather than leaving us to eyeball overlapping error bars.

The gate asks whether two separately-built bootstrap intervals are disjoint. That is a
*conservative* stand-in for the correct test: both arms are measured on the same tokens with
the same resamples, so the difference can be bootstrapped directly, which cancels the
sequence-level variation they share. `results/*/paired_bootstrap.md` reports the paired
version — every cell separates from zero there too, with intervals 1.8× to 10× narrower.

- **If yes** — run the **parameter-count-matched placebo** control before claiming
  anything. If protecting a randomly chosen module of the same size as the whole router
  budget helps just as much, the effect was about keeping *some* weights in high
  precision, not about routers being special.
- **If no** — the null result is honest and complete on its own, and Part 2 turns it into
  a contribution.

**What the placebo does and does not control for.** It is matched on parameter count and
on nothing else. With `placebo_seed=0` the selection is a *single* module —
`model.layers.14.mlp.experts.10.down_proj` on OLMoE (1.000000× the router budget) and
`model.layers.19.mlp.experts.45.down_proj` on Qwen (0.977778×) — against 16 and 24 routers
spread across every layer. Measured from the gold router logits, that module is on the
compute path for 11.43% (OLMoE) and 5.04% (Qwen) of routing tokens in one layer, whereas
the routers are read by 100% of tokens in every layer: a token×layer exposure gap of about
140× and 477×. So the control rules out *"any high-precision parameters help"* and does
**not** address *"any high-precision island on the every-token, every-layer main path
helps"*. The `attention` policy is the run that would close that second gap; it is
implemented and has not been run. `results/*/verification.md` records the derivation and
the exposure measurement.

There is a third outcome the gate reports separately: `mixed` can be measurably *worse*
than `uniform`, with the intervals disjoint in the other direction. That is a finding
rather than a null result, so it is not folded in with "no separation" — leaving the
routers in high precision while everything around them is quantized creates a scale
mismatch that can cost more than it saves, and that is worth reporting.

### Part 2 — where the routing damage comes from

Routing drift has exactly two possible sources: rounding in the router's **own weights**,
and drift in the **hidden states arriving** at the router. Part 1 cannot tell them apart.
This was originally framed as a contingency for a null Part 1; Part 1 came out strongly
positive and we ran it anyway, as a mechanism analysis rather than a fallback.

We record each router's input activations during the Part 1 runs, so all four
combinations are offline matrix multiplications rather than new cluster jobs:

|  | gold router weights | quantized router weights |
|--|--|--|
| **gold activations** | A: reference | B: router weight error only |
| **quantized activations** | C: activation drift only (= `mixed`) | D: both mechanisms |

Cell C uses the `mixed` run's activations, and that is what makes it exactly the `mixed`
condition: `mixed` and `uniform` quantize the same attention and expert weights, so
`mixed`'s hidden states carry upstream drift with no feedback from perturbed routing
decisions. The same choice means **cell D is not the `uniform` run**. D pairs quantized
router weights with `mixed`-run activations, so it omits that feedback — the effect of
changed routing on later layers' hidden states. The gap is measurable and grows with
damage: on Qwen at INT3, D rebuilds 68.58% top-1 flips against 72.48% recorded for
`uniform`. That difference is a small extra result, not an error.

Comparing **B against C** gives the headline sentence: *"of the top-k routing flips caused
by exactly one mechanism, X% come from the router's own weights"* — 62.2% on OLMoE at INT4
and 58.0% on Qwen at INT4.

That is deliberately a share of a well-defined set rather than a partition. The two
mechanisms are **not additive**: routing is a top-*k* argmax over a softmax, so a token
that either source alone would have flipped is counted once in D but twice across B and C.
The
interaction residual is large and negative on both models (−36% to −51%), so B and C are
not shares of anything that sums to 100%, and a "X% / (100−X)%" split is exactly the claim
the non-additivity forbids. `results/*/attribution.md` reports the residual with its own
bootstrap interval alongside every share.

We have already seen this effect in miniature. `tests/test_integration.py` builds a
`mixed` model whose router weights are *bit-identical* to gold, and its routing KL is
still non-zero, purely because the activations reaching those routers passed through
quantized attention and expert layers first. That is the Part 2 premise, reproduced on a
synthetic model in under a second.

## Where the literature stands

Being straight about this shapes how we frame whichever result we get, and it matters for
the literature-review portion of the grade.

- **EAQuant** ([arXiv:2506.13329](https://arxiv.org/abs/2506.13329), Fu et al.) runs
  experts at W4A4 *with the router at W8A8*. Router protection is already their baseline,
  so we cannot claim it as novel — but nobody has isolated how much it buys on its own.
- **"Router Choice Matters"** (Fang & Huang, ICLR 2026 submission) reports that most
  routing errors are near-neighbour rank flips at the top-*k* boundary. This is the
  strongest reason to expect Part 1 to come out null.
- **VSRAQ** ([arXiv:2606.05688](https://arxiv.org/html/2606.05688v1)) is the correct
  citation for the "Park et al. (2026)" entry in our proposal; the "Anonymous (2026)"
  entry is Fang & Huang. Both must be fixed before submission.
- **[Examining PTQ for MoE: A Benchmark](https://arxiv.org/abs/2406.08155)** sweeps
  bit-widths across MoE sub-structures and overlaps with us directly. Must be cited and
  differentiated.
- **MoQE** found expert FFNs tolerate 2-bit quantization while attention does not, which
  is why "protect attention instead" is available as an extra control policy.

## Deviations from the approved proposal

To be documented plainly in the paper's Experimental Setup section:

1. **Model substitution.** The proposal named `Qwen/Qwen2.5-MoE-1.4B-A14B`, which does
   not exist on HuggingFace and is internally contradictory (1.4B total with 14B active
   is impossible). We use **OLMoE-1B-7B** as the primary model — small enough to iterate
   on, and one of the three models EAQuant benchmarks, so our numbers are comparable —
   and **Qwen1.5-MoE-A2.7B** as the second architecture.
2. **Bit-width sweep** rather than INT8 alone, for the reason given above.
3. **Staged Part 1 / Part 2 framing**, so a null result would still have produced a
   contribution. Part 1 came out positive and Part 2 was run regardless, as a mechanism
   analysis; the staging shaped the design but not the final scope.
4. The proposal's contingency of "dynamically scaling routing scores by activation size"
   is replaced, if needed, by **margin-aware top-*k* widening**, which targets the
   near-neighbour rank-flip failure mode identified in the literature.

## Three traps we build against

None of these crash. They produce plausible-looking numbers that mean nothing, which is
why `src/moequant/verify.py` exists and why every run refuses to proceed if a check fails.

1. **DeepSeek's router is not an `nn.Linear`.** Its `MoEGate` holds a bare `nn.Parameter`
   and calls `F.linear`, so a quantizer that only swaps `nn.Linear` never touches it —
   `uniform` and `mixed` would be the same model. Its `forward` also returns
   `(topk_idx, topk_weight, aux_loss)` rather than logits, so a naive hook captures expert
   indices and any KL computed on them is garbage.
2. **transformers 5.x fuses MoE experts into 3D parameter stacks** that `nn.Linear`-based
   walkers silently skip, leaving the "quantized" experts at full precision. We pin
   transformers to 4.x and `scripts/inspect_model.py` reports which layout it found.
3. **Compute capability.** LLM.int8() needs 7.5+, and BF16 is not native below Ampere.
   The lab machine's GTX TITAN X cards are 5.2, so everything runs on the cluster.

## Reading order

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how the code is put together and why
- **[RUNNING.md](RUNNING.md)** — installation, cluster setup, and how to execute a sweep
