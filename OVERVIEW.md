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

**Why sweep bit-widths instead of only INT8, as the proposal said?** INT8 is nearly
lossless. The difference between `mixed` and `uniform` there will almost certainly be
smaller than the measurement noise, which would leave us unable to claim anything in
either direction. If the effect exists, it lives at 4 and 3 bits. Sweeping also gives a
*trend* rather than a single point.

### The decision gate

After the sweep, one question decides what happens next: **does `mixed` beat `uniform` by
more than the confidence intervals, at any bit-width?** `scripts/analyze.py` prints the
verdict rather than leaving us to eyeball overlapping error bars.

- **If yes** — run the **parameter-matched placebo** control before claiming anything.
  Routers are roughly 0.02% of the model. If protecting a *random* 0.02% helps just as
  much, the effect was about keeping *some* weights in high precision, not about routers
  being special. Pass that control and Part 1 is the whole paper.
- **If no** — the null result is honest and complete on its own, and Part 2 turns it into
  a contribution.

There is a third outcome the gate reports separately: `mixed` can be measurably *worse*
than `uniform`, with the intervals disjoint in the other direction. That is a finding
rather than a null result, so it is not folded in with "no separation" — leaving the
routers in high precision while everything around them is quantized creates a scale
mismatch that can cost more than it saves, and that is worth reporting.

### Part 2 — why it failed (only if Part 1 is null)

Routing drift has exactly two possible sources: rounding in the router's **own weights**,
and drift in the **hidden states arriving** at the router. Part 1 cannot tell them apart.

We record each router's input activations during the Part 1 runs, so all four
combinations are offline matrix multiplications rather than new cluster jobs:

|  | gold router weights | quantized router weights |
|--|--|--|
| **gold activations** | A: reference | B: router weight error only |
| **quantized activations** | C: activation drift only (= `mixed`) | D: both (= `uniform`) |

Comparing **B against C** gives the headline sentence: *"X% of top-k routing flips are
attributable to router weights, and (100−X)% to upstream activation drift."*

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
3. **Staged Part 1 / Part 2 framing**, so a null result still produces a contribution.
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
