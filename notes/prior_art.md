# Prior art: is the novelty claim defensible?

Checked 2026-08-22 by reading the arXiv HTML full text of the two reachable threats.
Fetched with `curl` from the cluster, which works fine; the agent web-fetch tool hangs on
arxiv.org, which is what stalled two earlier attempts at this question.

## 1. Verdict: the claim as drafted is too strong. Weaken it.

The claim under test was:

> Keeping the router in high precision while quantizing experts is already used as a
> baseline in the MoE PTQ literature (EAQuant). But nobody has isolated it as a single
> controlled variable and quantified what it buys on its own.

**The second sentence is false as written.** EAQuant's Table 4 varies router precision as a
controlled condition, on OLMoE-7B, our primary model, holding experts fixed at W3A4:

| Condition (experts fixed at W3A4) | WikiText2 PPL | C4 PPL | Avg. acc (5 tasks) |
|---|---|---|---|
| `w3a4_Rw3a4_Baseline` | 11.09 | 13.95 | 62.33 |
| `w3a4_Rw4a4_Baseline` | 10.94 | 13.68 | 63.00 |
| `w3a4_Rw8a8_Baseline` | 10.77 | 13.59 | 63.30 |

That is a monotone router-precision effect — about 0.32 perplexity and 1.0 accuracy points
from a 3-bit to an 8-bit router — and it is published. Any sentence claiming nobody has
measured a router-precision delta will not survive a reviewer who opens that table.

**But six substantive gaps remain, and they are the contribution.**

1. **It never reaches full precision.** The most-protected router in the paper is W8A8.
   `Rw16`, `R16`, "full-precision router" and "unquantized router" appear zero times in the
   text. The actual proposal under test here — router in BF16, i.e. *exempt from
   quantization entirely* — is not a condition EAQuant runs.
2. **Router precision is entangled with activation precision.** Every condition is
   weight-*and*-activation (`Rw3a4` vs `Rw8a8` moves both at once), so the comparison cannot
   attribute the gain to router weights. Our setting is weight-only, which is what makes
   the variable clean.
3. **It is calibration-based.** The baseline is DuQuant, a rotation method fitted to
   calibration data. Our entire setting is calibration-free — scales come from weight
   tensors alone — so the two measure router protection in different regimes. This is the
   single biggest difference and should lead the positioning.
4. **No routing metric of any kind.** Table 4 reports perplexity and zero-shot accuracy
   only. There is no routing KL, no top-1 flip rate, no Jaccard, no expert-usage entropy,
   no dead-expert count anywhere in the paper. So EAQuant shows *that* router precision
   affects end-task scores and says nothing about *whether or how routing behaviour
   changes*. Every routing-behaviour result we have is unduplicated.
5. **No controls, so no specificity claim.** Without a parameter-matched placebo or an
   exposure-matched control, the table cannot distinguish "the router specifically matters"
   from "any high-precision capacity helps." Our placebo and attention conditions exist
   precisely to answer that, and the attention control's split behaviour shows the question
   is not rhetorical.
6. **No uncertainty and no mechanism.** Single point estimates, no intervals, and no
   decomposition of routing error into router-weight versus upstream-activation sources.

Also worth knowing: those Table 4 rows exist to show that EA-RCA (their routing-consistency
module) helps across router configurations. The router-precision deltas are a by-product of
the baseline column and are **never discussed as a finding**. So "already used as a
baseline, adopted without comment" is fair; "never measured" is not.

### Recommended wording

> Router protection is standard practice rather than a novel proposal: EAQuant runs experts
> at W4A4 with the router at W8A8, and an ablation table incidentally reports that
> degrading the router from W8A8 to W3A4 at fixed W3A4 experts costs 0.32 WikiText-2
> perplexity and 1.0 points of average zero-shot accuracy on OLMoE. That measurement is
> confounded in three ways we remove: it moves router weight and activation precision
> together, it sits on top of a calibrated rotation-based baseline, and it never tests a
> genuinely full-precision router. More importantly, it is scored only on end-task metrics,
> so it cannot say what happens to routing itself. We contribute the calibration-free,
> weight-only isolation of that variable, the first routing-behaviour measurement of it
> (KL, top-1 flips, top-k Jaccard, per-layer load) with bootstrap confidence intervals, two
> control conditions that test whether the effect is router-specific rather than generic,
> and an attribution splitting the residual drift into router-weight and upstream-activation
> sources.

That is narrower than the original claim and considerably harder to attack.

### One comparability trap

EAQuant benchmarks `allenai/OLMoE-1B-7B-0924` — exactly our checkpoint, so the
"same model" claim holds. But it reports **FP16 WikiText-2 perplexity of 7.49** where our
`gold` BF16 run gives **8.3586** on the same checkpoint. That gap is almost certainly
evaluation protocol (sequence length, stride, split handling), but until someone reconciles
it, the paper must claim *the same model*, never *directly comparable numbers*. Any
sentence putting our perplexities beside theirs needs to go or be hedged explicitly.

## 2. Threat ranking

**1. EAQuant (arXiv:2506.13329v3), Fu et al. 2025 — partial hit, must be cited precisely.**
Confirmed: experts W4A4 with router W8A8 throughout; OLMoE-1B-7B is benchmarked; and
*contrary to our previous assumption*, Table 4 does contain a router-precision comparison.
Gaps as enumerated above.

**2. QuantMoE-Bench (arXiv:2406.08155v2), Li et al. — not a threat.** It sweeps bit-widths
across MoE sub-structures, but the router is not one of them. The word "router" appears
twice in the entire paper, both in background prose describing what an MoE is. Its actual
conditions are attention versus experts, first versus last MoE blocks, frequent versus rare
experts, and linear-layer types *within* experts. Its models are Mixtral-8x7B and
DeepSeek-MoE-16B, not OLMoE, and it is GPTQ-based with 512 calibration sequences.

*Trap for a careless reader:* its Figure 5 analyses the "gate projection" per-layer. That is
the FFN gate projection inside each expert, not the router gate. Do not cite it as router
analysis, and do not let a reviewer think we did.

## 3. Recommended Related Work order

1. **MoE PTQ generally** — establish that expert quantization is the well-studied part.
2. **Router protection as inherited practice** — EAQuant as the concrete instance, stated
   accurately: router at W8A8, plus the Table 4 delta. Concede this openly and early; it is
   much stronger than being caught omitting it.
3. **The disagreement we sit inside** — RouteQuant arguing full-precision routers are
   insufficient against the practice of protecting them. This is the live question our
   controlled design speaks to, and it is why isolating the variable is worth doing even
   though a confounded measurement exists.
4. **What is missing across all of it** — nobody reports routing-behaviour metrics for a
   router-precision manipulation, nobody controls for generic high-precision capacity, and
   nobody separates router-weight error from upstream drift. Then state the contribution.

## 4. What could not be checked

Everything on OpenReview: **RouteQuant** (`bPsPPI65hf`), **ExpertQuant / "Router Choice
Matters"** (`kPgLp47bJf`), and **SRA-MoE**. OpenReview returns 403 to unauthenticated
clients and its forum pages sit behind a browser check, so these were not readable from the
cluster. They need a human with a browser. Priority order:

- **RouteQuant is the important one.** If it already presents a controlled router-precision
  ablation to support "freezing the router is insufficient", that is a direct hit on the
  same variable and the framing in section 3 above has to be revisited.
- ExpertQuant matters less; we cite it for the near-neighbour rank-flip observation.
- SRA-MoE is reportedly close to our framing (routers kept full precision, yet upstream
  error still shifts routing — essentially our Part 2 premise). Worth ten minutes.

Also unchecked: **GEMQ** (arXiv:2605.23078) and **arXiv:2603.02217** (router calibration),
both fetchable by `curl` if wanted; and item 7 of the original brief — whether random
parameter-matched placebo controls have precedent in the compression literature, and
whether anyone has decomposed quantization-induced routing error into router-weight versus
upstream-activation components. Neither affects the verdict above, since both would only
add precedent for methodology we would still be first to apply here.
