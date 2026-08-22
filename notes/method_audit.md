# Methodology audit

Read-only audit of the finished Part 1 / Part 2 experiments, done before the paper is
written. Everything below was verified against the code and the artifacts on disk, not
against the prose in `OVERVIEW.md` / `ARCHITECTURE.md`. Where I re-derived something
myself the command is given so it can be reproduced.

No code, config, result, or Slurm job was modified.

---

## 1. Placebo verdict

### What it actually protects

**Exactly one module, in one layer, on each model.** Not a set. One `nn.Linear`.

| Model | Protected module | Params | Ratio to router budget | Layer |
|-------|------------------|--------|------------------------|-------|
| OLMoE-1B-7B | `model.layers.14.mlp.experts.10.down_proj` | 2,097,152 | **1.000000×** | 14 of 16 |
| Qwen1.5-MoE-A2.7B | `model.layers.19.mlp.experts.45.down_proj` | 2,883,584 | **0.977778×** | 19 of 24 |

Re-derived offline from the meta-device skeleton (config download only, no GPU):

```bash
python - <<'PY'
from moequant.registry import get_spec
from moequant.quantize import sample_placebo_modules
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
for key in ("olmoe", "qwen"):
    spec = get_spec(key)
    cfg = AutoConfig.from_pretrained(spec.model_id)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    print(key, sample_placebo_modules(m, spec, seed=0))
PY
```

This is consistent with the correctness gates: `uniform` 3152 / `placebo` 3151 /
`mixed` 3136 on OLMoE (3152 − 3136 = 16 = the router count), and 4536 / 4535 / 4512 on
Qwen (4536 − 4512 = 24). The `audit.num_protected_modules == 3` in the placebo
`metrics.json` files is `embed_tokens` + `lm_head` + **one** placebo module.

### Why the selection rule collapses to a single module

`sample_placebo_modules` (`src/moequant/quantize.py:177`) sets `budget` to the **total**
router parameter count across *all* layers, filters the candidate pool to modules no
larger than that whole budget, shuffles with `random.Random(seed=cfg.placebo_seed=0)`,
then greedily accumulates and **breaks as soon as `accumulated >= 0.75 * budget`**.

The arithmetic of these two checkpoints makes the first draw sufficient on its own:

- OLMoE: router budget = 16 × 64 × 2048 = 2,097,152. Every eligible candidate is an
  expert projection of `2048 × 1024 = 2,097,152` params — *exactly* the whole
  16-layer router budget. The attention projections (4,194,304) are excluded by the
  `n <= budget` pool filter. So the pool is 3072 modules, all identically sized, and
  one draw lands at 1.000× the budget.
- Qwen: router budget = 24 × 60 × 2048 = 2,949,120. Expert projections are
  `2048 × 1408 = 2,883,584` = 0.978× the budget. Pool is 4320 of those plus 24
  `shared_expert_gate` modules (2048 params each); everything else is over budget.
  One draw of an expert projection lands in tolerance immediately.

So the "greedily accumulate small modules until within tolerance" behaviour described in
`ARCHITECTURE.md` never happens on either real model. It is a one-shot draw. The
`tolerance=0.25` band is never exercised either.

### What is and is not matched

| Dimension | Matched? | Detail |
|-----------|----------|--------|
| Total parameter count | **Yes** | 1.000× (OLMoE), 0.978× (Qwen) |
| Number of modules | No | 1 vs 16 (OLMoE) / 24 (Qwen) |
| Distribution across layers | **No** | one mid-depth layer vs every layer |
| Module role / function | No | expert down-projection vs routing gate |
| Position in the network | No | layer 14/16 and 19/24 only |
| **Activation exposure** | **No, by ~2 orders of magnitude** | see below |

The exposure mismatch is the one that actually matters and it is worse than the placement
mismatch. A router is read by **100% of tokens in every layer**. Measured from the gold
logits in `results/<model>/gold/artifacts.pt`, the placebo-protected expert is on the
compute path for:

- OLMoE, layer 14 expert 10: **11.43%** of the 32,768 routing tokens (top-8 of 64)
- Qwen, layer 19 expert 45: **5.04%** of the 32,768 routing tokens (top-4 of 60)

Counting token×layer forward events, the routers are exposed
16 / 0.1143 ≈ **140×** more (OLMoE) and 24 / 0.0504 ≈ **476×** more (Qwen) than the
placebo's protected parameters. The placebo un-quantizes 0.03% of the weights in one
layer's expert bank, which ~90–95% of tokens never touch.

### Is "parameter-matched placebo" accurate?

Literally yes, and *only* that. It is matched on parameter count and on nothing else.
Used without qualification the phrase invites the reader to assume a matched control in
the experimental-design sense, which this is not.

**Defensible wording** (use this or something like it):

> As a control we ran a *parameter-count-matched* placebo: the same INT-*N* policy as
> `uniform`, except that a single randomly chosen non-router module — matched to the
> total router parameter budget (1.000× on OLMoE, 0.978× on Qwen, `placebo_seed=0`) — is
> left in BF16. On OLMoE this is `model.layers.14.mlp.experts.10.down_proj`; on Qwen,
> `model.layers.19.mlp.experts.45.down_proj`. The control matches the router budget in
> parameter count only: the routers are 16 (24) small modules read by every token in
> every layer, whereas the placebo is one large module in one layer, on the compute path
> for 11.4% (5.0%) of tokens.

**Wording to avoid:** "matched control", "an equivalent 0.02% of parameters",
"protecting a random 0.02% of the model" — the last one is the phrasing currently in
`OVERVIEW.md`, `RUNNING.md`, `analyze.py`'s verdict string and `compare_models.py`'s
legend label ("placebo (random 0.02% protected)"). It is true by parameter count and
misleading about everything else.

### Is it sufficient to support the router-specificity sentence?

**No, not on its own.** The sentence "the effect is specific to the routers rather than
to keeping some weights in high precision" has two competing explanations to kill, and
the current placebo only kills the weaker one:

1. *"Any 0.02% of weights in BF16 helps"* — killed. The placebo lands on `uniform` to
   six decimals (OLMoE INT4 KL 0.028579 vs 0.028579; INT3 0.104929 vs 0.104926).
2. *"Any high-precision island on the every-token, every-layer main path helps"* — **not
   addressed at all**, because the placebo deliberately is not on that path.

And the placebo's null result is close to guaranteed by construction: restoring one
expert projection out of 3072 (4320), touched by ~1 token in 9 (1 in 20), cannot plausibly
move a pooled routing metric. The control passed, but it was not a demanding test, and a
grader who works out what it protects will say so. My concern is that the paper currently
rests its central specificity claim on this one run.

Note also: the placebo module FQNs are **not recorded in any `metrics.json`**. Only
`num_protected_modules: 3`. The choice is deterministic given `placebo_seed`, so nothing
is unrecoverable, but the results files do not document what the control controlled and
the paper cannot cite them for it. Put the derived names and the derivation command in
the appendix.

### Is a layer-distributed variant needed?

Yes — but there is a hard obstacle you should know about before promising one, and it
changes the recommendation.

**You cannot have parameter-matching and layer-distribution at the same time at
`nn.Linear` granularity in either architecture.** A per-layer size-matched placebo would
need a ~131,072-param (OLMoE) or ~122,880-param (Qwen) module in each layer. The smallest
non-router `nn.Linear` in OLMoE is an expert projection at 2,097,152 params — 16× too
big. Qwen has a 2048-param `shared_expert_gate` per layer (24× too small) and then jumps
straight to 2,883,584. Getting a genuine per-layer size match would require quantizing a
*row slice* of a module, which torchao's `FqnToConfig` cannot express (it is per-module),
so it would mean writing custom quantization. Not worth it.

So the two honest options are:

- **(a) Recommended.** Keep the current parameter-matched placebo, describe it precisely
  as above, and add a second control that is *placement*-matched but larger: one randomly
  chosen expert projection **per layer**, i.e. 16 × 2,097,152 = 0.48% of OLMoE and
  24 × 2,883,584 = 0.48% of Qwen. This is not parameter-matched — it protects 16× / 24×
  the router budget — which makes it a strictly *harder* test, and that is the point: if
  protecting 16× more parameters spread across every layer still does not reduce routing
  drift, the router-specificity claim is far stronger than the current placebo makes it.
  Needs ~10 lines of new code (`per_layer=True` mode in `sample_placebo_modules`) plus a
  short GPU run.
- **(b) Cheaper and available today: the `attention` policy.** `quantize.py` already
  implements it, `verify.py` already asserts it, and it has *never been run*. It protects
  attention instead of routers: 3.88% of OLMoE (268,435,456 params) and 2.81% of Qwen
  (402,800,640). Attention is on the compute path for 100% of tokens in every layer —
  exactly the exposure profile the current placebo lacks. This is the single most
  informative missing run in the project and it needs no new code.

My recommendation: run (b) now, and add (a) if time allows. Together they close the
exposure confound from both directions and the paper can then say the specificity
sentence without hedging.

---

## 2. Dataset finding

### What was actually run

**WikiText-2 only.** Confirmed from `configs/olmoe.yaml`, `configs/qwen.yaml`, the
`config` block of every `metrics.json`, and `src/moequant/data.py`.

| Setting | Value | Source |
|---------|-------|--------|
| Corpus | `wikitext` / `wikitext-2-raw-v1` | `data.py:CORPORA` |
| Split | **test** | same |
| Assembly | all documents joined with `"\n\n"`, tokenized as one stream | `data.py:load_token_stream` |
| Total tokens | 288,730 (OLMoE) / 299,078 (Qwen) | `dataset.total_tokens` |
| PPL window | 1024 tokens | `ppl_seq_len` |
| PPL stride | 1024 (disjoint; every token scored exactly once) | `ppl_stride` |
| PPL windows | 282 (OLMoE) / 293 (Qwen), uncapped | tqdm bars in `logs/*.err` |
| Scored tokens | 288,448 / 298,785 | `lm.scored_tokens` |
| Routing subset | 128 sequences × 256 tokens = 32,768 tokens per layer | `routing_sequences`, `routing_seq_len` |
| Routing sampling | `default_rng(42).choice` over disjoint 256-token blocks, sorted | `data.py:routing_batches` |
| Captured activations | strided ×8 → 4,096 tokens per layer | `input_stride` |
| Output-drift positions | every 16th scoring position, top-M = 64 | `position_stride`, `top_m` |

### Is C4 supported?

**Yes, the data loading already supports it. No new code is needed for the corpus
itself.** `data.py:CORPORA` has a complete `"c4"` entry (`allenai/c4`, `validation`
split, shard `en/c4-validation.00000-of-00008.json.gz`), with a comment explaining why no
config `name` is passed. `ExperimentConfig.corpus` is a plain field.

What *is* missing is plumbing, and it is small:

1. `scripts/run.py` exposes no `--corpus` flag. Either add one (three lines: `argparse`
   entry plus an `overrides` key) or, with **zero code change**, copy
   `configs/olmoe.yaml` to `configs/olmoe_c4.yaml` with `corpus: c4`.
2. A C4 run needs its **own gold run**. `artifacts.pt` carries a SHA-256 token
   fingerprint and `runner._check_alignment` rejects any candidate whose corpus differs;
   `output_reference.pt` is likewise corpus-specific. So a C4 sweep must go to a separate
   `--results-dir` (e.g. `results_c4`) and re-run gold.
3. The C4 shard is **not in the cache** (`datasets_cache/` contains only `wikitext`), so
   the first run pays a ~300 MB download. That is a real risk on a preemptible partition,
   though `--skip-existing` limits the damage.

Total cost: one gold + N candidates per model. See §5 for the concrete recommendation.

### Are routing metrics and perplexity on the same data?

Yes. Both come from the same flat token stream of the WikiText-2 **test** split. The
routing subset (32,768 tokens) is a seeded selection of disjoint 256-token blocks drawn
from the same stream the 282/293 perplexity windows tile. There is no held-out split
anywhere; there does not need to be.

### Data-leakage assessment (state this explicitly in the paper)

**There is no train/test leakage.** This is calibration-free PTQ: nothing is fitted to
any data at any point. The quantization scales are computed per output channel from the
weight tensors alone (`IntxWeightOnlyConfig` with `PerAxis(0)`), never from activations,
so the evaluation corpus never influences the model. There is no calibration set, no
GPTQ/AWQ-style reconstruction, no hyperparameter selected on the evaluation data, and the
three policies are fixed a priori. Evaluating on the test split is the standard protocol
for this class of work.

Two honest caveats worth one sentence each rather than being buried:

- **Routing metrics and perplexity are not statistically independent.** They are measured
  on overlapping tokens from the same corpus, so they should not be treated as two
  independent confirmations of the same effect. (They *are* different quantities and the
  routing metrics carry bootstrap intervals while perplexity does not — see §4.2.)
- **Single-domain evaluation.** WikiText-2 is in-domain, encyclopedic English that both
  base models almost certainly saw near during pretraining. Nothing about the comparison
  is invalidated, but the *magnitudes* (e.g. gold PPL 8.36 / 7.97) are best-case and the
  paper should not generalize them.

### Run more, or disclose?

**Disclose as the primary answer; add one cheap C4 point if queue time allows.**

Disclose, because:

- The proposal says "standard text benchmarks (WikiText-2/C4)" — a slash. "WikiText-2"
  is a defensible reading of the commitment, and the existing "Deviations from the
  approved proposal" section in `OVERVIEW.md` is the natural place to say plainly that
  only WikiText-2 was used.
- The effect being claimed is huge, monotone in bit-width, and replicates across two
  architectures with disjoint intervals at every bit-width. A second corpus is a
  robustness nicety, not a validity fix. No reviewer is going to believe that a 3×
  reduction in routing KL is a WikiText-2 artifact.
- The cost is a full extra gold per model, which is the most expensive single run there
  is.

But the cheapest possible version of "we checked" is genuinely worth it: **OLMoE, C4,
INT4 only, gold + uniform + mixed = 3 runs, ~35 minutes.** That converts a paragraph of
prose into a data point and pre-empts the obvious question. See §5, item 5.

---

## 3. Expert collapse

### Is the current metric adequate?

**No. The metric family is fine; the aggregation is what breaks it.**

`metrics.py:usage_stats` builds the top-*k* selection-count distribution over experts,
normalizes to a load vector, and reports its Shannon entropy divided by
`log(num_experts)`. That is a reasonable balance measure. The problem is that the number
the summary tables and figures report is
`routing.pooled.cand_usage.marginal_entropy_normalized`, and `compare_routing` builds
`pooled` by **concatenating tokens across all layers** before calling `usage_stats`. So
16 (or 24) independent routers' selections are collapsed into a single 64-bin (60-bin)
histogram. Expert 5 in layer 0 and expert 5 in layer 12 are different experts; imbalance
in opposite directions across layers cancels.

How much does that cost? Compare the same OLMoE gold run:

| Aggregation | Normalized entropy | Max/mean load |
|-------------|--------------------|---------------|
| pooled over 16 layers | 0.9975 | 1.40 |
| layer 0 alone | **0.9699** | **4.01** |

Same data, same run. The pooled number understates the imbalance by roughly an order of
magnitude in effect size.

On top of that, normalized entropy is intrinsically the wrong shape for this job:
uniform load is its *maximum*, so its gradient there is zero and it is maximally
insensitive exactly where the null hypothesis sits. A distribution where one expert of 64
takes 4× its fair share and the rest split the remainder still scores ~0.99.

**Would a real collapse move it? Barely.** Qwen at uniform INT3 *does* collapse hard, and
the reported pooled entropy only fell 0.9996 → 0.9929 — which reads as a null finding.
Per layer, in the same run, entropy falls to **0.7585**, max/mean load reaches **12.53**,
and one layer has **18 of 60** experts below a tenth of fair share. The headline metric
hid a real, large effect.

So: this is **not** a genuine null finding on Qwen. It is a real collapse, present in
data already on disk, obscured by pooling. On OLMoE it *is* close to a genuine null
(6 dead expert-slots of 1024 at uniform INT3), which is itself a result worth reporting —
collapse appears where the model is being pushed past its breaking point, and OLMoE at
INT3 is not there yet while Qwen is.

### What is already saved — the important part

**Everything needed for sharper metrics is on disk. No GPU run is required for any of
this.** Two tiers:

#### Tier 1 — already computed, zero compute needed

`results/<model>/<run>/metrics.json` → `routing.per_layer.<L>.cand_usage` and
`.gold_usage`, for **every layer of every run**, each containing:

- `marginal_entropy`, `marginal_entropy_normalized`
- `mean_token_entropy` (router confidence, separate from load imbalance — good design)
- `dead_experts` — count with load < 0.1 × uniform
- `unused_experts` — count with exactly zero selections
- `max_over_mean_load`
- `num_experts`, `num_tokens` (full 32,768-token resolution)

This is a complete layer-wise collapse table that is currently plotted nowhere and
tabulated nowhere. It is sitting unused. Pulled out right now, at INT3:

| Model / policy | Dead expert-slots (sum over layers) | Layers with ≥1 fully unused expert | Worst-layer norm. entropy | Worst-layer max/mean |
|---|---|---|---|---|
| Qwen `uniform` INT3 | **222 / 1440 (15.4%)** | 5 | 0.7585 | 12.53 |
| Qwen `mixed` INT3 | **55 / 1440 (3.8%)** | 0 | 0.8752 | 9.94 |
| Qwen `placebo` INT3 | 220 / 1440 | 5 | 0.7585 | 12.53 |
| OLMoE `uniform` INT3 | 6 / 1024 | 0 | 0.9389 | 5.86 |
| OLMoE `mixed` INT3 | 9 / 1024 | 0 | 0.9425 | 4.69 |

That first pair is the proposal's promise delivered verbatim: quantization causes expert
collapse (Qwen INT3, 15.4% of expert slots starved, 5 layers with experts receiving zero
tokens), and the router safeguard keeps the rare experts working (down to 3.8%, no fully
unused expert anywhere). `placebo` landing next to `uniform` is the control passing on
this metric too. And OLMoE showing nothing is the honest boundary condition.

*Resolved.* This analysis now exists as `results/<model>/collapse.md`. Two counts above
were read off a quick pull and are corrected here to match it: 5 layers with a fully
unused expert rather than 6, and `placebo` at 220 rather than exactly 222. The finished
table also carries `attention`, which sits at 74 / 1440 with no fully unused expert —
closer to `mixed` than to `uniform`, which is a qualification on router specificity that
this audit could not have seen.

#### Tier 2 — computable offline from the `.pt` artifacts

For Gini, top-10% load share, KL(load ‖ uniform), max share, and full per-expert
histograms you need per-expert counts, which are not stored as such. They are fully
recoverable:

- **Gold, full resolution:** `results/<model>/gold/artifacts.pt` → `logits[layer]`, shape
  `[32768, num_experts]`, all tokens. `topk_mask(...).sum(0)` gives exact counts.
- **Every candidate run, 1-in-8 resolution:**
  `results/<model>/<policy>_int<bits>/router_inputs.pt` contains
  `inputs[layer]` (`[4096, 2048]` fp16, every 8th token) **and**
  `router_weights[layer]` (`[num_experts, 2048]` fp32 — the *dequantized weights that run
  actually used*). Then
  `logits = F.linear(inputs[layer].float(), router_weights[layer].float())`
  reproduces that run's router logits exactly at those positions. This is the same
  reconstruction `attribution.py` already performs and validates: `check_reconstruction`
  reports 99.66% (OLMoE) / 99.37% (Qwen) top-1 agreement with the recorded gold logits.

I verified this end-to-end. Measured at INT3, per layer then averaged, 4096 strided
tokens per layer:

| Model | Policy | Gini (mean/max) | Unused (sum) | Max share | Top-10% share | KL(load‖unif) mean |
|---|---|---|---|---|---|---|
| Qwen | gold | 0.167 / 0.240 | 0 | 3.69× | 0.162 | 0.0465 |
| Qwen | mixed INT3 | 0.382 / 0.526 | 1 | 9.66× | 0.276 | 0.2693 |
| Qwen | uniform INT3 | **0.551 / 0.708** | **23** | **12.07×** | **0.389** | **0.5717** |
| OLMoE | gold | 0.273 / 0.345 | 0 | 3.98× | 0.200 | 0.1307 |
| OLMoE | mixed INT3 | 0.284 / 0.374 | 0 | 4.59× | 0.206 | 0.1405 |
| OLMoE | uniform INT3 | 0.307 / 0.384 | 0 | 5.71× | 0.214 | 0.1630 |

Qwen's KL of the load distribution against uniform goes 0.047 → 0.269 → 0.572, a **12×**
increase under uniform quantization and a clean 2.1× reduction from router protection.
That is a far sharper instrument than a normalized entropy that moved 0.9996 → 0.9929.

Sketch of the offline script (~40 lines, no GPU, ~1 min per model per bit-width):

```python
import torch, numpy as np
from moequant.attribution import strided_positions
from moequant.metrics import topk_mask

gold = torch.load(f"results/{m}/gold/artifacts.pt", map_location="cpu",
                  mmap=True, weights_only=False)
k = int(gold["topology"]["top_k"])
mask = torch.from_numpy(strided_positions(np.asarray(gold["groups"]), 8))

# gold, full resolution: counts = topk_mask(gold["logits"][L].float(), k).sum(0)
# gold at matched resolution: gold["logits"][L][mask]
d = torch.load(f"results/{m}/{policy}_int{bits}/router_inputs.pt", map_location="cpu",
               mmap=True, weights_only=False)
for L in sorted(d["inputs"]):
    lg = torch.nn.functional.linear(d["inputs"][L].float(), d["router_weights"][L].float())
    counts = topk_mask(lg, k).sum(0).double()
    p = counts / counts.sum()          # then Gini / max share / top-10% / KL vs uniform
```

Use `mmap=True` — the files are 277 MB (OLMoE) to 414 MB (Qwen) each.

**One caveat to state in the paper.** Candidate metrics from Tier 2 are on 1 token in 8
(4096 tokens/layer → ~512 expected selections per expert on OLMoE, ~273 on Qwen). That is
plenty for Gini, max share, top-10% share and KL, but it inflates zero-count and
near-zero-count statistics: gold shows 18 "dead" expert-slots at strided resolution on
OLMoE versus 0 at full resolution. So report **dead/unused counts from the Tier 1 per-layer
fields** (full 32,768 tokens) and **Gini / shares / KL from the Tier 2 reconstruction**
(comparing candidates against gold at the same strided resolution, which the table above
does). Say which resolution each number uses.

### Verdict

The proposal's expert-collapse commitment can be **delivered in full with zero GPU
time**, and delivering it turns an apparent null into one of the paper's better results.
This is the highest-value item in the entire audit. Do it before queueing anything.

---

## 4. Other soft spots, ranked by how likely a grader is to raise them

### 4.1 The disjoint-interval test is not a paired test (most likely to be raised)

**Resampling is genuinely at the sequence level — this part is correct.**
`metrics.bootstrap_ci` takes `groups` (a sequence id per token, built by
`data.routing_batches` as `np.repeat(np.arange(128), 256)`), reduces to per-group sums and
counts with `bincount`, and resamples group indices. `tests/test_metrics.py` constructs
data with strong between-sequence and negligible within-sequence variance and asserts the
grouped interval comes out wider. For the pooled metric, `groups` is tiled across layers,
so a resampled sequence carries all its layers together — which is the right thing to do,
since a sequence's layers are the correlated unit.

Two real problems:

**(a) `n_boot = 200` is small.** The 2.5% and 97.5% percentiles rest on roughly the 5th
and 195th order statistics, so the interval endpoints are themselves noisy. It is the
number a reviewer will circle. Mitigation: the gaps being claimed are nowhere near the
decision boundary — OLMoE INT4 routing KL is uniform 0.028579 [0.028096, 0.029102] versus
mixed 0.009005 [0.008730, 0.009304], a 3.2× separation, not a marginal one. Every
"MIXED WINS" line in both summaries has the same character. So the conclusions do not
depend on `n_boot`, but the paper should either raise it or say why 200 suffices.

**(b) Comparing two independently-constructed intervals is the wrong test in principle —
but it is the *conservative* wrong test, and the runs are already paired.** I checked:
both `uniform` and `mixed` call `bootstrap_ci` with `seed=cfg.seed=42`, on the same 128
groups in the same order, producing the same
`default_rng(42).integers(0, 128, size=(200, 128))` draw matrix. The bootstrap replicates
are therefore *literally the same resamples of the same sequences*. The disjointness test
just discards that pairing. A paired-difference bootstrap would give a strictly narrower
interval on `mixed − uniform`, so every "intervals disjoint" claim in `summary.md` and
`comparison.md` would survive a proper paired test — it is not at risk of being
overturned, only of being under-powered.

*Honest response:* describe it as "non-overlapping 95% percentile bootstrap intervals
over sequences, a conservative stand-in for a paired-difference test" and report effect
sizes with intervals rather than only the disjointness verdict.

*Fix, no GPU:* the exact Part 1 paired difference cannot be recomputed offline, because
per-token metric values are not saved and candidate logits exist only at 1-in-8 stride.
But it *can* be computed at 1-in-8 stride from `router_inputs.pt` with
`n_boot = 10000` and reported as a supplementary confirmation. Separately,
`scripts/attribute.py` already takes `--n-boot`, so re-running the Part 2 attribution at
`--n-boot 10000` costs nothing.

### 4.2 Perplexity carries no uncertainty at all

`decision_gate` deliberately excludes perplexity from the verdict, with a comment saying
so — that is the right call. But perplexity is the *first* column of every results table
and the first figure, and it has no interval anywhere. Per-window NLL is not saved
(`evaluate_lm` accumulates a scalar `total_nll`), so no interval is recoverable offline.

*Honest response:* "Perplexity is a point estimate over all 288,448 (OLMoE) / 298,785
(Qwen) scored tokens of the WikiText-2 test set; the decision gate rests on the routing
metrics, which carry sequence-level bootstrap intervals." That is defensible. A proper fix
needs a code change plus a re-run and is not worth it.

### 4.3 Seeds and run-to-run variance

Single seed throughout: `seed: 42` in both configs, `placebo_seed: 0`. Three distinct
questions hide behind "is one seed enough", and they have different answers:

- **Numerical determinism:** established. The gold self-comparison gives KL exactly 0 and
  top-1 error exactly 0 on both models (`self_check.passed`), and quantization is
  deterministic rounding.
- **Sensitivity to the routing-data draw:** this is exactly what the sequence-level
  bootstrap over 128 sequences answers. No extra seeds needed; say so.
- **Sensitivity to the placebo draw: not addressed at all, and this is the weak point.**
  The placebo is *one module* drawn from a pool of 3072 (OLMoE) / 4344 (Qwen). With
  `placebo_seed=0` the control is n=1. Because it happens to be near a null-op by
  construction (§1) the answer is unlikely to change, but "you drew one module once" is a
  fair hit and it is cheap to fix.

There is also a plumbing wrinkle: `placebo_seed` is a YAML-only field with no
`scripts/run.py` flag, and the run directory is `placebo_int4` regardless of seed, so
additional seeds would overwrite each other unless you vary `--results-dir` or add a
config per seed.

### 4.4 `mixed` vs `uniform` differ only in the routers — verified, and provable from the artifacts

The paper's central single-variable claim. I checked it three ways and it holds.

*In the code:* `build_policy` gives `uniform` skip list `ALWAYS_SKIP` and `mixed` the same
plus `spec.protect_patterns`, which is `(r".*\.mlp\.gate",)` for both models — routers
only, no `shared_expert_gate`. `resolve_skips` expands with `re.fullmatch`, which cannot
reach `model.layers.N.mlp.shared_expert_gate` or `model.layers.N.mlp.experts.J.gate_proj`
(both fail the required `.mlp.gate` terminal). `tests/test_quantize.py::
test_uniform_and_mixed_differ_only_by_routers` asserts the symmetric difference of the two
skip sets contains only routers.

*In the audit counts:* 3152 − 3136 = 16 = the OLMoE router count; 4536 − 4512 = 24 = the
Qwen router count. `mixed` reports 0/16 and 0/24 routers quantized, `uniform` 16/16 and
24/24, and `protected_but_quantized` is empty everywhere.

*Empirically, from the saved artifacts — this is the strongest form and it belongs in the
appendix.* Comparing the `router_weights` stored in each run's `router_inputs.pt` against
the gold `router_weights` in `artifacts.pt`:

| Model | Run | Max distinct values per output channel | Allowed (2^bits) | Layers bit-identical to gold |
|---|---|---|---|---|
| OLMoE | `uniform_int8` | 60 | 256 | 0/16 |
| OLMoE | `uniform_int4` | **16** | 16 | 0/16 |
| OLMoE | `uniform_int3` | **8** | 8 | 0/16 |
| OLMoE | `mixed_int8/4/3` | 1161 | — | **16/16** |
| OLMoE | `placebo_int4` | **16** | 16 | 0/16 |
| OLMoE | `placebo_int3` | **8** | 8 | 0/16 |
| Qwen | `uniform_int4` | **16** | 16 | 0/24 |
| Qwen | `uniform_int3` | **8** | 8 | 0/24 |
| Qwen | `mixed_int8/4/3` | 1107 | — | **24/24** |
| Qwen | `placebo_int4/int3` | 16 / 8 | 16 / 8 | 0/24 |

`mixed`'s router weights are **bit-identical to gold in every layer of both models**,
while `uniform` and `placebo` hit exactly the INT4 and INT3 level ceilings. That is direct
evidence for the single-variable claim, computed with no GPU. (The 60 and 46 at INT8 are
expected: with a per-channel scale of `max|w|/127` and roughly Gaussian router weights,
the bulk of 2048 values per row lands within ~±32 levels.)

*One residual caveat, worth a sentence but not a concern:* `device_map="auto"` placement
differs across runs. `results/olmoe/uniform_int3/metrics.json` and `uniform_int4` record
`device_map = {"": 0}` — the whole quantized model on GPU 0 — whereas the placebo runs
record a per-module map. Nothing is offloaded to CPU or disk in any run
(`offloaded_modules` is empty everywhere), and placement does not change the arithmetic
of these dense ops. But do not write "the quantized OLMoE runs shard across three cards";
the memory report says otherwise for at least two of them.

### 4.5 The bit-width check never inspects a router or an expert on the `uniform`/`mixed` runs

This one is uncomfortable, because it is exactly the class of silent failure `verify.py`
exists to prevent.

`verify.check_bit_width` picks its sample as `candidates[::step][:4]` with
`step = len(candidates) // 4`. The stride interacts pathologically with the module layout:

- OLMoE has exactly **197** quantizable modules per layer (4 attention + 1 router + 192
  expert projections), and `3152 // 4 = 788 = 4 × 197`. So the stride lands on
  `self_attn.q_proj` in layers 0, 4, 8, 12 — every single time.
- Qwen has **189** per layer, and `4536 // 4 = 1134 = 6 × 189`. Same outcome:
  `self_attn.q_proj` in layers 0, 6, 12, 18.

You can see it in the artifacts: every `uniform` and `mixed` run's
`bit_width_check.samples` lists four `self_attn.q_proj` modules and nothing else. The
`placebo` runs, whose candidate list is one element shorter, escape the resonance and do
sample experts and (on Qwen) the shared-expert gate — which is why their samples look
different.

So the "bit-width check passed" line in both `summary.md` files is, for `uniform` and
`mixed`, a statement about four attention projections. It says nothing about the experts
(the thing being quantized) or the routers (the thing the experiment is about). The
`weight_types` audit does independently confirm that 3151/3152/4536 modules carry the
quantized tensor subclass, so nothing was silently skipped — but the *precision* claim was
never checked where it matters.

*Fix, no GPU:* the routers are fully verifiable offline from the saved `router_weights`,
and §4.4 above does exactly that — exactly 16 levels at INT4 and 8 at INT3 on both
models. Put that table in the appendix instead of, or alongside, the current check. The
code should also pick its sample randomly (or one per role) rather than by stride; that is
a one-line change but it needs a re-run to show up in existing results.

### 4.6 The attribution's use of `mixed` activations for cell C, and the interaction residual

**Cell C is sound.** Using the `mixed` run's activations to isolate upstream drift is the
right choice, and `attribution.py`'s module docstring argues it correctly: `mixed` and
`uniform` quantize the same attention and expert weights, but only `mixed` leaves the
routers alone, so its hidden states carry activation drift *without* feedback from
perturbed routing decisions. Cell C then genuinely equals the `mixed` condition, and the
cross-check confirms it: OLMoE INT4 C = 16.62% vs mixed measured 16.86%; Qwen INT4
C = 18.11% vs 17.90%.

**Cell D is mislabelled.** D pairs quantized router weights with `mixed`-run activations,
so it is *not* the `uniform` model — it omits the feedback of changed routing on later
layers' hidden states. The gap is visible and grows with damage: OLMoE INT3 D = 50.01% vs
uniform 52.52%; Qwen INT3 D = 68.58% vs uniform 72.48%. `scripts/attribute.py`'s
`cross_check` docstring gets this right, but the user-facing labels do not: the `LABELS`
dict says `"both (= uniform)"`, the markdown table header says `Both (uniform)`, and
`attribution.py`'s module docstring says `D = ... (= uniform)`. Fix the labels to
"both (uniform without routing feedback)" and report the gap as a measurement of the
feedback effect — it is a small bonus result, not an embarrassment.

**The interaction residual is handled honestly but described in a way the numbers do not
support.** `shares["interaction"]` is computed as `1 − weights_of_both − activations_of_both`
and is **reported with no confidence interval** while both of its inputs have one. It is
also large and negative everywhere: −36.4% / −40.2% / −51.0% (OLMoE) and −39.1% / −42.5%
/ −47.0% (Qwen). The `attribution.md` "Reading these numbers" section explains this
correctly — overlapping flip sets, counted once in `both` and twice across the two
single-mechanism cells — and correctly identifies `weights_of_mechanisms` as the
interpretable column.

The problem is upstream, in `OVERVIEW.md`, which promises the headline sentence:

> Comparing **B against C** gives the headline sentence: *"X% of top-k routing flips are
> attributable to router weights, and (100−X)% to upstream activation drift."*

**That sentence is not supported and should be dropped.** With a residual of −40% to
−51%, the two "shares of both" are not shares of anything additive, and a
"X% / (100−X)%" split is exactly the claim the non-additivity forbids. Keep the framing
that `attribution.md` already uses: "of the routing flips caused by exactly one mechanism,
X% come from the router's own weights" — 62.2% (OLMoE INT4), 58.0% (Qwen INT4). And give
the interaction term a bootstrap interval; it is one extra `_grouped_ratio_ci` call in
`attribution.py` and needs no GPU.

*One more thing to state:* `check_reconstruction` gates cell A at 95% top-1 agreement and
achieves 99.66% (OLMoE) / 99.37% (Qwen), worst layer 99.46% / 98.49%. Genuinely
reassuring. But it means all four cells carry a ~0.5–1.5% floor of reconstruction noise,
from fp16-stored activations recomputed in CPU fp32 against a bf16 GPU forward pass.
Every difference the attribution reports is far above that floor (the smallest is Qwen
INT8: weights-only 8.81% vs activations-only 13.04%), so it does not bite — but state the
floor rather than leaving a reader to wonder.

### 4.7 Qwen INT3 is genuine saturation, not a bug

Confirmed. Five independent lines of evidence:

1. **`mean_nll` is finite and sane:** 9.1649 (uniform INT3) and 7.1200 (mixed INT3) over
   298,785 scored tokens. `exp(9.1649) = 9555` and `exp(7.1200) = 1236` reproduce the
   reported perplexities exactly. No inf, no NaN, nothing that looks like an overflow.
2. **The same code gives sane numbers next door:** Qwen INT4 = 12.01, OLMoE INT3 = 30.23,
   from the same pipeline, same corpus, same window count.
3. **Four unrelated metrics agree the model is destroyed:** top-1 output agreement with
   gold falls to 6.60% (uniform) / 12.81% (mixed), top-M output KL rises to 3.69 / 2.90,
   pooled routing top-1 flip rate hits 72.48% / 55.69%, and per-layer dead-expert counts
   reach 18 of 60. A numerical blow-up in the perplexity path would not move the routing
   and output-distribution metrics coherently.
4. **`placebo` INT3 reproduces `uniform` INT3 to four significant figures** (9563.04 vs
   9555.45; `mean_nll` 9.1657 vs 9.1649). Two near-identical models producing
   near-identical catastrophic numbers is what genuine saturation looks like; a random
   numerical fault would not replicate that closely.
5. **INT3 really was applied:** `bit_width_check` reports 8 distinct levels per channel,
   and the offline router check in §4.4 independently confirms exactly 8 levels.

The mechanism is unremarkable: calibration-free per-output-channel symmetric INT3 on a
14.3B model whose experts are only 1408 wide, plus a 5632-wide shared expert that is also
quantized. There is nothing to fix.

*But do not build a claim on the ratio.* At that damage level perplexity is not a
meaningful scale, and "router protection improves perplexity 7.7×" (9555 → 1236) is not a
sentence to put in a paper — both models are unusable. State that INT3 saturates Qwen,
report the routing metrics and the collapse metrics (which stay in range and behave
monotonically), and make the perplexity-based claims at INT8 and INT4.

### 4.8 `verify.py`: checked versus silently assumed

**Actually checked** (and this is a genuinely strong gate set):

- per-module quantized/plain tensor kind, parameter counts, and a `weight_types` census
- per-router quantized state, by name, recorded in full in the results JSON
- `mixed` → zero routers quantized; `uniform`/`placebo`/`attention` → all routers
  quantized (with a comment explaining that a placebo sparing a router would be a second
  `mixed` run in disguise)
- `gold` → nothing quantized; any quantized policy → something quantized
- `protected_but_quantized` — anything the policy meant to protect but did not
- distinct-value ceiling `2**bits` on 4 sampled modules (but see §4.5)
- config expert count == router weight output dimension (`resolve_topology`)
- gold self-comparison exactly zero KL and zero top-1 error
- SHA-256 token fingerprint match between candidate and gold, plus topology match
- full consumption of every gold output-reference position (`evaluate.py`)

**Assumed and never checked:**

- **That the experts were quantized to the requested width.** The stride resonance in
  §4.5 means no expert projection is ever inspected on a `uniform` or `mixed` run. The
  `weight_types` census proves they carry a quantized subclass; nothing proves the width.
- **Which modules the placebo protected.** `num_protected_modules: 3` is recorded; the
  FQNs are not. `policy.skip_exact` is never written to `metrics.json`. Deterministic
  given `placebo_seed`, so recoverable, but the results files do not document the control.
  One-line fix for future runs; for the current ones, put the derived names in the
  appendix.
- **`effective_levels` inspects at most 8 rows** (`max_rows=8`) of each sampled module.
  Fine as a smoke test, worth stating.
- `ALWAYS_SKIP` (embeddings, `lm_head`) is *not* an unchecked assumption — it lives in
  `skip_patterns`, so `protected_but_quantized` would catch a leak.

### 4.9 Narrative inconsistencies a careful reader will notice

- `OVERVIEW.md` frames Part 2 as conditional: "Part 2 — why it failed (**only if Part 1
  is null**)", and the decision gate's own verdict string tells you to run the placebo and
  stop there ("Pass that control and Part 1 is the whole paper"). Part 1 came out strongly
  positive and Part 2 was run anyway. Running it was the right call — the attribution is
  one of the more interesting results — but the repo docs now describe a decision tree
  that was not followed. Re-frame Part 2 as a mechanism analysis rather than a
  contingency.
- `OVERVIEW.md` flags its own citation problems ("VSRAQ (arXiv:2606.05688)", "Anonymous
  (2026) is Fang & Huang", "Both must be fixed before submission"). Still unfixed as of
  this audit. Not my scope to verify the references, but it is on the critical path for
  the literature-review portion of the grade.
- `placebo` is **missing at INT8** on both models, so the control is absent at exactly the
  bit-width where the router effect is smallest and a placebo could most plausibly have
  mattered. See §5.4 — and note the two currently queued jobs may not produce it.

---

## 5. Recommended additional runs

Conventions followed from `scripts/slurm/`: partition `studentkillable`,
`--gres=gpu:geforce_rtx_2080:N` (the TITAN Xp cards are sm_61 and the torch wheel has no
kernels for them), `--cpus-per-task=8`, `--requeue --open-mode=append` for the
preemptible partition, `MOEQUANT_MIN_VRAM_GB` = 20 for OLMoE and 40 for Qwen,
`_preflight.sh` sourced first, `--skip-existing` so a requeue resumes.

Runtime estimates are from `sacct` on the completed jobs:

| Job | Runs | Elapsed | Per run |
|-----|------|---------|---------|
| 758411 OLMoE full sweep | 7 | 1h05m | ~9.4 min |
| 758518 OLMoE placebo | 2 | 30m | ~15 min |
| 758904 Qwen full sweep | 7 | 2h10m | ~18.5 min |
| 768396 Qwen placebo | 2 | 52m | ~26 min |

Per-phase, from the tqdm bars in `logs/`: OLMoE routing ~2.5 min + eval ~6 min; Qwen
routing ~4.3 min + eval ~11.8 min. The rest is model load and quantization.

### Do these first — NO GPU REQUIRED

All five are offline analyses on artifacts already on disk. They are the highest
value-per-minute work available and none of them needs the queue.

1. **Layer-wise expert-collapse table** from `routing.per_layer.*.cand_usage` /
   `.gold_usage` in every `metrics.json`. Zero compute — the numbers are already
   computed. Delivers the proposal's expert-collapse commitment (§3, Tier 1).
2. **Gini / max share / top-10% load share / KL(load ‖ uniform) / per-expert histograms**,
   per layer per run, reconstructed from `router_inputs.pt` + gold `artifacts.pt`. ~1 min
   per model per bit-width (§3, Tier 2). Verified working.
3. **Offline bit-width verification of the routers**, and the proof that `mixed`'s router
   weights are bit-identical to gold in every layer (§4.4, §4.5). Appendix material that
   is stronger than the check currently in the pipeline. ~25 s per model.
4. **Re-run the Part 2 attribution with `--n-boot 10000`** (the flag already exists) and
   add a bootstrap interval to the interaction residual. Also fix the "= uniform" labels
   on cell D (§4.6).
5. **Paired-difference bootstrap** of `mixed` − `uniform` on routing KL and top-1 flips,
   at 1-in-8 stride from `router_inputs.pt`, `n_boot = 10000`, as a supplementary
   confirmation of the disjointness claims (§4.1).

### Worth queueing

**1. `attention` control — OLMoE and Qwen, INT4 + INT3.** *The most important missing
run.* It is the only available control whose protected weights have the same activation
exposure as the routers (every token, every layer), so it is the only run that tests
"is it the routers, or is it any high-precision island on the main path" — the confound
the parameter-matched placebo structurally cannot address (§1). Already implemented in
`quantize.py`, already asserted by `verify.py`, never run. Note it protects far more than
the router budget (3.88% of OLMoE, 2.81% of Qwen), which makes it a *harder* test: if
protecting 130× the router budget on the main path does not reduce routing drift the way
protecting the routers does, the specificity claim is airtight.

No existing job script runs an arbitrary policy, so either add a `run_policy.sh`
generalization of `run_placebo.sh` (five lines) or use `--wrap`, which works today:

```bash
# OLMoE: ~30 min for 2 runs
sbatch --job-name=moe-attention --partition=studentkillable \
       --gres=gpu:geforce_rtx_2080:3 --cpus-per-task=8 --mem=64G \
       --time=4:00:00 --requeue --open-mode=append \
       --output=logs/attention_%j.out --error=logs/attention_%j.err \
       --wrap 'set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; \
               export MOEQUANT_MIN_VRAM_GB=20; \
               source scripts/slurm/_preflight.sh; \
               "$PY_BIN" scripts/run.py --config configs/olmoe.yaml \
                   --policies attention --bits 4 3 --keep-going --skip-existing; \
               "$PY_BIN" scripts/analyze.py --results-dir results/olmoe'

# Qwen: ~55 min for 2 runs
sbatch --job-name=moe-attention --partition=studentkillable \
       --gres=gpu:geforce_rtx_2080:5 --cpus-per-task=8 --mem=96G \
       --time=6:00:00 --requeue --open-mode=append \
       --output=logs/attention_%j.out --error=logs/attention_%j.err \
       --wrap 'set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; \
               export MOEQUANT_MIN_VRAM_GB=40; \
               source scripts/slurm/_preflight.sh; \
               "$PY_BIN" scripts/run.py --config configs/qwen.yaml \
                   --policies attention --bits 4 3 --keep-going --skip-existing; \
               "$PY_BIN" scripts/analyze.py --results-dir results/qwen'
```

*Buys:* closes the activation-exposure confound that the placebo leaves wide open, which
is the single biggest threat to the paper's central claim.

**2. Layer-distributed placebo — OLMoE and Qwen, INT4 + INT3.** One randomly chosen
expert projection *per layer* (0.48% of params on both models). Needs ~10 lines of new
code first (a `per_layer=True` mode in `sample_placebo_modules`), so it is gated on that.
Placement-matched but deliberately **not** parameter-matched — as established in §1, at
`nn.Linear` granularity you cannot have both in either architecture, and the paper should
say so explicitly. Same allocations and roughly the same runtime as item 1 (~30 min OLMoE,
~55 min Qwen). Command form identical to item 1 with `--policies placebo` and the new
mode enabled.

*Buys:* the placement-matched control, which together with item 1 lets the specificity
sentence stand unhedged.

**3. Placebo seeds 1, 2, 3 — OLMoE and Qwen, INT4 + INT3.** Turns an n=1 control into
n=4 and answers "you drew one module once" (§4.3). Needs plumbing: `placebo_seed` is
YAML-only and the run directory name ignores it, so use one config per seed plus a
distinct `--results-dir`, or add a `--placebo-seed` flag that also namespaces the run
directory. 6 runs per model: ~1.5 h OLMoE, ~2.6 h Qwen.

*Buys:* removes a cheap statistical objection. Ranked below items 1–2 because more draws
of a control that is near a null-op by construction cannot address the exposure confound —
it makes the weak control better characterized, not stronger.

**4. Missing placebo at INT8 — both models.** Completes the results table at the
bit-width where the effect is smallest. Uses the existing script exactly as documented:

```bash
sbatch scripts/slurm/run_placebo.sh olmoe 8                                    # ~15 min
sbatch --gres=gpu:geforce_rtx_2080:5 --mem=96G scripts/slurm/run_placebo.sh qwen 8   # ~26 min
```

**Check before submitting anything.** Jobs **773490** (3 GPUs, 64 G) and **773491**
(5 GPUs, 96 G) are pending and *must be left alone*. As recorded by
`scontrol show job`, **neither carries a model or bit-width argument** —
`Command=.../run_placebo.sh` with nothing after it. `run_placebo.sh` then defaults to
`MODEL=olmoe` and `BITS=(4 3)`, both of which already exist on disk, and `run.py` is
invoked with `--skip-existing`. If that is what Slurm really recorded, both jobs will skip
every run and only re-run `analyze.py` — and the 5-GPU allocation on 773491 strongly
suggests Qwen was intended. So the INT8 placebo cell is probably **not** coming from the
queue. Verify the arguments before either duplicating this work or assuming it is done.

*Buys:* a complete placebo row; small.

*Resolved, and this reasoning was wrong.* Slurm's `Command=` field simply does not echo
positional arguments; a completed earlier job showed it running with the `qwen` arguments
it had been given. The arguments do survive submission, jobs 773490 and 773491 ran as
intended, and the INT8 placebo now exists on both models. The placebo row is complete at
all three bit-widths.

**5. C4 robustness point — OLMoE, INT4 only.** `gold` + `uniform` + `mixed` = 3 runs,
~35 min. Needs `configs/olmoe_c4.yaml` (a copy of `configs/olmoe.yaml` with
`corpus: c4` — config only, no code, since `data.py` already supports C4) and a separate
results directory, because a C4 run needs its own gold and its own token fingerprint.
First run pays a ~300 MB C4 shard download, which is not currently cached.

```bash
sbatch --job-name=moe-c4 --partition=studentkillable \
       --gres=gpu:geforce_rtx_2080:3 --cpus-per-task=8 --mem=64G \
       --time=4:00:00 --requeue --open-mode=append \
       --output=logs/c4_%j.out --error=logs/c4_%j.err \
       --wrap 'set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; \
               export MOEQUANT_MIN_VRAM_GB=20; \
               source scripts/slurm/_preflight.sh; \
               "$PY_BIN" scripts/run.py --config configs/olmoe_c4.yaml \
                   --policies gold uniform mixed --bits 4 \
                   --results-dir results_c4 --keep-going --skip-existing; \
               "$PY_BIN" scripts/analyze.py --results-dir results_c4/olmoe'
```

*Buys:* converts the "you promised C4" deviation from a paragraph of prose into one data
point, for 35 minutes of queue time. Cheapest reputational win available.

### Not worth it

- **Full C4 sweep, both models, all three bit-widths.** 14 runs, ~3.5 GPU-hours, plus a
  fresh gold per model and a download on a preemptible partition. The marginal claim over
  the single OLMoE INT4 C4 point in item 5 is small; the risk of a half-finished second
  corpus in the paper is not.
- **Extending the sweep to INT2.** Qwen is already destroyed at INT3 (PPL 9555, 6.6%
  top-1 output agreement). INT2 adds a row of noise, not a trend, and invites the
  reviewer to ask why you reported a model that outputs nothing.
- **Re-running Part 1 purely to raise `n_boot` from 200 to 10,000.** The intervals are
  nowhere near the decision boundary — OLMoE INT4 routing KL separates 3.2× with
  intervals ~2% wide. Do the offline 1-in-8-stride paired bootstrap instead (no-GPU item
  5) and note in the paper why 200 sufficed.
- **A bitsandbytes cross-check** (the `bnb` optional dependency exists). A second backend
  would guard against a torchao-specific artifact, but the offline verification that
  `uniform`/`placebo` router weights carry exactly `2**bits` levels per channel, and that
  `mixed`'s are bit-identical to gold (§4.4), already rules out the plausible artifacts.
  Large code change, small marginal assurance.
- **DeepSeek as a third architecture.** `registry.py` marks it "STRETCH ONLY": needs
  `trust_remote_code`, layer 0 is dense, and `MoEGate` is not an `nn.Linear` so it needs
  the `recompute` path. High integration risk for low marginal value now that the effect
  already replicates on two architectures.
- **Memory or throughput measurement.** An explicit non-goal in `ARCHITECTURE.md`, and
  correctly so: these runs dequantize for compute on the 2080 Ti, so any savings number
  would be misleading.
- **The proposal's "dynamically scaling routing scores by activation size" contingency,
  or the margin-aware top-*k* widening replacement.** Both are mitigations for a null
  Part 1. Part 1 is strongly positive, so neither is needed and building one now would be
  scope creep three days before a paper.

---

## Bottom line

Three things need attention before the paper is written, in this order:

1. **The placebo protects one expert down-projection in one layer, read by ~11% (OLMoE)
   and ~5% (Qwen) of tokens.** "Parameter-matched" is literally true and matches nothing
   else. The control as it stands cannot support the router-specificity claim on its own;
   the `attention` policy — already coded, never run, ~85 min of GPU across both models —
   fixes that.
2. **The expert-collapse promise is already satisfied by data on disk and nobody has
   looked.** Qwen INT3 starves 15.4% of expert slots under `uniform` and 3.8% under
   `mixed`, with zero fully-unused experts under `mixed` versus five layers' worth under
   `uniform`. Zero GPU. Do this first.
3. **The disjointness test and the "X% / (100−X)%" attribution sentence both need
   rewording**, and the bit-width gate never actually inspected a router or an expert on
   the runs that matter — though the routers can be, and now have been, verified offline.

Nothing found here overturns the central result. `mixed` really does beat `uniform`, the
routers really are the only difference between those two models (bit-identical to gold in
every layer, verified from the artifacts), and the effect replicates across two
architectures at three bit-widths. What is soft is the *framing* around it: one weak
control carrying a strong claim, a promised analysis left on the floor, and a couple of
sentences that claim more decomposition than the numbers support.

---

## Submission log

### 2026-08-22 10:46 — `attention` control queued (§5, item 1)

| Job | Model | Policy / bits | Allocation | State at submission |
|-----|-------|---------------|------------|---------------------|
| **773602** | OLMoE | `attention`, INT4 + INT3 | 3 × geforce_rtx_2080, 64 G, 8 h | PENDING (Priority) |
| **773603** | Qwen | `attention`, INT4 + INT3 | 5 × geforce_rtx_2080, 96 G, 8 h | PENDING (Priority) |

Both via a new `scripts/slurm/run_control.sh`, which is `run_placebo.sh` generalized to an
arbitrary policy name (`run_control.sh POLICY [MODEL] [BITS...]`):

```bash
sbatch --job-name=moe-attention --output=logs/attention_%j.out \
       --error=logs/attention_%j.err \
       scripts/slurm/run_control.sh attention olmoe 4 3
sbatch --job-name=moe-attention --gres=gpu:geforce_rtx_2080:5 --mem=96G \
       --output=logs/attention_%j.out --error=logs/attention_%j.err \
       scripts/slurm/run_control.sh attention qwen 4 3
```

A script rather than the `--wrap` form suggested in §5: `sbatch --wrap` runs the command
under `#!/bin/sh`, which on these nodes is dash, so the wrapped `set -euo pipefail` aborts
immediately (`set: Illegal option -o pipefail`) and `_preflight.sh`'s `[[` would fail too.
The §5 `--wrap` snippets should not be used as written. Logs land in
`logs/attention_7736{02,03}.{out,err}`.

Verified before submitting: `attention` is in `POLICIES`; its skip set is `ALWAYS_SKIP`
plus `.*\.self_attn(\..*)?`, which on the real checkpoints resolves to
`model.layers.N.self_attn.{q,k,v,o}_proj` for every layer — 64 modules / 268,435,456
params / 3.88% of OLMoE and 96 modules / 402,800,640 params / 2.81% of Qwen, matching §1.
All 16 / 24 routers stay quantized, so `verify.py`'s `attention` assertion is satisfied and
the run is distinguishable from `mixed`. Both gold `artifacts.pt` files are present.
Expected ~30 min (OLMoE) and ~55 min (Qwen) of compute once scheduled.

Jobs 773490 and 773491 (INT8 placebo) were left untouched.
