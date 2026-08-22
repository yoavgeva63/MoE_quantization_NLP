# qwen - expert collapse

Per-layer expert-load balance for every run, from data already on disk. The pooled entropy in `summary.md` concatenates all layers into one histogram before measuring, which cancels independent per-layer imbalances; everything below is resolved per layer.

## Did collapse occur on qwen?

Baseline: the unquantized model already leaves 0 of 1440 expert slots below a tenth of fair share (0.0%), with 0 receiving nothing. Natural routers are not balanced, so a candidate is only collapsing if it is worse than this.

- **INT8: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 0/1440 (0.0%) under `uniform` vs 0/1440 (0.0%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9685 vs 0.9703; worst-layer max/mean load 3.82 vs 3.86.
  Load-distribution KL against uniform (strided) 0.0558 under `uniform` vs 0.0547 under `mixed`, a 1.02x reduction; Gini 0.182 vs 0.181.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.
- **INT4: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 1/1440 (0.1%) under `uniform` vs 0/1440 (0.0%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9703 vs 0.9694; worst-layer max/mean load 3.20 vs 3.36.
  Load-distribution KL against uniform (strided) 0.0752 under `uniform` vs 0.0644 under `mixed`, a 1.17x reduction; Gini 0.213 vs 0.196.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.
- **INT3: **collapse** under `uniform`.** 
  Dead slots 222/1440 (15.4%) under `uniform` vs 55/1440 (3.8%) under `mixed`; layers containing a fully unused expert 5 vs 0; worst-layer entropy 0.7585 vs 0.8752; worst-layer max/mean load 12.53 vs 9.94.
  Load-distribution KL against uniform (strided) 0.5717 under `uniform` vs 0.2693 under `mixed`, a 2.12x reduction; Gini 0.551 vs 0.382.
  Router protection **mitigated** it.

## Per-layer expert collapse (full resolution, 32,768 tokens per layer)

Dead means an expert receiving less than a tenth of its fair share of tokens in that layer; unused means exactly zero selections. Counts are summed over layers and expressed against the total expert *slots* (layers x experts), since expert *j* in one layer is a different expert from expert *j* in another.

| Run | Dead slots | Unused slots | Layers with an unused expert | Worst-layer norm. entropy | Worst-layer max/mean load |
|-----|------------|--------------|------------------------------|--------------------------|---------------------------|
| gold BF16 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9742 (layer 23) | 3.83 (layer 23) |
| uniform INT8 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9685 (layer 23) | 3.82 (layer 23) |
| mixed INT8 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9703 (layer 23) | 3.86 (layer 23) |
| placebo INT8 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9685 (layer 23) | 3.82 (layer 23) |
| uniform INT4 | 1 / 1440 (0.1%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9703 (layer 23) | 3.20 (layer 23) |
| mixed INT4 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9694 (layer 23) | 3.36 (layer 23) |
| placebo INT4 | 1 / 1440 (0.1%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9703 (layer 23) | 3.20 (layer 23) |
| attention INT4 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9694 (layer 23) | 3.17 (layer 23) |
| uniform INT3 | 222 / 1440 (15.4%) | 7 / 1440 (0.49%) | 5 of 24 | 0.7585 (layer 8) | 12.53 (layer 23) |
| mixed INT3 | 55 / 1440 (3.8%) | 0 / 1440 (0.00%) | 0 of 24 | 0.8752 (layer 12) | 9.94 (layer 23) |
| placebo INT3 | 220 / 1440 (15.3%) | 7 / 1440 (0.49%) | 5 of 24 | 0.7585 (layer 8) | 12.53 (layer 23) |
| attention INT3 | 74 / 1440 (5.1%) | 0 / 1440 (0.00%) | 0 of 24 | 0.7850 (layer 23) | 11.77 (layer 23) |

## Why the pooled number missed this

Both columns are the same normalized usage entropy on the same run. The left one is what `summary.md` quotes; it concatenates every layer's tokens into a single histogram before measuring, so imbalances pointing in different directions across layers cancel.

| Run | Pooled (reported) | Worst layer | Mean over layers |
|-----|-------------------|-------------|------------------|
| gold BF16 | 0.9996 | 0.9742 | 0.9900 |
| uniform INT8 | 0.9996 | 0.9685 | 0.9879 |
| mixed INT8 | 0.9996 | 0.9703 | 0.9882 |
| placebo INT8 | 0.9996 | 0.9685 | 0.9879 |
| uniform INT4 | 0.9993 | 0.9703 | 0.9833 |
| mixed INT4 | 0.9995 | 0.9694 | 0.9859 |
| placebo INT4 | 0.9993 | 0.9703 | 0.9833 |
| attention INT4 | 0.9994 | 0.9694 | 0.9859 |
| uniform INT3 | 0.9929 | 0.7585 | 0.8575 |
| mixed INT3 | 0.9964 | 0.8752 | 0.9340 |
| placebo INT3 | 0.9929 | 0.7585 | 0.8575 |
| attention INT3 | 0.9944 | 0.7850 | 0.9158 |

## Load-distribution shape (strided capture, 1 token in 8)

These are the sensitive instruments. Normalized entropy is maximal at balance, so its gradient there is zero; Gini and KL against uniform are not. `max share` is the busiest expert's load as a multiple of fair share, `top-10%` the fraction of routing slots the busiest tenth of experts absorbs (0.100 under perfect balance).

Gold is measured on the same strided positions, so every row is comparable. Dead and unused counts are **not** reported here - at 1 token in 8 the zero counts are inflated, which is exactly why the table above uses the full-resolution fields.

| Run | Gini (mean / max) | Max share | Top-10% share | KL(load &#124;&#124; uniform) mean |
|-----|-------------------|-----------|---------------|----------------------------|
| gold BF16 | 0.167 / 0.240 | 3.69x | 0.162 | 0.0465 |
| uniform INT8 | 0.182 / 0.275 | 3.65x | 0.169 | 0.0558 |
| mixed INT8 | 0.181 / 0.265 | 3.71x | 0.167 | 0.0547 |
| placebo INT8 | 0.182 / 0.275 | 3.65x | 0.169 | 0.0558 |
| uniform INT4 | 0.213 / 0.269 | 3.12x | 0.178 | 0.0752 |
| mixed INT4 | 0.196 / 0.278 | 3.26x | 0.172 | 0.0644 |
| placebo INT4 | 0.213 / 0.269 | 3.11x | 0.178 | 0.0752 |
| attention INT4 | 0.198 / 0.279 | 3.08x | 0.172 | 0.0653 |
| uniform INT3 | 0.551 / 0.708 | 12.07x | 0.389 | 0.5717 |
| mixed INT3 | 0.382 / 0.526 | 9.66x | 0.276 | 0.2693 |
| placebo INT3 | 0.551 / 0.708 | 12.07x | 0.389 | 0.5717 |
| attention INT3 | 0.427 / 0.647 | 11.87x | 0.309 | 0.3461 |

## Why dead counts and ratio metrics come from different resolutions

The candidate router inputs were captured every 8th token, so the reconstruction sees one token in 8. Ratio statistics are unaffected, but a smaller sample makes low counts hit zero for reasons that have nothing to do with quantization. Measured on the same runs:

| Run | Dead slots, full resolution | Dead slots, strided | Unused slots, full | Unused slots, strided |
|-----|-----------------------------|---------------------|--------------------|-----------------------|
| gold BF16 | 0 | 0 | 0 | 0 |
| uniform INT8 | 0 | 0 | 0 | 0 |
| mixed INT8 | 0 | 0 | 0 | 0 |
| placebo INT8 | 0 | 0 | 0 | 0 |
| uniform INT4 | 1 | 1 | 0 | 0 |
| mixed INT4 | 0 | 0 | 0 | 0 |
| placebo INT4 | 1 | 1 | 0 | 0 |
| attention INT4 | 0 | 0 | 0 | 0 |
| uniform INT3 | 222 | 223 | 7 | 23 |
| mixed INT3 | 55 | 59 | 0 | 1 |
| placebo INT3 | 220 | 223 | 7 | 23 |
| attention INT3 | 74 | 84 | 0 | 2 |

The inflation is systematic and affects gold as much as the candidates, so the full-resolution fields are the ones quoted for dead and unused experts and the reconstruction is used only for the shape statistics.
