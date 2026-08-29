# qwen - expert collapse

Per-layer expert-load balance for every run, from data already on disk. The pooled entropy in `summary.md` concatenates all layers into one histogram before measuring, which cancels independent per-layer imbalances; everything below is resolved per layer.

## Did collapse occur on qwen?

Baseline: the unquantized model already leaves 0 of 1440 expert slots below a tenth of fair share (0.0%), with 0 receiving nothing. Natural routers are not balanced, so a candidate is only collapsing if it is worse than this.

- **INT4: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 0/1440 (0.0%) under `uniform` vs 0/1440 (0.0%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9790 vs 0.9854; worst-layer max/mean load 3.50 vs 2.69.
  Load-distribution KL against uniform (strided) 0.0349 under `uniform` vs 0.0203 under `mixed`, a 1.72x reduction; Gini 0.141 vs 0.107.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.

## Per-layer expert collapse (full resolution, 32,768 tokens per layer)

Dead means an expert receiving less than a tenth of its fair share of tokens in that layer; unused means exactly zero selections. Counts are summed over layers and expressed against the total expert *slots* (layers x experts), since expert *j* in one layer is a different expert from expert *j* in another.

| Run | Dead slots | Unused slots | Layers with an unused expert | Worst-layer norm. entropy | Worst-layer max/mean load |
|-----|------------|--------------|------------------------------|--------------------------|---------------------------|
| gold BF16 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9982 (layer 2) | 1.59 (layer 2) |
| uniform INT4 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9790 (layer 2) | 3.50 (layer 2) |
| mixed INT4 | 0 / 1440 (0.0%) | 0 / 1440 (0.00%) | 0 of 24 | 0.9854 (layer 2) | 2.69 (layer 2) |

## Why the pooled number missed this

Both columns are the same normalized usage entropy on the same run. The left one is what `summary.md` quotes; it concatenates every layer's tokens into a single histogram before measuring, so imbalances pointing in different directions across layers cancel.

| Run | Pooled (reported) | Worst layer | Mean over layers |
|-----|-------------------|-------------|------------------|
| gold BF16 | 0.9999 | 0.9982 | 0.9988 |
| uniform INT4 | 0.9997 | 0.9790 | 0.9924 |
| mixed INT4 | 0.9998 | 0.9854 | 0.9958 |

## Load-distribution shape (strided capture, 1 token in 8)

These are the sensitive instruments. Normalized entropy is maximal at balance, so its gradient there is zero; Gini and KL against uniform are not. `max share` is the busiest expert's load as a multiple of fair share, `top-10%` the fraction of routing slots the busiest tenth of experts absorbs (0.100 under perfect balance).

Gold is measured on the same strided positions, so every row is comparable. Dead and unused counts are **not** reported here - at 1 token in 8 the zero counts are inflated, which is exactly why the table above uses the full-resolution fields.

| Run | Gini (mean / max) | Max share | Top-10% share | KL(load &#124;&#124; uniform) mean |
|-----|-------------------|-----------|---------------|----------------------------|
| gold BF16 | 0.070 / 0.088 | 1.92x | 0.127 | 0.0085 |
| uniform INT4 | 0.141 / 0.214 | 3.45x | 0.152 | 0.0349 |
| mixed INT4 | 0.107 / 0.180 | 2.57x | 0.140 | 0.0203 |

## Why dead counts and ratio metrics come from different resolutions

The candidate router inputs were captured every 8th token, so the reconstruction sees one token in 8. Ratio statistics are unaffected, but a smaller sample makes low counts hit zero for reasons that have nothing to do with quantization. Measured on the same runs:

| Run | Dead slots, full resolution | Dead slots, strided | Unused slots, full | Unused slots, strided |
|-----|-----------------------------|---------------------|--------------------|-----------------------|
| gold BF16 | 0 | 0 | 0 | 0 |
| uniform INT4 | 0 | 0 | 0 | 0 |
| mixed INT4 | 0 | 0 | 0 | 0 |

The inflation is systematic and affects gold as much as the candidates, so the full-resolution fields are the ones quoted for dead and unused experts and the reconstruction is used only for the shape statistics.
