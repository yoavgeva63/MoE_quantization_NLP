# olmoe - expert collapse

Per-layer expert-load balance for every run, from data already on disk. The pooled entropy in `summary.md` concatenates all layers into one histogram before measuring, which cancels independent per-layer imbalances; everything below is resolved per layer.

## Did collapse occur on olmoe?

Baseline: the unquantized model already leaves 17 of 1024 expert slots below a tenth of fair share (1.7%), with 0 receiving nothing. Natural routers are not balanced, so a candidate is only collapsing if it is worse than this.

- **INT8: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 13/1024 (1.3%) under `uniform` vs 13/1024 (1.3%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9507 vs 0.9510; worst-layer max/mean load 3.67 vs 3.75.
  Load-distribution KL against uniform (strided) 0.1289 under `uniform` vs 0.1292 under `mixed`, a 1.00x reduction; Gini 0.273 vs 0.273.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.
- **INT4: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 15/1024 (1.5%) under `uniform` vs 15/1024 (1.5%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9483 vs 0.9508; worst-layer max/mean load 4.18 vs 4.13.
  Load-distribution KL against uniform (strided) 0.1291 under `uniform` vs 0.1302 under `mixed`, a 0.99x reduction; Gini 0.271 vs 0.273.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.
- **INT3: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 6/1024 (0.6%) under `uniform` vs 9/1024 (0.9%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9389 vs 0.9425; worst-layer max/mean load 5.86 vs 4.69.
  Load-distribution KL against uniform (strided) 0.1630 under `uniform` vs 0.1405 under `mixed`, a 1.16x reduction; Gini 0.307 vs 0.284.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.

## Per-layer expert collapse (full resolution, 32,768 tokens per layer)

Dead means an expert receiving less than a tenth of its fair share of tokens in that layer; unused means exactly zero selections. Counts are summed over layers and expressed against the total expert *slots* (layers x experts), since expert *j* in one layer is a different expert from expert *j* in another.

| Run | Dead slots | Unused slots | Layers with an unused expert | Worst-layer norm. entropy | Worst-layer max/mean load |
|-----|------------|--------------|------------------------------|--------------------------|---------------------------|
| gold BF16 | 17 / 1024 (1.7%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9533 (layer 15) | 4.01 (layer 0) |
| uniform INT8 | 13 / 1024 (1.3%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9507 (layer 15) | 3.67 (layer 12) |
| mixed INT8 | 13 / 1024 (1.3%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9510 (layer 15) | 3.75 (layer 12) |
| placebo INT8 | 13 / 1024 (1.3%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9507 (layer 15) | 3.67 (layer 12) |
| uniform INT4 | 15 / 1024 (1.5%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9483 (layer 15) | 4.18 (layer 0) |
| mixed INT4 | 15 / 1024 (1.5%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9508 (layer 15) | 4.13 (layer 0) |
| placebo INT4 | 15 / 1024 (1.5%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9483 (layer 15) | 4.18 (layer 0) |
| attention INT4 | 12 / 1024 (1.2%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9498 (layer 15) | 3.99 (layer 0) |
| uniform INT3 | 6 / 1024 (0.6%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9389 (layer 15) | 5.86 (layer 3) |
| mixed INT3 | 9 / 1024 (0.9%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9425 (layer 15) | 4.69 (layer 3) |
| placebo INT3 | 6 / 1024 (0.6%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9389 (layer 15) | 5.86 (layer 3) |
| attention INT3 | 8 / 1024 (0.8%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9465 (layer 15) | 5.09 (layer 3) |

## Why the pooled number missed this

Both columns are the same normalized usage entropy on the same run. The left one is what `summary.md` quotes; it concatenates every layer's tokens into a single histogram before measuring, so imbalances pointing in different directions across layers cancel.

| Run | Pooled (reported) | Worst layer | Mean over layers |
|-----|-------------------|-------------|------------------|
| gold BF16 | 0.9975 | 0.9533 | 0.9678 |
| uniform INT8 | 0.9977 | 0.9507 | 0.9682 |
| mixed INT8 | 0.9976 | 0.9510 | 0.9681 |
| placebo INT8 | 0.9977 | 0.9507 | 0.9682 |
| uniform INT4 | 0.9976 | 0.9483 | 0.9682 |
| mixed INT4 | 0.9976 | 0.9508 | 0.9676 |
| placebo INT4 | 0.9976 | 0.9483 | 0.9682 |
| attention INT4 | 0.9976 | 0.9498 | 0.9689 |
| uniform INT3 | 0.9968 | 0.9389 | 0.9603 |
| mixed INT3 | 0.9973 | 0.9425 | 0.9656 |
| placebo INT3 | 0.9968 | 0.9389 | 0.9603 |
| attention INT3 | 0.9974 | 0.9465 | 0.9653 |

## Load-distribution shape (strided capture, 1 token in 8)

These are the sensitive instruments. Normalized entropy is maximal at balance, so its gradient there is zero; Gini and KL against uniform are not. `max share` is the busiest expert's load as a multiple of fair share, `top-10%` the fraction of routing slots the busiest tenth of experts absorbs (0.100 under perfect balance).

Gold is measured on the same strided positions, so every row is comparable. Dead and unused counts are **not** reported here - at 1 token in 8 the zero counts are inflated, which is exactly why the table above uses the full-resolution fields.

| Run | Gini (mean / max) | Max share | Top-10% share | KL(load &#124;&#124; uniform) mean |
|-----|-------------------|-----------|---------------|----------------------------|
| gold BF16 | 0.273 / 0.345 | 3.98x | 0.200 | 0.1307 |
| uniform INT8 | 0.273 / 0.348 | 3.68x | 0.199 | 0.1289 |
| mixed INT8 | 0.273 / 0.346 | 3.71x | 0.199 | 0.1292 |
| placebo INT8 | 0.273 / 0.348 | 3.68x | 0.199 | 0.1289 |
| uniform INT4 | 0.271 / 0.360 | 4.15x | 0.199 | 0.1291 |
| mixed INT4 | 0.273 / 0.346 | 4.14x | 0.200 | 0.1302 |
| placebo INT4 | 0.271 / 0.360 | 4.15x | 0.199 | 0.1291 |
| attention INT4 | 0.268 / 0.356 | 3.97x | 0.197 | 0.1263 |
| uniform INT3 | 0.307 / 0.384 | 5.71x | 0.214 | 0.1630 |
| mixed INT3 | 0.284 / 0.374 | 4.59x | 0.206 | 0.1405 |
| placebo INT3 | 0.307 / 0.385 | 5.71x | 0.214 | 0.1630 |
| attention INT3 | 0.286 / 0.365 | 4.96x | 0.206 | 0.1419 |

## Why dead counts and ratio metrics come from different resolutions

The candidate router inputs were captured every 8th token, so the reconstruction sees one token in 8. Ratio statistics are unaffected, but a smaller sample makes low counts hit zero for reasons that have nothing to do with quantization. Measured on the same runs:

| Run | Dead slots, full resolution | Dead slots, strided | Unused slots, full | Unused slots, strided |
|-----|-----------------------------|---------------------|--------------------|-----------------------|
| gold BF16 | 17 | 18 | 0 | 0 |
| uniform INT8 | 13 | 14 | 0 | 0 |
| mixed INT8 | 13 | 13 | 0 | 0 |
| placebo INT8 | 13 | 14 | 0 | 0 |
| uniform INT4 | 15 | 13 | 0 | 0 |
| mixed INT4 | 15 | 14 | 0 | 0 |
| placebo INT4 | 15 | 13 | 0 | 0 |
| attention INT4 | 12 | 15 | 0 | 0 |
| uniform INT3 | 6 | 8 | 0 | 0 |
| mixed INT3 | 9 | 9 | 0 | 0 |
| placebo INT3 | 6 | 8 | 0 | 0 |
| attention INT3 | 8 | 10 | 0 | 0 |

The inflation is systematic and affects gold as much as the candidates, so the full-resolution fields are the ones quoted for dead and unused experts and the reconstruction is used only for the shape statistics.
