# olmoe - expert collapse

Per-layer expert-load balance for every run, from data already on disk. The pooled entropy in `summary.md` concatenates all layers into one histogram before measuring, which cancels independent per-layer imbalances; everything below is resolved per layer.

## Did collapse occur on olmoe?

Baseline: the unquantized model already leaves 0 of 1024 expert slots below a tenth of fair share (0.0%), with 0 receiving nothing. Natural routers are not balanced, so a candidate is only collapsing if it is worse than this.

- **INT4: no collapse beyond the unquantized baseline under `uniform`.** 
  Dead slots 0/1024 (0.0%) under `uniform` vs 0/1024 (0.0%) under `mixed`; layers containing a fully unused expert 0 vs 0; worst-layer entropy 0.9639 vs 0.9637; worst-layer max/mean load 4.27 vs 4.17.
  Load-distribution KL against uniform (strided) 0.0682 under `uniform` vs 0.0701 under `mixed`, a 0.97x reduction; Gini 0.196 vs 0.200.
  Nothing to mitigate at this precision; the comparison is a null and the boundary condition is worth reporting as such.

## Per-layer expert collapse (full resolution, 32,768 tokens per layer)

Dead means an expert receiving less than a tenth of its fair share of tokens in that layer; unused means exactly zero selections. Counts are summed over layers and expressed against the total expert *slots* (layers x experts), since expert *j* in one layer is a different expert from expert *j* in another.

| Run | Dead slots | Unused slots | Layers with an unused expert | Worst-layer norm. entropy | Worst-layer max/mean load |
|-----|------------|--------------|------------------------------|--------------------------|---------------------------|
| gold BF16 | 0 / 1024 (0.0%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9690 (layer 15) | 4.31 (layer 0) |
| uniform INT4 | 0 / 1024 (0.0%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9639 (layer 0) | 4.27 (layer 0) |
| mixed INT4 | 0 / 1024 (0.0%) | 0 / 1024 (0.00%) | 0 of 16 | 0.9637 (layer 0) | 4.17 (layer 0) |

## Why the pooled number missed this

Both columns are the same normalized usage entropy on the same run. The left one is what `summary.md` quotes; it concatenates every layer's tokens into a single histogram before measuring, so imbalances pointing in different directions across layers cancel.

| Run | Pooled (reported) | Worst layer | Mean over layers |
|-----|-------------------|-------------|------------------|
| gold BF16 | 0.9990 | 0.9690 | 0.9832 |
| uniform INT4 | 0.9989 | 0.9639 | 0.9825 |
| mixed INT4 | 0.9989 | 0.9637 | 0.9818 |

## Load-distribution shape (strided capture, 1 token in 8)

These are the sensitive instruments. Normalized entropy is maximal at balance, so its gradient there is zero; Gini and KL against uniform are not. `max share` is the busiest expert's load as a multiple of fair share, `top-10%` the fraction of routing slots the busiest tenth of experts absorbs (0.100 under perfect balance).

Gold is measured on the same strided positions, so every row is comparable. Dead and unused counts are **not** reported here - at 1 token in 8 the zero counts are inflated, which is exactly why the table above uses the full-resolution fields.

| Run | Gini (mean / max) | Max share | Top-10% share | KL(load &#124;&#124; uniform) mean |
|-----|-------------------|-----------|---------------|----------------------------|
| gold BF16 | 0.193 / 0.278 | 4.24x | 0.160 | 0.0653 |
| uniform INT4 | 0.196 / 0.286 | 4.20x | 0.160 | 0.0682 |
| mixed INT4 | 0.200 / 0.281 | 4.11x | 0.163 | 0.0701 |

## Why dead counts and ratio metrics come from different resolutions

The candidate router inputs were captured every 8th token, so the reconstruction sees one token in 8. Ratio statistics are unaffected, but a smaller sample makes low counts hit zero for reasons that have nothing to do with quantization. Measured on the same runs:

| Run | Dead slots, full resolution | Dead slots, strided | Unused slots, full | Unused slots, strided |
|-----|-----------------------------|---------------------|--------------------|-----------------------|
| gold BF16 | 0 | 0 | 0 | 0 |
| uniform INT4 | 0 | 0 | 0 | 0 |
| mixed INT4 | 0 | 0 | 0 | 0 |

The inflation is systematic and affects gold as much as the candidates, so the full-resolution fields are the ones quoted for dead and unused experts and the reconstruction is used only for the shape statistics.
