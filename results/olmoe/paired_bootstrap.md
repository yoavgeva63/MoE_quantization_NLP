# olmoe - paired-difference bootstrap

Supplementary to the Part 1 decision gate, which compares two independently bootstrapped intervals. Both arms use `seed=42` over the same 128 sequence groups, so their replicates are the same resamples and the difference can be bootstrapped directly. Rebuilt from the saved router inputs at one token in 8; 10,000 replicates, sequence-level resampling.

## Paired-difference intervals on `mixed` - `uniform`

Negative means `mixed` is better. Each replicate resamples the 128 sequence groups once and differences the two arms within that resample, so the sequence-level variation they share cancels instead of being counted twice.

| Bits | Metric | uniform | mixed | difference | 95% paired CI | replicates &gt; 0 |
|------|--------|---------|-------|------------|---------------|------------------|
| INT8 | routing KL | 0.009409 | 0.007285 | -0.002123 | [-0.002178, -0.002071] | 0.00% |
| INT8 | top-1 flip rate | 0.177078 | 0.135178 | -0.041901 | [-0.044357, -0.039413] | 0.00% |
| INT4 | routing KL | 0.029576 | 0.009921 | -0.019654 | [-0.019996, -0.019350] | 0.00% |
| INT4 | top-1 flip rate | 0.322845 | 0.165863 | -0.156982 | [-0.161728, -0.152435] | 0.00% |
| INT3 | routing KL | 0.106419 | 0.039327 | -0.067093 | [-0.069430, -0.065174] | 0.00% |
| INT3 | top-1 flip rate | 0.518875 | 0.317230 | -0.201645 | [-0.206863, -0.196594] | 0.00% |

## What the pairing buys

Both columns are computed on the same reconstructed per-token values with the same resamples. The unpaired column is the gap between the two marginal intervals - the quantity the Part 1 disjointness test inspects - and the paired column is the interval on the difference itself.

| Bits | Metric | Unpaired interval width (sum of both arms) | Paired interval width | Narrower by |
|------|--------|--------------------------------------------|-----------------------|-------------|
| INT8 | routing KL | 0.001070 | 0.000107 | 9.96x |
| INT8 | top-1 flip rate | 0.011704 | 0.004944 | 2.37x |
| INT4 | routing KL | 0.001757 | 0.000645 | 2.72x |
| INT4 | top-1 flip rate | 0.016282 | 0.009293 | 1.75x |
| INT3 | routing KL | 0.009748 | 0.004256 | 2.29x |
| INT3 | top-1 flip rate | 0.021210 | 0.010269 | 2.07x |

## Reconstruction against the recorded Part 1 numbers

The paired test is computed on the strided capture, so its marginal means are not expected to equal Part 1's numbers exactly - Part 1 measured every token, and the rebuilt logits carry a small reconstruction floor from fp16-stored activations recomputed in CPU fp32 against a bf16 GPU forward pass. They should be close, and they are:

| Bits | Metric | uniform rebuilt | uniform recorded | mixed rebuilt | mixed recorded |
|------|--------|-----------------|------------------|---------------|----------------|
| INT8 | routing KL | 0.009409 | 0.008535 | 0.007285 | 0.006356 |
| INT8 | top-1 flip rate | 0.177078 | 0.180132 | 0.135178 | 0.137630 |
| INT4 | routing KL | 0.029576 | 0.028579 | 0.009921 | 0.009005 |
| INT4 | top-1 flip rate | 0.322845 | 0.330120 | 0.165863 | 0.168568 |
| INT3 | routing KL | 0.106419 | 0.104926 | 0.039327 | 0.038463 |
| INT3 | top-1 flip rate | 0.518875 | 0.525169 | 0.317230 | 0.321545 |

## Which number the paper should quote

**Quote the Part 1 intervals from `summary.md` as the headline, and this paired test as the confirmatory one.** The Part 1 numbers are measured on all 32,768 routing tokens per layer, which is the full sample the experiment collected; the paired test here is on one token in 8, because the per-token metric values were never saved and only the strided router inputs can be replayed offline. Trading the full sample for the pairing is not worth it for the headline.

What the paired test settles is the *method* objection: that comparing two separately-bootstrapped intervals discards a pairing that is present in the data. It does, and correcting it moves the conclusion in the safe direction. Every cell separates from zero in the paired test as well, so no claim in `summary.md` depends on the choice of test.

Recommended wording: report the Part 1 means and intervals, describe the disjointness check as a conservative stand-in for a paired-difference test, and cite this table for the paired confirmation.
