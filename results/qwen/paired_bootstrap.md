# qwen - paired-difference bootstrap

Supplementary to the Part 1 decision gate, which compares two independently bootstrapped intervals. Both arms use `seed=42` over the same 128 sequence groups, so their replicates are the same resamples and the difference can be bootstrapped directly. Rebuilt from the saved router inputs at one token in 8; 10,000 replicates, sequence-level resampling.

## Paired-difference intervals on `mixed` - `uniform`

Negative means `mixed` is better. Each replicate resamples the 128 sequence groups once and differences the two arms within that resample, so the sequence-level variation they share cancels instead of being counted twice.

| Bits | Metric | uniform | mixed | difference | 95% paired CI | replicates &gt; 0 |
|------|--------|---------|-------|------------|---------------|------------------|
| INT8 | routing KL | 0.026579 | 0.020770 | -0.005809 | [-0.005966, -0.005660] | 0.00% |
| INT8 | top-1 flip rate | 0.159047 | 0.130269 | -0.028778 | [-0.030609, -0.026978] | 0.00% |
| INT4 | routing KL | 0.089627 | 0.038760 | -0.050866 | [-0.051526, -0.050205] | 0.00% |
| INT4 | top-1 flip rate | 0.310303 | 0.181061 | -0.129242 | [-0.132833, -0.125661] | 0.00% |
| INT3 | routing KL | 0.491533 | 0.314537 | -0.176996 | [-0.181264, -0.173012] | 0.00% |
| INT3 | top-1 flip rate | 0.716492 | 0.550273 | -0.166219 | [-0.171214, -0.161346] | 0.00% |

## What the pairing buys

Both columns are computed on the same reconstructed per-token values with the same resamples. The unpaired column is the gap between the two marginal intervals - the quantity the Part 1 disjointness test inspects - and the paired column is the interval on the difference itself.

| Bits | Metric | Unpaired interval width (sum of both arms) | Paired interval width | Narrower by |
|------|--------|--------------------------------------------|-----------------------|-------------|
| INT8 | routing KL | 0.002843 | 0.000306 | 9.29x |
| INT8 | top-1 flip rate | 0.011719 | 0.003632 | 3.23x |
| INT4 | routing KL | 0.005154 | 0.001321 | 3.90x |
| INT4 | top-1 flip rate | 0.014323 | 0.007172 | 2.00x |
| INT3 | routing KL | 0.023389 | 0.008252 | 2.83x |
| INT3 | top-1 flip rate | 0.021373 | 0.009868 | 2.17x |

## Reconstruction against the recorded Part 1 numbers

The paired test is computed on the strided capture, so its marginal means are not expected to equal Part 1's numbers exactly - Part 1 measured every token, and the rebuilt logits carry a small reconstruction floor from fp16-stored activations recomputed in CPU fp32 against a bf16 GPU forward pass. They should be close, and they are:

| Bits | Metric | uniform rebuilt | uniform recorded | mixed rebuilt | mixed recorded |
|------|--------|-----------------|------------------|---------------|----------------|
| INT8 | routing KL | 0.026579 | 0.023470 | 0.020770 | 0.017484 |
| INT8 | top-1 flip rate | 0.159047 | 0.156452 | 0.130269 | 0.127462 |
| INT4 | routing KL | 0.089627 | 0.086117 | 0.038760 | 0.035489 |
| INT4 | top-1 flip rate | 0.310303 | 0.306221 | 0.181061 | 0.179049 |
| INT3 | routing KL | 0.491533 | 0.490062 | 0.314537 | 0.314730 |
| INT3 | top-1 flip rate | 0.716492 | 0.724822 | 0.550273 | 0.556900 |

## Which number the paper should quote

**Quote the Part 1 intervals from `summary.md` as the headline, and this paired test as the confirmatory one.** The Part 1 numbers are measured on all 32,768 routing tokens per layer, which is the full sample the experiment collected; the paired test here is on one token in 8, because the per-token metric values were never saved and only the strided router inputs can be replayed offline. Trading the full sample for the pairing is not worth it for the headline.

What the paired test settles is the *method* objection: that comparing two separately-bootstrapped intervals discards a pairing that is present in the data. It does, and correcting it moves the conclusion in the safe direction. Every cell separates from zero in the paired test as well, so no claim in `summary.md` depends on the choice of test.

Recommended wording: report the Part 1 means and intervals, describe the disjointness check as a conservative stand-in for a paired-difference test, and cite this table for the paired confirmation.
