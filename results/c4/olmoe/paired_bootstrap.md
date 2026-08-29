# olmoe - paired-difference bootstrap

Supplementary to the Part 1 decision gate, which compares two independently bootstrapped intervals. Both arms use `seed=42` over the same 128 sequence groups, so their replicates are the same resamples and the difference can be bootstrapped directly. Rebuilt from the saved router inputs at one token in 8; 10,000 replicates, sequence-level resampling.

## Paired-difference intervals on `mixed` - `uniform`

Negative means `mixed` is better. Each replicate resamples the 128 sequence groups once and differences the two arms within that resample, so the sequence-level variation they share cancels instead of being counted twice.

| Bits | Metric | uniform | mixed | difference | 95% paired CI | replicates &gt; 0 |
|------|--------|---------|-------|------------|---------------|------------------|
| INT4 | routing KL | 0.028087 | 0.009027 | -0.019059 | [-0.019272, -0.018850] | 0.00% |
| INT4 | top-1 flip rate | 0.333054 | 0.168335 | -0.164719 | [-0.169815, -0.159607] | 0.00% |

## What the pairing buys

Both columns are computed on the same reconstructed per-token values with the same resamples. The unpaired column is the gap between the two marginal intervals - the quantity the Part 1 disjointness test inspects - and the paired column is the interval on the difference itself.

| Bits | Metric | Unpaired interval width (sum of both arms) | Paired interval width | Narrower by |
|------|--------|--------------------------------------------|-----------------------|-------------|
| INT4 | routing KL | 0.001337 | 0.000423 | 3.16x |
| INT4 | top-1 flip rate | 0.018219 | 0.010208 | 1.78x |

## Reconstruction against the recorded Part 1 numbers

The paired test is computed on the strided capture, so its marginal means are not expected to equal Part 1's numbers exactly - Part 1 measured every token, and the rebuilt logits carry a small reconstruction floor from fp16-stored activations recomputed in CPU fp32 against a bf16 GPU forward pass. They should be close, and they are:

| Bits | Metric | uniform rebuilt | uniform recorded | mixed rebuilt | mixed recorded |
|------|--------|-----------------|------------------|---------------|----------------|
| INT4 | routing KL | 0.028087 | 0.027266 | 0.009027 | 0.008310 |
| INT4 | top-1 flip rate | 0.333054 | 0.339539 | 0.168335 | 0.169977 |

## Which number the paper should quote

**Quote the Part 1 intervals from `summary.md` as the headline, and this paired test as the confirmatory one.** The Part 1 numbers are measured on all 32,768 routing tokens per layer, which is the full sample the experiment collected; the paired test here is on one token in 8, because the per-token metric values were never saved and only the strided router inputs can be replayed offline. Trading the full sample for the pairing is not worth it for the headline.

What the paired test settles is the *method* objection: that comparing two separately-bootstrapped intervals discards a pairing that is present in the data. It does, and correcting it moves the conclusion in the safe direction. Every cell separates from zero in the paired test as well, so no claim in `summary.md` depends on the choice of test.

Recommended wording: report the Part 1 means and intervals, describe the disjointness check as a conservative stand-in for a paired-difference test, and cite this table for the paired confirmation.
