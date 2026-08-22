# qwen - Part 2 attribution

Routing drift split into the router's own rounded weights and the drift in the hidden states arriving at the router, rebuilt offline from the captured inputs.

| Bits | Weights only | Activations only | Both (uniform) | Weight share of both | Weight share of mechanisms |
|------|--------------|------------------|----------------|----------------------|----------------------------|
| INT8 | 8.81% | 13.04% | 15.70% | 56.1% [55.0, 57.3] | 40.3% [39.8, 40.9] |
| INT4 | 25.00% | 18.11% | 30.25% | 82.6% [81.9, 83.4] | 58.0% [57.6, 58.4] |
| INT3 | 45.76% | 55.06% | 68.58% | 66.7% [66.1, 67.3] | 45.4% [45.1, 45.6] |

## Routing KL by mechanism

| Bits | Weights only | Activations only | Both (uniform) |
|------|--------------|------------------|----------------|
| INT8 | 0.005292 | 0.020749 | 0.026038 |
| INT4 | 0.044367 | 0.038740 | 0.083575 |
| INT3 | 0.166601 | 0.314523 | 0.453984 |

## Reconstruction check

Cell A is rebuilt from the stored activations and weights; it must reproduce the routing decisions the gold run recorded.

- INT8: mean top-1 agreement with recorded gold logits 99.37%, worst layer 98.49% (passed)
- INT4: mean top-1 agreement with recorded gold logits 99.37%, worst layer 98.49% (passed)
- INT3: mean top-1 agreement with recorded gold logits 99.37%, worst layer 98.49% (passed)

## Reading these numbers

The two mechanisms are not additive, so the shares of `both` do not sum to 100%. A negative residual means the mechanisms flip overlapping sets of tokens: a token that either source alone would have flipped is counted once in `both` but twice across the two single-mechanism cells. The residual is reported rather than folded into a share: INT8 -39.1%, INT4 -42.5%, INT3 -47.0%.

The last column is the interpretable one: of the routing flips caused by exactly one mechanism, what fraction comes from the router's own weights. Above 50% means router protection addresses the dominant source at that precision.

## Cross-check against Part 1

Cell C is the `mixed` condition and cell D is `uniform` without routing feedback. Part 1 measured both on every token; the cells use the strided capture subset.

| Bits | C (rebuilt) | mixed (Part 1) | D (rebuilt) | uniform (Part 1) |
|------|-------------|----------------|-------------|------------------|
| INT8 | 13.04% | 12.75% | 15.70% | 15.65% |
| INT4 | 18.11% | 17.90% | 30.25% | 30.62% |
| INT3 | 55.06% | 55.69% | 68.58% | 72.48% |
