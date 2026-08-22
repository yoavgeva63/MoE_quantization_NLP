# olmoe - Part 2 attribution

Routing drift split into the router's own rounded weights and the drift in the hidden states arriving at the router, rebuilt offline from the captured inputs.

| Bits | Weights only | Activations only | Both (uniform) | Weight share of both | Weight share of mechanisms |
|------|--------------|------------------|----------------|----------------------|----------------------------|
| INT8 | 10.38% | 13.54% | 17.54% | 59.2% [57.8, 60.2] | 43.4% [42.7, 43.9] |
| INT4 | 27.40% | 16.62% | 31.41% | 87.2% [86.2, 88.2] | 62.2% [61.7, 62.8] |
| INT3 | 43.78% | 31.74% | 50.01% | 87.5% [86.7, 88.3] | 58.0% [57.6, 58.4] |

## Routing KL by mechanism

| Bits | Weights only | Activations only | Both (uniform) |
|------|--------------|------------------|----------------|
| INT8 | 0.001833 | 0.007284 | 0.009110 |
| INT4 | 0.017479 | 0.009920 | 0.027223 |
| INT3 | 0.058610 | 0.039325 | 0.092915 |

## Reconstruction check

Cell A is rebuilt from the stored activations and weights; it must reproduce the routing decisions the gold run recorded.

- INT8: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)
- INT4: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)
- INT3: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)

## Reading these numbers

The two mechanisms are not additive, so the shares of `both` do not sum to 100%. A negative residual means the mechanisms flip overlapping sets of tokens: a token that either source alone would have flipped is counted once in `both` but twice across the two single-mechanism cells. The residual is reported rather than folded into a share: INT8 -36.4%, INT4 -40.2%, INT3 -51.0%.

The last column is the interpretable one: of the routing flips caused by exactly one mechanism, what fraction comes from the router's own weights. Above 50% means router protection addresses the dominant source at that precision.

## Cross-check against Part 1

Cell C is the `mixed` condition and cell D is `uniform` without routing feedback. Part 1 measured both on every token; the cells use the strided capture subset.

| Bits | C (rebuilt) | mixed (Part 1) | D (rebuilt) | uniform (Part 1) |
|------|-------------|----------------|-------------|------------------|
| INT8 | 13.54% | 13.76% | 17.54% | 18.01% |
| INT4 | 16.62% | 16.86% | 31.41% | 33.01% |
| INT3 | 31.74% | 32.15% | 50.01% | 52.52% |
