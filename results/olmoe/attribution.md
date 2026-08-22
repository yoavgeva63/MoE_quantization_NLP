# olmoe - Part 2 attribution

Routing drift split into the router's own rounded weights and the drift in the hidden states arriving at the router, rebuilt offline from the captured inputs.

Cell D (`both`) is **not** the `uniform` run: it pairs quantized router weights with the `mixed` run's activations, so it omits the feedback of changed routing on later layers' hidden states. The cross-check at the bottom measures that gap.

| Bits | Weights only | Activations only | Both (no routing feedback) | Weight share of both | Weight share of mechanisms |
|------|--------------|------------------|---------------------------|----------------------|----------------------------|
| INT8 | 10.38% | 13.54% | 17.54% | 59.2% [58.0, 60.4] | 43.4% [42.7, 44.1] |
| INT4 | 27.40% | 16.62% | 31.41% | 87.2% [86.3, 88.2] | 62.2% [61.7, 62.8] |
| INT3 | 43.78% | 31.74% | 50.01% | 87.5% [86.7, 88.3] | 58.0% [57.6, 58.4] |

## Routing KL by mechanism

| Bits | Weights only | Activations only | Both (no routing feedback) |
|------|--------------|------------------|---------------------------|
| INT8 | 0.001833 | 0.007284 | 0.009110 |
| INT4 | 0.017479 | 0.009920 | 0.027223 |
| INT3 | 0.058610 | 0.039325 | 0.092915 |

## Reconstruction check

Cell A is rebuilt from the stored activations and weights; it must reproduce the routing decisions the gold run recorded.

- INT8: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)
- INT4: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)
- INT3: mean top-1 agreement with recorded gold logits 99.66%, worst layer 99.46% (passed)

## Reading these numbers

The two mechanisms are not additive, so the shares of `both` do not sum to 100% and **must not be quoted as a partition** of the form "X% router weights, (100-X)% activation drift". A negative residual means the mechanisms flip overlapping sets of tokens: a token that either source alone would have flipped is counted once in `both` but twice across the two single-mechanism cells. The residual is reported with its own interval rather than folded into a share: INT8 -36.4% [-38.0, -34.9], INT4 -40.2% [-41.4, -38.9], INT3 -51.0% [-52.1, -49.9].

The last column is the interpretable one: of the routing flips caused by exactly one mechanism, what fraction comes from the router's own weights. Above 50% means router protection addresses the dominant source at that precision. This is the quantity the paper should quote, because it is a share of a well-defined set.

Every cell also carries a reconstruction-noise floor: cell A is rebuilt from fp16-stored activations in CPU fp32 against a bf16 GPU forward pass, so the agreement reported above is the ceiling on how exactly any cell can reproduce the run. All the differences reported here are well clear of it.

## Cross-check against Part 1

Cell C is the `mixed` condition and cell D is `uniform` without routing feedback. Part 1 measured both on every token; the cells use the strided capture subset.

| Bits | C (rebuilt) | mixed (Part 1) | D (rebuilt) | uniform (Part 1) |
|------|-------------|----------------|-------------|------------------|
| INT8 | 13.54% | 13.76% | 17.54% | 18.01% |
| INT4 | 16.62% | 16.86% | 31.41% | 33.01% |
| INT3 | 31.74% | 32.15% | 50.01% | 52.52% |
