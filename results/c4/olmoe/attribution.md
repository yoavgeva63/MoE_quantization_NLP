# olmoe - Part 2 attribution

Routing drift split into the router's own rounded weights and the drift in the hidden states arriving at the router, rebuilt offline from the captured inputs.

Cell D (`both`) is **not** the `uniform` run: it pairs quantized router weights with the `mixed` run's activations, so it omits the feedback of changed routing on later layers' hidden states. The cross-check at the bottom measures that gap.

| Bits | Weights only | Activations only | Both (no routing feedback) | Weight share of both | Weight share of mechanisms |
|------|--------------|------------------|---------------------------|----------------------|----------------------------|
| INT4 | 28.66% | 16.84% | 32.60% | 87.9% [87.0, 88.8] | 63.0% [62.4, 63.5] |

## Routing KL by mechanism

| Bits | Weights only | Activations only | Both (no routing feedback) |
|------|--------------|------------------|---------------------------|
| INT4 | 0.017238 | 0.009025 | 0.026045 |

## Reconstruction check

Cell A is rebuilt from the stored activations and weights; it must reproduce the routing decisions the gold run recorded.

- INT4: mean top-1 agreement with recorded gold logits 99.70%, worst layer 99.49% (passed)

## Reading these numbers

The two mechanisms are not additive, so the shares of `both` do not sum to 100% and **must not be quoted as a partition** of the form "X% router weights, (100-X)% activation drift". A negative residual means the mechanisms flip overlapping sets of tokens: a token that either source alone would have flipped is counted once in `both` but twice across the two single-mechanism cells. The residual is reported with its own interval rather than folded into a share: INT4 -39.6% [-40.7, -38.5].

The last column is the interpretable one: of the routing flips caused by exactly one mechanism, what fraction comes from the router's own weights. Above 50% means router protection addresses the dominant source at that precision. This is the quantity the paper should quote, because it is a share of a well-defined set.

Every cell also carries a reconstruction-noise floor: cell A is rebuilt from fp16-stored activations in CPU fp32 against a bf16 GPU forward pass, so the agreement reported above is the ceiling on how exactly any cell can reproduce the run. All the differences reported here are well clear of it.

## Cross-check against Part 1

Cell C is the `mixed` condition and cell D is `uniform` without routing feedback. Part 1 measured both on every token; the cells use the strided capture subset.

| Bits | C (rebuilt) | mixed (Part 1) | D (rebuilt) | uniform (Part 1) |
|------|-------------|----------------|-------------|------------------|
| INT4 | 16.84% | 17.00% | 32.60% | 33.95% |
