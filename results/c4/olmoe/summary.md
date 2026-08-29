# olmoe - Part 1 results

## Results

| Policy | Bits | PPL | Routing KL | Top-1 flip | Jaccard dist | Usage entropy |
|--------|------|-----|------------|------------|--------------|---------------|
| gold | BF16 | 12.2879 | 0.000000 | 0.00% | 0.0000 | 0.9990 |
| uniform | INT4 | 14.1275 | 0.027266 | 33.95% | 0.2438 | 0.9989 |
| mixed | INT4 | 13.4498 | 0.008310 | 17.00% | 0.1528 | 0.9989 |

## Part 1 decision gate

- INT4 routing KL: uniform 0.027266 [0.026938, 0.027675] vs mixed 0.008310 [0.008029, 0.008595] -> **MIXED WINS**
- INT4 top-1 flip rate: uniform 0.339539 [0.334427, 0.343715] vs mixed 0.169977 [0.168455, 0.171660] -> **MIXED WINS**
- INT4 perplexity: uniform 14.1275 vs mixed 13.4498

**Verdict: at least one bit-width shows a real advantage for router protection.** Run the parameter-count-matched placebo control before claiming it, to rule out that protecting any equally-sized set of weights would do as well. The placebo is matched on parameter count only - see `verification.md` for what it protects and how few tokens reach it.

## Correctness gates

- **gold BF16**: 0 modules quantized, 0/16 routers quantized, bit-width check passed
  - gold self-comparison: KL=0.00e+00, top-1 error=0.00e+00 (passed)
- **mixed INT4**: 3136 modules quantized, 0/16 routers quantized, bit-width check passed
- **uniform INT4**: 3152 modules quantized, 16/16 routers quantized, bit-width check passed
- Router parameters: 2,097,152 of 6,919,161,856 (0.0303% of the model)
