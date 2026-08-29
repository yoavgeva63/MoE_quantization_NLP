# qwen - Part 1 results

## Results

| Policy | Bits | PPL | Routing KL | Top-1 flip | Jaccard dist | Usage entropy |
|--------|------|-----|------------|------------|--------------|---------------|
| gold | BF16 | 10.3631 | 0.000000 | 0.00% | 0.0000 | 0.9999 |
| uniform | INT4 | 16.0001 | 0.073746 | 30.03% | 0.3636 | 0.9997 |
| mixed | INT4 | 14.8474 | 0.027177 | 16.57% | 0.2420 | 0.9998 |

## Part 1 decision gate

- INT4 routing KL: uniform 0.073746 [0.072276, 0.075176] vs mixed 0.027177 [0.026207, 0.028180] -> **MIXED WINS**
- INT4 top-1 flip rate: uniform 0.300345 [0.297908, 0.303075] vs mixed 0.165653 [0.163156, 0.168672] -> **MIXED WINS**
- INT4 perplexity: uniform 16.0001 vs mixed 14.8474

**Verdict: at least one bit-width shows a real advantage for router protection.** Run the parameter-count-matched placebo control before claiming it, to rule out that protecting any equally-sized set of weights would do as well. The placebo is matched on parameter count only - see `verification.md` for what it protects and how few tokens reach it.

## Correctness gates

- **gold BF16**: 0 modules quantized, 0/24 routers quantized, bit-width check passed
  - gold self-comparison: KL=0.00e+00, top-1 error=0.00e+00 (passed)
- **mixed INT4**: 4512 modules quantized, 0/24 routers quantized, bit-width check passed
- **uniform INT4**: 4536 modules quantized, 24/24 routers quantized, bit-width check passed
- Router parameters: 2,949,120 of 14,315,784,192 (0.0206% of the model)
