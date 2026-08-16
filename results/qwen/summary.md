# qwen - Part 1 results

## Results

| Policy | Bits | PPL | Routing KL | Top-1 flip | Jaccard dist | Usage entropy |
|--------|------|-----|------------|------------|--------------|---------------|
| gold | BF16 | 7.9687 | 0.000000 | 0.00% | 0.0000 | 0.9996 |
| uniform | INT8 | 9.3465 | 0.023470 | 15.65% | 0.2118 | 0.9996 |
| mixed | INT8 | 9.2846 | 0.017484 | 12.75% | 0.1853 | 0.9996 |
| uniform | INT4 | 12.0080 | 0.086117 | 30.62% | 0.3558 | 0.9993 |
| mixed | INT4 | 11.2445 | 0.035489 | 17.90% | 0.2444 | 0.9995 |
| uniform | INT3 | 9555.4541 | 0.490062 | 72.48% | 0.7444 | 0.9929 |
| mixed | INT3 | 1236.4343 | 0.314730 | 55.69% | 0.6166 | 0.9964 |

## Part 1 decision gate

- INT8 routing KL: uniform 0.023470 [0.022859, 0.024039] vs mixed 0.017484 [0.016887, 0.018043] -> **MIXED WINS**
- INT8 top-1 flip rate: uniform 0.156452 [0.154739, 0.158031] vs mixed 0.127462 [0.125777, 0.128955] -> **MIXED WINS**
- INT4 routing KL: uniform 0.086117 [0.084862, 0.087378] vs mixed 0.035489 [0.034591, 0.036461] -> **MIXED WINS**
- INT4 top-1 flip rate: uniform 0.306221 [0.304423, 0.308183] vs mixed 0.179049 [0.176900, 0.181250] -> **MIXED WINS**
- INT3 routing KL: uniform 0.490062 [0.484581, 0.496481] vs mixed 0.314730 [0.310617, 0.319030] -> **MIXED WINS**
- INT3 top-1 flip rate: uniform 0.724822 [0.721136, 0.728533] vs mixed 0.556900 [0.554063, 0.559925] -> **MIXED WINS**
- INT8 perplexity: uniform 9.3465 vs mixed 9.2846
- INT4 perplexity: uniform 12.0080 vs mixed 11.2445
- INT3 perplexity: uniform 9555.4541 vs mixed 1236.4343

**Verdict: at least one bit-width shows a real advantage for router protection.** Run the parameter-matched placebo control before claiming it, to rule out that any protected 0.02% would do as well.

## Correctness gates

- **gold BF16**: 0 modules quantized, 0/24 routers quantized, bit-width check passed
  - gold self-comparison: KL=0.00e+00, top-1 error=0.00e+00 (passed)
- **mixed INT3**: 4512 modules quantized, 0/24 routers quantized, bit-width check passed
- **mixed INT4**: 4512 modules quantized, 0/24 routers quantized, bit-width check passed
- **mixed INT8**: 4512 modules quantized, 0/24 routers quantized, bit-width check passed
- **uniform INT3**: 4536 modules quantized, 24/24 routers quantized, bit-width check passed
- **uniform INT4**: 4536 modules quantized, 24/24 routers quantized, bit-width check passed
- **uniform INT8**: 4536 modules quantized, 24/24 routers quantized, bit-width check passed
- Router parameters: 2,949,120 of 14,315,784,192 (0.0206% of the model)
