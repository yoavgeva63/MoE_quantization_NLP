# olmoe - Part 1 results

## Results

| Policy | Bits | PPL | Routing KL | Top-1 flip | Jaccard dist | Usage entropy |
|--------|------|-----|------------|------------|--------------|---------------|
| gold | BF16 | 8.3586 | 0.000000 | 0.00% | 0.0000 | 0.9975 |
| uniform | INT8 | 9.3941 | 0.008535 | 18.01% | 0.1445 | 0.9977 |
| mixed | INT8 | 9.3234 | 0.006356 | 13.76% | 0.1277 | 0.9976 |
| placebo | INT8 | 9.3941 | 0.008535 | 18.01% | 0.1445 | 0.9977 |
| uniform | INT4 | 10.4946 | 0.028579 | 33.01% | 0.2362 | 0.9976 |
| mixed | INT4 | 9.6938 | 0.009005 | 16.86% | 0.1494 | 0.9976 |
| placebo | INT4 | 10.4946 | 0.028579 | 33.01% | 0.2362 | 0.9976 |
| attention | INT4 | 9.7346 | 0.023662 | 30.65% | 0.2063 | 0.9976 |
| uniform | INT3 | 30.2290 | 0.104926 | 52.52% | 0.4341 | 0.9968 |
| mixed | INT3 | 18.6417 | 0.038463 | 32.15% | 0.2771 | 0.9973 |
| placebo | INT3 | 30.2259 | 0.104929 | 52.52% | 0.4341 | 0.9968 |
| attention | INT3 | 14.3592 | 0.077658 | 47.59% | 0.3627 | 0.9974 |

## Part 1 decision gate

- INT8 routing KL: uniform 0.008535 [0.008326, 0.008758] vs mixed 0.006356 [0.006160, 0.006550] -> **MIXED WINS**
- INT8 top-1 flip rate: uniform 0.180132 [0.178133, 0.181826] vs mixed 0.137630 [0.136053, 0.139009] -> **MIXED WINS**
- INT4 routing KL: uniform 0.028579 [0.028096, 0.029102] vs mixed 0.009005 [0.008730, 0.009304] -> **MIXED WINS**
- INT4 top-1 flip rate: uniform 0.330120 [0.327099, 0.333057] vs mixed 0.168568 [0.167144, 0.169880] -> **MIXED WINS**
- INT3 routing KL: uniform 0.104926 [0.102413, 0.107471] vs mixed 0.038463 [0.036767, 0.040156] -> **MIXED WINS**
- INT3 top-1 flip rate: uniform 0.525169 [0.522129, 0.528336] vs mixed 0.321545 [0.318613, 0.323855] -> **MIXED WINS**
- INT8 perplexity: uniform 9.3941 vs mixed 9.3234
- INT4 perplexity: uniform 10.4946 vs mixed 9.6938
- INT3 perplexity: uniform 30.2290 vs mixed 18.6417

**Verdict: at least one bit-width shows a real advantage for router protection.** Run the parameter-count-matched placebo control before claiming it, to rule out that protecting any equally-sized set of weights would do as well. The placebo is matched on parameter count only - see `verification.md` for what it protects and how few tokens reach it.

## Correctness gates

- **attention INT3**: 3088 modules quantized, 16/16 routers quantized, bit-width check passed
- **attention INT4**: 3088 modules quantized, 16/16 routers quantized, bit-width check passed
- **gold BF16**: 0 modules quantized, 0/16 routers quantized, bit-width check passed
  - gold self-comparison: KL=0.00e+00, top-1 error=0.00e+00 (passed)
- **mixed INT3**: 3136 modules quantized, 0/16 routers quantized, bit-width check passed
- **mixed INT4**: 3136 modules quantized, 0/16 routers quantized, bit-width check passed
- **mixed INT8**: 3136 modules quantized, 0/16 routers quantized, bit-width check passed
- **placebo INT3**: 3151 modules quantized, 16/16 routers quantized, bit-width check passed
- **placebo INT4**: 3151 modules quantized, 16/16 routers quantized, bit-width check passed
- **placebo INT8**: 3151 modules quantized, 16/16 routers quantized, bit-width check passed
- **uniform INT3**: 3152 modules quantized, 16/16 routers quantized, bit-width check passed
- **uniform INT4**: 3152 modules quantized, 16/16 routers quantized, bit-width check passed
- **uniform INT8**: 3152 modules quantized, 16/16 routers quantized, bit-width check passed
- Router parameters: 2,097,152 of 6,919,161,856 (0.0303% of the model)
