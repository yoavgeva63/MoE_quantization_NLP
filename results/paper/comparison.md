# Cross-model comparison

| Model | Policy | Bits | PPL | Routing KL | Top-1 flip |
|-------|--------|------|-----|------------|------------|
| OLMoE-1B-7B | gold | BF16 | 8.3586 | 0.000000 | 0.00% |
| OLMoE-1B-7B | uniform | INT8 | 9.3941 | 0.008535 | 18.01% |
| OLMoE-1B-7B | mixed | INT8 | 9.3234 | 0.006356 | 13.76% |
| OLMoE-1B-7B | uniform | INT4 | 10.4946 | 0.028579 | 33.01% |
| OLMoE-1B-7B | mixed | INT4 | 9.6938 | 0.009005 | 16.86% |
| OLMoE-1B-7B | placebo | INT4 | 10.4946 | 0.028579 | 33.01% |
| OLMoE-1B-7B | uniform | INT3 | 30.2290 | 0.104926 | 52.52% |
| OLMoE-1B-7B | mixed | INT3 | 18.6417 | 0.038463 | 32.15% |
| OLMoE-1B-7B | placebo | INT3 | 30.2259 | 0.104929 | 52.52% |
| Qwen1.5-MoE-A2.7B | gold | BF16 | 7.9687 | 0.000000 | 0.00% |
| Qwen1.5-MoE-A2.7B | uniform | INT8 | 9.3465 | 0.023470 | 15.65% |
| Qwen1.5-MoE-A2.7B | mixed | INT8 | 9.2846 | 0.017484 | 12.75% |
| Qwen1.5-MoE-A2.7B | uniform | INT4 | 12.0080 | 0.086117 | 30.62% |
| Qwen1.5-MoE-A2.7B | mixed | INT4 | 11.2445 | 0.035489 | 17.90% |
| Qwen1.5-MoE-A2.7B | placebo | INT4 | 12.0076 | 0.086117 | 30.62% |
| Qwen1.5-MoE-A2.7B | uniform | INT3 | 9555.4541 | 0.490062 | 72.48% |
| Qwen1.5-MoE-A2.7B | mixed | INT3 | 1236.4343 | 0.314730 | 55.69% |
| Qwen1.5-MoE-A2.7B | placebo | INT3 | 9563.0391 | 0.490072 | 72.48% |

## Does the effect replicate?

- OLMoE-1B-7B INT8: mixed cuts top-1 flips by 23.6% (intervals disjoint)
- OLMoE-1B-7B INT4: mixed cuts top-1 flips by 48.9%, placebo +0.0% (intervals disjoint)
- OLMoE-1B-7B INT3: mixed cuts top-1 flips by 38.8%, placebo -0.0% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT8: mixed cuts top-1 flips by 18.5% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT4: mixed cuts top-1 flips by 41.5%, placebo +0.0% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT3: mixed cuts top-1 flips by 23.2%, placebo +0.0% (intervals disjoint)
