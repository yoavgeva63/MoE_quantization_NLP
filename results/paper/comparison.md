# Cross-model comparison

| Model | Policy | Bits | PPL | Routing KL | Top-1 flip |
|-------|--------|------|-----|------------|------------|
| OLMoE-1B-7B | gold | BF16 | 8.3586 | 0.000000 | 0.00% |
| OLMoE-1B-7B | uniform | INT8 | 9.3941 | 0.008535 | 18.01% |
| OLMoE-1B-7B | mixed | INT8 | 9.3234 | 0.006356 | 13.76% |
| OLMoE-1B-7B | placebo | INT8 | 9.3941 | 0.008535 | 18.01% |
| OLMoE-1B-7B | uniform | INT4 | 10.4946 | 0.028579 | 33.01% |
| OLMoE-1B-7B | mixed | INT4 | 9.6938 | 0.009005 | 16.86% |
| OLMoE-1B-7B | placebo | INT4 | 10.4946 | 0.028579 | 33.01% |
| OLMoE-1B-7B | attention | INT4 | 9.7346 | 0.023662 | 30.65% |
| OLMoE-1B-7B | uniform | INT3 | 30.2290 | 0.104926 | 52.52% |
| OLMoE-1B-7B | mixed | INT3 | 18.6417 | 0.038463 | 32.15% |
| OLMoE-1B-7B | placebo | INT3 | 30.2259 | 0.104929 | 52.52% |
| OLMoE-1B-7B | attention | INT3 | 14.3592 | 0.077658 | 47.59% |
| Qwen1.5-MoE-A2.7B | gold | BF16 | 7.9687 | 0.000000 | 0.00% |
| Qwen1.5-MoE-A2.7B | uniform | INT8 | 9.3465 | 0.023470 | 15.65% |
| Qwen1.5-MoE-A2.7B | mixed | INT8 | 9.2846 | 0.017484 | 12.75% |
| Qwen1.5-MoE-A2.7B | placebo | INT8 | 9.3461 | 0.023470 | 15.65% |
| Qwen1.5-MoE-A2.7B | uniform | INT4 | 12.0080 | 0.086117 | 30.62% |
| Qwen1.5-MoE-A2.7B | mixed | INT4 | 11.2445 | 0.035489 | 17.90% |
| Qwen1.5-MoE-A2.7B | placebo | INT4 | 12.0076 | 0.086117 | 30.62% |
| Qwen1.5-MoE-A2.7B | attention | INT4 | 10.5069 | 0.069109 | 27.95% |
| Qwen1.5-MoE-A2.7B | uniform | INT3 | 9555.4541 | 0.490062 | 72.48% |
| Qwen1.5-MoE-A2.7B | mixed | INT3 | 1236.4343 | 0.314730 | 55.69% |
| Qwen1.5-MoE-A2.7B | placebo | INT3 | 9563.0391 | 0.490072 | 72.48% |
| Qwen1.5-MoE-A2.7B | attention | INT3 | 287.7667 | 0.337966 | 59.87% |

## Does the effect replicate?

- OLMoE-1B-7B INT8: mixed cuts top-1 flips by 23.6%, placebo -0.0% (intervals disjoint)
- OLMoE-1B-7B INT4: mixed cuts top-1 flips by 48.9%, placebo +0.0% (intervals disjoint)
- OLMoE-1B-7B INT3: mixed cuts top-1 flips by 38.8%, placebo -0.0% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT8: mixed cuts top-1 flips by 18.5%, placebo -0.0% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT4: mixed cuts top-1 flips by 41.5%, placebo +0.0% (intervals disjoint)
- Qwen1.5-MoE-A2.7B INT3: mixed cuts top-1 flips by 23.2%, placebo +0.0% (intervals disjoint)

## The `attention` control: routing versus perplexity

`captured` is the fraction of `mixed`'s improvement over `uniform` that `attention` reproduces. Negative means `attention` is better than `mixed`.

| Model | Bits | Metric | uniform | mixed | attention | mixed captured | attention captured |
|-------|------|--------|---------|-------|-----------|----------------|--------------------|
| OLMoE-1B-7B | INT4 | routing KL | 0.028579 | 0.009005 | 0.023662 | 100.0% | 25.1% |
| OLMoE-1B-7B | INT4 | top-1 flip rate | 0.3301 | 0.1686 | 0.3065 | 100.0% | 14.6% |
| OLMoE-1B-7B | INT4 | perplexity | 10.4946 | 9.6938 | 9.7346 | 100.0% | 94.9% |
| OLMoE-1B-7B | INT3 | routing KL | 0.104926 | 0.038463 | 0.077658 | 100.0% | 41.0% |
| OLMoE-1B-7B | INT3 | top-1 flip rate | 0.5252 | 0.3215 | 0.4759 | 100.0% | 24.2% |
| OLMoE-1B-7B | INT3 | perplexity | 30.2290 | 18.6417 | 14.3592 | 100.0% | 137.0% |
| Qwen1.5-MoE-A2.7B | INT4 | routing KL | 0.086117 | 0.035489 | 0.069109 | 100.0% | 33.6% |
| Qwen1.5-MoE-A2.7B | INT4 | top-1 flip rate | 0.3062 | 0.1790 | 0.2795 | 100.0% | 21.0% |
| Qwen1.5-MoE-A2.7B | INT4 | perplexity | 12.0080 | 11.2445 | 10.5069 | 100.0% | 196.6% |
| Qwen1.5-MoE-A2.7B | INT3 | routing KL | 0.490062 | 0.314730 | 0.337966 | 100.0% | 86.7% |
| Qwen1.5-MoE-A2.7B | INT3 | top-1 flip rate | 0.7248 | 0.5569 | 0.5987 | 100.0% | 75.1% |
| Qwen1.5-MoE-A2.7B | INT3 | perplexity | 9555.4541 | 1236.4343 | 287.7667 | 100.0% | 111.4% |
