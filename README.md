# Optimal Quantization in Mixture-of-Experts

Does keeping only the tiny router layers in high precision protect an MoE model under
post-training quantization?

NLP final project, Tel Aviv University. Tomer Alfandary and Yoav Geva.

```bash
pip install -e ".[dev]" && pytest              # 136 tests, CPU only, ~6s
python scripts/inspect_model.py olmoe          # verify architecture, no GPU
python scripts/run.py --config configs/olmoe.yaml \
    --policies gold uniform mixed --bits 8 4 3
python scripts/analyze.py --results-dir results/olmoe
```

## Documentation

| | |
|---|---|
| **[OVERVIEW.md](OVERVIEW.md)** | The research question, the two-part plan, where the literature stands |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How the code is built and why |
| **[RUNNING.md](RUNNING.md)** | Installation, cluster setup, running a sweep, troubleshooting |

## The experiment

Three configurations differing in exactly one respect — whether the routers are quantized:

| Config | Experts | Routers |
|--------|---------|---------|
| `gold` | BF16 | BF16 |
| `uniform` | INT*N* | INT*N* |
| `mixed` | INT*N* | BF16 |

Swept across INT8/4/3 on OLMoE-1B-7B and Qwen1.5-MoE-A2.7B, measured by routing KL
divergence, top-*k* expert mismatch, expert-usage entropy, perplexity, and
output-distribution drift, all with bootstrap confidence intervals.

`scripts/analyze.py` then prints a verdict on whether `mixed` genuinely beats `uniform` or
whether their error bars overlap.

## Status

Part 1 is implemented and tested. No cluster runs yet.
