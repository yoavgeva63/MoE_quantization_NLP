# Does Protecting the Router in Quantized MoE Improve Quality, and How?

A Mixture-of-Experts layer routes each token through a small linear **router** and then
through a handful of **experts**. The experts are almost the whole model; the routers are
0.02–0.03% of it. This project asks a narrow question:

> Under calibration-free, weight-only post-training quantization, does leaving **only** the
> routers in BF16 protect routing and language-model quality?

Final project for the NLP course at Tel Aviv University.
Tomer Alfandary and Yoav Geva. Paper: [`paper/main.pdf`](paper/main.pdf).

## Headline result

Protecting the routers reduces routing drift in **every** model × bit-width cell we ran,
with disjoint 95% bootstrap intervals, and a parameter-count-matched placebo reproduces the
unprotected baseline to several decimals. It is **not** sufficient for end-task quality: at
INT8 most of the residual routing error comes from the activations arriving at the router,
not from the router's own rounded weights.

WikiText-2, routing KL and perplexity, `uniform` → `mixed`:

| Model | Bits | Routing KL | Perplexity |
|---|---|---|---|
| OLMoE-1B-7B (gold 8.36) | INT8 | 0.0085 → 0.0064 | 9.39 → 9.32 |
| | INT4 | 0.0286 → 0.0090 | 10.49 → 9.69 |
| | INT3 | 0.1049 → 0.0385 | 30.23 → 18.64 |
| Qwen1.5-MoE-A2.7B (gold 7.97) | INT8 | 0.0235 → 0.0175 | 9.35 → 9.28 |
| | INT4 | 0.0861 → 0.0355 | 12.01 → 11.24 |
| | INT3 | 0.4901 → 0.3147 | 9555 → 1236 |

Qwen INT3 is saturation, not evidence: both policies are destroyed, so we use INT3 for
routing and collapse metrics only and make perplexity claims at INT8 and INT4.

## The experiment

Five policies, each differing from `uniform` in exactly one respect:

| Policy | Experts | Routers | Purpose |
|---|---|---|---|
| `gold` | BF16 | BF16 | upper bound |
| `uniform` | INT*N* | INT*N* | negative baseline |
| `mixed` | INT*N* | BF16 | the proposal |
| `placebo` | INT*N* | INT*N* | control: one random non-router module of matched parameter count is spared instead |
| `attention` | INT*N* | INT*N* | control: attention projections spared instead, same every-token exposure |

Swept over INT8 / INT4 / INT3 on two architectures, scored on routing KL, top-1 expert flip
rate, top-*k* Jaccard distance, per-layer expert load, perplexity and output drift, with
sequence-level bootstrap intervals. There is **no calibration anywhere**: scales come from
the weight tensors alone, so no data touches the quantizer.

A second, offline analysis rebuilds routing decisions under four combinations of
gold/quantized router weights and gold/quantized incoming activations, splitting routing
drift into the router's own rounding versus upstream activation drift. It costs no GPU time.

WikiText-2 throughout, plus an INT4 out-of-domain check on C4 for both models.

## Quickstart

```bash
pip install -e ".[dev]" && pytest          # 188 tests, CPU only, no downloads
python scripts/inspect_model.py olmoe      # verify the registry, meta device, no GPU
python scripts/run.py --config configs/olmoe.yaml \
    --policies gold uniform mixed placebo attention --bits 8 4 3
python scripts/analyze.py --results-dir results/olmoe
```

Everything under `scripts/` except `run.py` is CPU-only and reads artifacts already on
disk, so the whole analysis side reruns without a GPU. See [RUNNING.md](RUNNING.md) for
installation, cluster setup and troubleshooting.

## Where every number in the paper comes from

Each table and figure in the paper traces to a committed artifact in `results/`. Nothing is
transcribed by hand.

| Paper | Produced by | Artifact |
|---|---|---|
| Tables 2, 3, 4, 8 — main results, flip rate, routing KL, Jaccard | `scripts/analyze.py` | `results/{olmoe,qwen}/summary.md` |
| Table 5 — attribution of top-1 flips | `scripts/attribute.py` | `results/{olmoe,qwen}/attribution.md` |
| Table 6 — architectures and router share | `scripts/inspect_model.py` | `results/{olmoe,qwen}/architecture.json` |
| Table 7 — expert collapse per layer | `scripts/collapse.py` | `results/{olmoe,qwen}/collapse.md` |
| Table 9 — ATTENTION routing KL and recovery | run output | `results/*/attention_int{8,4,3}/metrics.json` |
| Figure 1 — headline | `scripts/analyze.py` | `paper/figures/headline.pdf` |
| Figure 2 — starved experts by layer | `scripts/collapse.py` | `paper/figures/collapse_qwen.pdf` |
| §5, the C4 paragraph | `analyze.py`, `attribute.py` | `results/c4/{olmoe,qwen}/summary.md`, `attribution.md` |
| §5, disjoint bootstrap intervals | `scripts/paired_bootstrap.py` | `results/*/paired_bootstrap.md` |
| §4, the single-variable and bit-identity claims | `scripts/verify_offline.py` | `results/*/verification.md` |

The last two rows have no table of their own in the paper. The bit-identity audit and the
per-mechanism routing KL were cut from the appendix for length, so `verification.md` and
`attribution.md` in this repository are the only full record of them.

Every `metrics.json` is self-describing: config, seeds, torch/transformers/torchao versions,
GPU name and compute capability, git commit, the quantization audit, dataset statistics, and
all metrics with confidence intervals.

## Correctness gates

Every failure mode here is silent. A quantizer that skips a module it does not recognise
still returns a working model with a plausible perplexity; a router hook on the wrong module
still produces numbers. So runs abort rather than produce misleading output:

| Gate | Catches |
|---|---|
| gold self-comparison gives KL 0 and zero flips | nondeterminism anywhere upstream |
| `uniform` has **all** routers quantized | a silently skipped router making `uniform` == `mixed` |
| `mixed` has **no** router quantized | the skip rule not matching |
| something was quantized at all | `_default` matching nothing, e.g. fused expert stacks |
| distinct values per channel ≤ `2**bits` | a bit-width silently not applied |
| config expert count == router weight shape | the registry pointing at the wrong module |

`scripts/verify_offline.py` is the strong version: it re-reads the stored router weights and
counts distinct values in every output channel of every router in every layer of every run.
All runs pass on both models.

## Layout

```text
src/moequant/     the library: registry, quantization policy, capture, metrics,
                  collapse, attribution, evaluation, runner
scripts/          CLI entry points; everything but run.py is CPU-only
scripts/slurm/    the cluster job scripts actually used
configs/          olmoe.yaml, qwen.yaml, and their _c4 counterparts
tests/            188 tests, CPU only, no model downloads
results/          every committed artifact behind the paper
paper/            ACL source, figures, bibliography
```

## Documentation

| | |
|---|---|
| **[OVERVIEW.md](OVERVIEW.md)** | The research question, the two-part plan, where the literature stands |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | How the code is built and why |
| **[RUNNING.md](RUNNING.md)** | Installation, cluster setup, running a sweep, troubleshooting |

## Reuse

Coursework, shared so the results in the paper can be checked and reproduced. The two model
checkpoints and the WikiText-2 and C4 corpora are covered by their own upstream licenses.
