# Running the experiments

## Quick reference

```bash
pytest                                                    # 136 tests, CPU only, ~6s
python scripts/inspect_model.py olmoe                     # check the registry, no GPU
python scripts/run.py --config configs/olmoe.yaml \
    --policies gold uniform mixed --bits 8 4 3            # the Part 1 sweep
python scripts/analyze.py --results-dir results/olmoe     # figures + decision gate
```

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Dependencies are pinned in `pyproject.toml`, and the pins matter. **transformers must stay
on 4.x**: version 5 fuses MoE experts into 3D parameter stacks, which changes module
naming and causes `nn.Linear`-based quantization walkers to skip the experts entirely
while still reporting a successful load. `scripts/inspect_model.py` re-checks the layout
at runtime and warns if it sees the fused form.

Verify the install:

```bash
pytest -q
```

All 136 tests run on CPU with no model downloads. Most of the suite drives a synthetic
MoE through a stand-in quantizer, but `tests/test_torchao_backend.py` runs the **real**
torchao path — actual `FqnToConfig` targeting, actual quantized tensor subclasses — and
is the one that catches a torchao release moving the API out from under us. It skips
automatically if torchao is missing, so check it was not skipped:

```bash
pytest tests/test_torchao_backend.py -v
```

If those pass, the pipeline logic and the quantization backend are both sound, and the
only remaining variable is the real checkpoints.

## GPU requirements

The lab workstation's TITAN Xp cards are compute capability 6.1. They **cannot** run this:
the torch wheel ships kernels for sm_70 and up, so a TITAN Xp loads, reports
`cuda.is_available() == True`, and then dies on the first kernel launch. Do not trust
`torch.cuda.is_bf16_supported()` either — it returns `True` on these cards. Everything runs
on the TAU Slurm cluster.

A student account is limited to the `studentkillable` partition, whose only usable card is
the **RTX 2080 Ti (11 GB, sm_75)**; the TITAN Xp nodes on that same partition are dead
weight for us. Since nothing there is large enough to hold a model on its own, every job
shards across several cards through `device_map="auto"`, and every job pins the GPU type:

| Model | BF16 gold | What to request |
|-------|-----------|-----------------|
| OLMoE-1B-7B | ~14 GB | `--gres=gpu:geforce_rtx_2080:3` |
| Qwen1.5-MoE-A2.7B | ~28 GB | `--gres=gpu:geforce_rtx_2080:5` |

Switching to `dtype: float16` does not reduce this; both are two bytes per parameter.

`_preflight.sh` enforces both constraints before any GPU work: it aborts if any allocated
card is below the wheel's minimum architecture, and if the total VRAM is under
`MOEQUANT_MIN_VRAM_GB`, which each job script sets. The larger partitions (`killable`, with
a5000/3090/l40s/a6000) need a separate Slurm association; if you get one, drop the GPU-type
pin and go back to a single card.

## Cluster setup (once)

**The repo must live on `/home/morg`.** The lab workstation's home directory is a local
disk that shadows a different NFS home mounted by the compute nodes, so a repo cloned into
`~` is invisible to every job. `/home/morg/NLP_2526b/$USER` is the one path both machines
see, and it is where the caches and the venv belong anyway.

```bash
# 1. Fill the Slurm access form, listing "NLP class 2025/2026" as PI/lab:
#    https://www.cs.tau.ac.il/system/SlurmRequestForm

# 2. Everything lives in course storage; the home quota will not hold these models.
export STORAGE=/home/morg/NLP_2526b/$USER
export HF_HOME=$STORAGE/hf_cache
export HF_DATASETS_CACHE=$STORAGE/datasets_cache
export PIP_CACHE_DIR=$STORAGE/pip_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"
cd "$STORAGE/MoE_quantization_NLP" && mkdir -p logs

# 3. Create the venv. `python3 -m venv` fails on these hosts — ensurepip is not
#    installed — so bootstrap virtualenv into storage and use that instead.
pip install --target "$STORAGE/.tools" virtualenv
PYTHONPATH="$STORAGE/.tools" python3 -m virtualenv "$STORAGE/venv"
source "$STORAGE/venv/bin/activate"
pip install -e ".[dev]"
```

The Slurm scripts locate the environment themselves, in this order: `$MOEQUANT_VENV`,
then a local `.venv/`, then a `.deps/` directory added to `PYTHONPATH`. With the venv in
storage rather than in the repo, export `MOEQUANT_VENV=$STORAGE/venv`.

If the repo ever moves, the editable install keeps pointing at the old path and jobs will
silently run stale code. Re-point it with `pip install -e . --no-deps` from the new
location and confirm with `python -c "import moequant; print(moequant.__file__)"`.

Every job sources `scripts/slurm/_preflight.sh` first, which verifies the environment,
confirms torch sees a CUDA device, aborts if any allocated card is below the wheel's
minimum architecture or if the total VRAM is under `MOEQUANT_MIN_VRAM_GB`, and runs the
torchao backend tests. All of that fails in seconds rather than part-way into a job.

Course storage is **not backed up**. Push to GitHub before running anything long.

## Step 1: check the architecture before spending GPU hours

```bash
python scripts/inspect_model.py olmoe --out results/olmoe/architecture.json
```

Loads on the meta device, so it needs no GPU and downloads only the config. It reports:

- which modules the router pattern matched, and their shapes and types
- the parameter census, including the router fraction we cite in the paper
- whether experts are individual `nn.Linear` modules or fused 3D parameters
- a preview of how many modules each policy would quantize

**What to confirm before continuing:**

- routers matched equals the layer count (16 for OLMoE, 24 for Qwen)
- router fraction is a few hundredths of a percent (OLMoE: 2,097,152 of 6.92B = 0.0303%)
- expert layout says `individual-linear`, not `fused-3d`
- the policy preview shows `uniform` including all routers and `mixed` including none

## Step 2: run the sweep

Gold must exist before anything else, since every metric is relative to it. `run.py`
orders jobs so this happens automatically when `gold` is in `--policies`.

```bash
python scripts/run.py \
    --config configs/olmoe.yaml \
    --policies gold uniform mixed \
    --bits 8 4 3
```

That is 7 runs: one gold plus two policies at three bit-widths.

Or submit to Slurm, which also runs the inspection and analysis:

```bash
sbatch scripts/slurm/run_olmoe.sh
sbatch scripts/slurm/run_qwen.sh
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--keep-going` | continue the sweep after a failure instead of stopping |
| `--routing-sequences 32` | smaller routing subset, for a fast smoke test |
| `--max-ppl-windows 20` | cap perplexity windows, for a fast smoke test |
| `--results-dir /path` | write elsewhere |
| `--cache-dir /path` | override the HuggingFace cache |

A quick end-to-end check on real weights before committing to the full sweep, written to a
throwaway `results/smoke/` so a later sweep does not mistake it for gold:

```bash
sbatch scripts/slurm/run_smoke.sh olmoe
# or, on a machine with a usable GPU:
python scripts/run.py --config configs/olmoe.yaml --policies gold uniform mixed \
    --bits 4 --routing-sequences 16 --max-ppl-windows 10 --results-dir results/smoke
```

## Step 3: analyse

```bash
python scripts/analyze.py --results-dir results/olmoe
```

Writes `figures.pdf` (perplexity, routing drift, top-1 flips, Jaccard distance, expert
entropy, output drift, and layer-wise KL, all with error bars) and `summary.md` containing
the results table, the decision gate, and the correctness-gate report.

The decision gate is the point. It prints, per bit-width, whether `mixed` beats `uniform`
by more than the bootstrap intervals:

```
- INT4 routing KL: uniform 0.000342 [0.000283, 0.000416] vs mixed 0.000063
  [0.000052, 0.000075] -> **MIXED WINS**
```

and then a verdict telling you which branch of the plan you are on.

## Step 4: follow the verdict

**If the verdict says at least one bit-width shows a real advantage**, run the
parameter-matched placebo control before believing it:

```bash
sbatch scripts/slurm/run_placebo.sh olmoe
# or: python scripts/run.py --config configs/olmoe.yaml --policies placebo --bits 4 3
```

This protects a random set of modules with the same parameter count as the routers. If it
helps as much as protecting the routers, the effect was about keeping *some* weights in
high precision and the router hypothesis is not supported.

**If the verdict says no separation**, that is the honest Part 1 answer. Proceed to Part 2
(see [OVERVIEW.md](OVERVIEW.md)); the router input activations it needs were already
captured during these runs.

## Output layout

```text
results/olmoe/
├── architecture.json          from inspect_model.py
├── gold/
│   ├── metrics.json
│   ├── artifacts.pt           router logits, weights, groups, token fingerprint
│   ├── output_reference.pt    gold top-M output distribution
│   └── router_inputs.pt       Part 2 only; the largest file here
├── uniform_int8/metrics.json
├── uniform_int4/metrics.json
├── mixed_int4/metrics.json
├── figures.pdf
└── summary.md
```

The gold artifacts are split by consumer on purpose. A candidate run reads only
`artifacts.pt` and `output_reference.pt`; `router_inputs.pt` is the biggest file and is
needed solely by the Part 2 offline attribution, so candidate runs never page it in.

Each `metrics.json` is self-describing: config, environment (torch/transformers/torchao
versions, GPU name and compute capability, git commit), the quantization audit, dataset
statistics, and all metrics with confidence intervals.

`.pt` artifacts are gitignored; `metrics.json` files are small and worth committing.

## Correctness gates

Runs abort rather than produce misleading numbers. Each check and what it catches:

| Gate | Catches |
|------|---------|
| gold self-comparison gives KL 0 and zero flips | nondeterminism anywhere upstream |
| `uniform` has **all** routers quantized | a silently skipped router making `uniform` == `mixed` |
| `mixed` has **no** router quantized | the skip rule not matching |
| something was quantized at all | `_default` matching nothing, e.g. fused expert stacks |
| distinct values per channel ≤ `2**bits` | a bit-width silently not applied |
| config expert count == router weight shape | registry pointing at the wrong module |

All of it lands in `summary.md` under "Correctness gates", ready for the paper appendix.

## Troubleshooting

**`No router modules matched '.*\.mlp\.gate'`** — the checkpoint's naming differs from the
registry. Run `python scripts/inspect_model.py <model> --show-tree` and update
`router_pattern` in `src/moequant/registry.py`.

**`uniform policy must quantize all N routers but only M are quantized`** — working as
intended. Something skipped the routers; check the transformers version and whether
`inspect_model.py` reports fused 3D experts.

**`nothing was quantized`** — the `_default` rule matched no modules. Almost always the
transformers 5.x fused-expert layout.

**`torch.int3 is unavailable`** — sub-byte dtypes need torch 2.6+. Upgrade, or drop 3-bit
from `--bits`.

**`Gold artifacts missing`** — run the gold policy for that model first.

**CUDA out of memory on Qwen gold** — request a larger card or `--gres=gpu:2`;
`device_map="auto"` will shard. Do not silently drop the model from the paper.

**Perplexity differs slightly between identical reruns** — should not happen. All batching
is seeded and quantization is deterministic. If it does, the gold self-check will catch it
first; investigate before trusting anything.
