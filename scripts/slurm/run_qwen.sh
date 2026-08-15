#!/bin/bash
#SBATCH --job-name=moe-qwen
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err

# Part 1 sweep on Qwen1.5-MoE-A2.7B.
#
# ~28GB in BF16 for the gold run, sharded over five RTX 2080 Ti (11GB each) by
# device_map="auto". Five rather than three leaves room for activations and the router
# capture. See run_olmoe.sh for why the GPU type is pinned.

set -euo pipefail

# Slurm runs a copy of this file out of its spool directory, so "$0" says nothing about
# where the repo is. Prefer the submission directory, and fall back to "$0" for the case
# where the script is executed directly rather than submitted.
cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

export MOEQUANT_MIN_VRAM_GB=40

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

"${PY_BIN}" scripts/inspect_model.py qwen --out results/qwen/architecture.json

# See run_olmoe.sh: this partition preempts without warning and restarts from the top.
"${PY_BIN}" scripts/run.py \
    --config configs/qwen.yaml \
    --policies gold uniform mixed \
    --bits 8 4 3 \
    --keep-going \
    --skip-existing

"${PY_BIN}" scripts/analyze.py --results-dir results/qwen
