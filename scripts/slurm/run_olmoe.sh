#!/bin/bash
#SBATCH --job-name=moe-olmoe
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/olmoe_%j.out
#SBATCH --error=logs/olmoe_%j.err

# Part 1 sweep on OLMoE-1B-7B: gold, then uniform and mixed at INT8/4/3.
#
# ~14GB in BF16. studentkillable has no card that large, so this shards over three
# RTX 2080 Ti (11GB each) via device_map="auto". The GPU type is pinned because the
# TITAN Xp cards on this partition are sm_61 and the torch wheel has no kernels for them.

set -euo pipefail

# Slurm runs a copy of this file out of its spool directory, so "$0" says nothing about
# where the repo is. Prefer the submission directory, and fall back to "$0" for the case
# where the script is executed directly rather than submitted.
cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

export MOEQUANT_MIN_VRAM_GB=20

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

# Confirm the registry still matches the checkpoint before spending GPU hours.
"${PY_BIN}" scripts/inspect_model.py olmoe --out results/olmoe/architecture.json

"${PY_BIN}" scripts/run.py \
    --config configs/olmoe.yaml \
    --policies gold uniform mixed \
    --bits 8 4 3 \
    --keep-going

"${PY_BIN}" scripts/analyze.py --results-dir results/olmoe
