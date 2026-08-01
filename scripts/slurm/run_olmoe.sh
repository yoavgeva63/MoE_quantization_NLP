#!/bin/bash
#SBATCH --job-name=moe-olmoe
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/olmoe_%j.out
#SBATCH --error=logs/olmoe_%j.err

# Part 1 sweep on OLMoE-1B-7B: gold, then uniform and mixed at INT8/4/3.
# ~14GB in BF16, so a single 24GB card is enough.

set -euo pipefail
cd "$(dirname "$0")/../.."

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
