#!/bin/bash
#SBATCH --job-name=moe-qwen
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err

# Part 1 sweep on Qwen1.5-MoE-A2.7B.
# ~28GB in BF16 for the gold run: request a 40GB+ card, or --gres=gpu:2 and let
# device_map="auto" shard it.

set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

"${PY_BIN}" scripts/inspect_model.py qwen --out results/qwen/architecture.json

"${PY_BIN}" scripts/run.py \
    --config configs/qwen.yaml \
    --policies gold uniform mixed \
    --bits 8 4 3 \
    --keep-going

"${PY_BIN}" scripts/analyze.py --results-dir results/qwen
