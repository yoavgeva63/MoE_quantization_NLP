#!/bin/bash
#SBATCH --job-name=moe-placebo
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/placebo_%j.out
#SBATCH --error=logs/placebo_%j.err

# The control that makes a positive Part 1 result interpretable.
#
# Only run this if the decision gate in results/<model>/summary.md says MIXED WINS.
# It protects a random set of modules with the same parameter count as the routers.
# If that helps as much as protecting the routers, then the effect was about keeping
# *some* weights in high precision, and the router hypothesis is not supported.
#
# Requires the gold run for this model to already exist.

set -euo pipefail

# Slurm runs a copy of this file out of its spool directory, so "$0" says nothing about
# where the repo is. Prefer the submission directory, and fall back to "$0" for the case
# where the script is executed directly rather than submitted.
cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

MODEL="${1:-olmoe}"

export MOEQUANT_MIN_VRAM_GB=20

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

if [[ ! -f "results/${MODEL}/gold/artifacts.pt" ]]; then
    echo "ERROR: no gold artifacts for ${MODEL}; run the main sweep first." >&2
    exit 1
fi

"${PY_BIN}" scripts/run.py \
    --config "configs/${MODEL}.yaml" \
    --policies placebo \
    --bits 4 3 \
    --keep-going

"${PY_BIN}" scripts/analyze.py --results-dir "results/${MODEL}"
