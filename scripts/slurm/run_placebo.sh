#!/bin/bash
#SBATCH --job-name=moe-placebo
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
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

#   sbatch scripts/slurm/run_placebo.sh olmoe
#
# Qwen needs the same allocation as its own sweep. The #SBATCH directives above are read
# before this script runs, so they cannot depend on "$1"; pass the override on the command
# line, where it takes precedence:
#
#   sbatch --gres=gpu:geforce_rtx_2080:5 --mem=96G scripts/slurm/run_placebo.sh qwen
#
# Forgetting the override is not silent: the VRAM floor below makes the preflight refuse
# the job up front rather than let it OOM partway through the run.
#
# Any further arguments are the bit-widths to control, so the sweep can be extended after
# the fact without re-running what is already on disk:
#
#   sbatch scripts/slurm/run_placebo.sh olmoe 8
MODEL="${1:-olmoe}"
if [[ $# -gt 1 ]]; then
    BITS=("${@:2}")
else
    BITS=(4 3)
fi

# Match each model's real sweep: run_olmoe.sh asks for 20, run_qwen.sh for 40.
case "${MODEL}" in
    qwen) export MOEQUANT_MIN_VRAM_GB=40 ;;
    *) export MOEQUANT_MIN_VRAM_GB=20 ;;
esac

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

if [[ ! -f "results/${MODEL}/gold/artifacts.pt" ]]; then
    echo "ERROR: no gold artifacts for ${MODEL}; run the main sweep first." >&2
    exit 1
fi

"${PY_BIN}" scripts/run.py \
    --config "configs/${MODEL}.yaml" \
    --policies placebo \
    --bits "${BITS[@]}" \
    --keep-going \
    --skip-existing

"${PY_BIN}" scripts/analyze.py --results-dir "results/${MODEL}"
