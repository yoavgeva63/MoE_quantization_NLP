#!/bin/bash
#SBATCH --job-name=moe-smoke
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# End-to-end check on real weights before committing to a full sweep: OLMoE at INT4 only,
# on a small routing subset and a capped number of perplexity windows. Exercises the same
# code path as run_olmoe.sh — load, quantize, capture, score, correctness gates — in
# minutes instead of hours.
#
# Results go to a throwaway directory so a later real sweep does not read these as gold.

set -euo pipefail

# Slurm runs a copy of this file out of its spool directory, so "$0" says nothing about
# where the repo is. Prefer the submission directory, and fall back to "$0" for the case
# where the script is executed directly rather than submitted.
cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

#   sbatch scripts/slurm/run_smoke.sh olmoe        # INT4 only, the quickest check
#   sbatch scripts/slurm/run_smoke.sh olmoe 8 4 3  # prove every bit-width the sweep uses
MODEL="${1:-olmoe}"
shift || true
BITS=("$@")
[[ ${#BITS[@]} -eq 0 ]] && BITS=(4)

export MOEQUANT_MIN_VRAM_GB=20

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

"${PY_BIN}" scripts/inspect_model.py "${MODEL}" \
    --out "results/smoke/${MODEL}/architecture.json"

"${PY_BIN}" scripts/run.py \
    --config "configs/${MODEL}.yaml" \
    --policies gold uniform mixed \
    --bits "${BITS[@]}" \
    --routing-sequences 16 \
    --max-ppl-windows 10 \
    --results-dir results/smoke \
    --keep-going

echo "Smoke test finished. Inspect results/smoke/${MODEL}/ then delete it before the real sweep."
