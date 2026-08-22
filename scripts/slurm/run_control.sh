#!/bin/bash
#SBATCH --job-name=moe-control
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/control_%j.out
#SBATCH --error=logs/control_%j.err

# run_placebo.sh generalized to an arbitrary control policy.
#
# The parameter-matched placebo controls for "any 0.02% of weights in BF16 helps". It
# cannot control for "any high-precision island on the every-token, every-layer main path
# helps", because it protects one expert projection in one layer, which most tokens never
# route through. The `attention` policy is the control that closes that gap: attention is
# read by every token in every layer, exactly like the routers, so if protecting it does
# not reduce routing drift the way protecting the routers does, the effect is specific to
# the routers rather than to high-precision weights on the main path.
#
# Requires the gold run for this model to already exist.

set -euo pipefail

# Slurm runs a copy of this file out of its spool directory, so "$0" says nothing about
# where the repo is. Prefer the submission directory, and fall back to "$0" for the case
# where the script is executed directly rather than submitted.
cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

# The policy is required; the model and the bit-widths follow run_placebo.sh's defaults:
#
#   sbatch scripts/slurm/run_control.sh attention olmoe 4 3
#
# Qwen needs the same allocation as its own sweep. The #SBATCH directives above are read
# before this script runs, so they cannot depend on "$1"; pass the override on the command
# line, where it takes precedence:
#
#   sbatch --gres=gpu:geforce_rtx_2080:5 --mem=96G scripts/slurm/run_control.sh attention qwen 4 3
#
# Forgetting the override is not silent: the VRAM floor below makes the preflight refuse
# the job up front rather than let it OOM partway through the run.
if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") POLICY [MODEL] [BITS...]" >&2
    echo "  e.g. $(basename "$0") attention olmoe 4 3" >&2
    exit 1
fi
POLICY="$1"
MODEL="${2:-olmoe}"
if [[ $# -gt 2 ]]; then
    BITS=("${@:3}")
else
    BITS=(4 3)
fi

# `placebo` needs a module list sampled inside run.py, which this script does not plumb,
# but it needs nothing extra from here either — run_placebo.sh exists for it and is the
# documented entry point. Everything else is validated by run.py's --policies choices.
if [[ "${POLICY}" == "gold" ]]; then
    echo "ERROR: gold is not a control; use the model's own sweep script." >&2
    exit 1
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

echo "Control policy '${POLICY}' on ${MODEL} at bit-widths: ${BITS[*]}"

# studentkillable preempts without warning and restarts from the top, so --skip-existing
# is what makes a requeue resume rather than repeat.
"${PY_BIN}" scripts/run.py \
    --config "configs/${MODEL}.yaml" \
    --policies "${POLICY}" \
    --bits "${BITS[@]}" \
    --keep-going \
    --skip-existing

"${PY_BIN}" scripts/analyze.py --results-dir "results/${MODEL}"
