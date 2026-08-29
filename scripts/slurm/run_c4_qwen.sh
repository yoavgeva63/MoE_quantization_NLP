#!/bin/bash
#SBATCH --job-name=moe-c4-qwen
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:5
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/c4_qwen_%j.out
#SBATCH --error=logs/c4_qwen_%j.err

# Second-corpus check on Qwen1.5-MoE-A2.7B: gold, uniform and mixed at INT4 only, on C4.
# See run_c4_olmoe.sh for why INT4 alone, and run_qwen.sh for the five-GPU shard.

set -euo pipefail

cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

export MOEQUANT_MIN_VRAM_GB=40

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

"${PY_BIN}" scripts/inspect_model.py qwen --out results/c4/qwen/architecture.json

"${PY_BIN}" scripts/run.py \
    --config configs/qwen_c4.yaml \
    --policies gold uniform mixed \
    --bits 4 \
    --keep-going \
    --skip-existing

# See run_c4_olmoe.sh: the offline analyses run here so the job either produces a complete
# second-corpus result or fails loudly.
"${PY_BIN}" scripts/analyze.py --results-dir results/c4/qwen
"${PY_BIN}" scripts/attribute.py --results-dir results/c4/qwen --bits 4
"${PY_BIN}" scripts/paired_bootstrap.py --results-dir results/c4/qwen --bits 4
"${PY_BIN}" scripts/collapse.py --results-dir results/c4/qwen
"${PY_BIN}" scripts/verify_offline.py --results-dir results/c4/qwen --model-key qwen
