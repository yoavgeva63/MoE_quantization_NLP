#!/bin/bash
#SBATCH --job-name=moe-c4-olmoe
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/c4_olmoe_%j.out
#SBATCH --error=logs/c4_olmoe_%j.err

# Second-corpus check on OLMoE: gold, uniform and mixed at INT4 only, on C4.
#
# INT4 is the bit-width where router protection helps most and where the attribution
# crossover sits, so it is the one cell worth repeating on a second corpus. Results land
# under results/c4/olmoe, which keeps the WikiText-2 tree untouched.
#
# Hardware matches run_olmoe.sh: ~14GB in BF16 sharded over three RTX 2080 Ti, with the
# GPU type pinned because the TITAN Xp cards on this partition are sm_61.

set -euo pipefail

cd "${MOEQUANT_REPO:-${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}}"
[[ -f pyproject.toml ]] || { echo "ERROR: $PWD is not the repo root; submit from there or set MOEQUANT_REPO." >&2; exit 1; }

export MOEQUANT_MIN_VRAM_GB=20

# shellcheck disable=SC1091
source scripts/slurm/_preflight.sh

"${PY_BIN}" scripts/inspect_model.py olmoe --out results/c4/olmoe/architecture.json

# See run_olmoe.sh: this partition preempts without warning and restarts from the top.
"${PY_BIN}" scripts/run.py \
    --config configs/olmoe_c4.yaml \
    --policies gold uniform mixed \
    --bits 4 \
    --keep-going \
    --skip-existing

# The offline analyses need no GPU, but running them here means the job either produces a
# complete second-corpus result or fails loudly, rather than leaving artifacts to process
# by hand later.
"${PY_BIN}" scripts/analyze.py --results-dir results/c4/olmoe
"${PY_BIN}" scripts/attribute.py --results-dir results/c4/olmoe --bits 4
"${PY_BIN}" scripts/paired_bootstrap.py --results-dir results/c4/olmoe --bits 4
"${PY_BIN}" scripts/collapse.py --results-dir results/c4/olmoe
"${PY_BIN}" scripts/verify_offline.py --results-dir results/c4/olmoe --model-key olmoe
