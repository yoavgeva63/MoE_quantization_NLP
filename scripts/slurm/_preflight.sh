#!/bin/bash
# Shared setup, sourced by every Slurm job before any GPU work.
#
# The point is to fail in seconds with an actionable message rather than part-way into a
# multi-hour job. Everything checked here has actually bitten this project: a venv that
# was never created, a torch build without CUDA, and a torchao whose selective-quantization
# API had moved.

# Caches go to project storage, not the home quota, which is far too small for these
# checkpoints. Override MOEQUANT_STORAGE if your group directory differs.
STORAGE="${MOEQUANT_STORAGE:-/home/morg/NLP_2526b/${USER}}"
export HF_HOME="${STORAGE}/hf_cache"
export HF_DATASETS_CACHE="${STORAGE}/datasets_cache"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${STORAGE}/mpl_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}" logs

# Activate whichever environment exists. MOEQUANT_VENV wins; then a local .venv; then a
# PYTHONPATH-style .deps tree for hosts where `python -m venv` cannot bootstrap pip.
if [[ -n "${MOEQUANT_VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${MOEQUANT_VENV}/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -d .deps ]]; then
    export PYTHONPATH="${PWD}/.deps:${PWD}/src:${PYTHONPATH:-}"
else
    echo "ERROR: no Python environment found." >&2
    echo "  Create one with:  python -m venv .venv && source .venv/bin/activate \\" >&2
    echo "                    && pip install -e '.[dev]'" >&2
    echo "  Or set MOEQUANT_VENV to an existing environment." >&2
    exit 1
fi

# Resolve an interpreter explicitly. A venv provides `python`, but the .deps fallback
# runs against the system install, where only `python3` may exist.
if command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
else
    echo "ERROR: no python interpreter on PATH." >&2
    exit 1
fi
export PY_BIN

echo "Node:   $(hostname)"
echo "Python: $("${PY_BIN}" -V 2>&1) at ${PY_BIN}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# Every import the run needs, checked up front. A missing torchao here costs one second;
# discovered after the gold run it costs hours.
"${PY_BIN}" - <<'PY' || exit 1
import sys
missing = []
for mod in ("torch", "torchao", "transformers", "datasets", "accelerate", "yaml"):
    try:
        __import__(mod)
    except Exception as exc:
        missing.append(f"{mod} ({type(exc).__name__}: {exc})")
if missing:
    print("ERROR: missing dependencies:\n  - " + "\n  - ".join(missing), file=sys.stderr)
    print("Install with: pip install -e '.[dev]'", file=sys.stderr)
    sys.exit(1)

import torch, torchao, transformers
print(f"torch {torch.__version__} | torchao {torchao.__version__} "
      f"| transformers {transformers.__version__}")

if not torch.cuda.is_available():
    print("ERROR: torch reports no CUDA device. This is usually a CPU-only torch build; "
          "reinstall a CUDA wheel.", file=sys.stderr)
    sys.exit(1)

cap = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)} (compute {cap[0]}.{cap[1]})")
if cap[0] < 7:
    print(f"WARNING: compute capability {cap[0]}.{cap[1]} predates BF16 support; "
          "set dtype: float16 in the config or expect a failure.", file=sys.stderr)
PY

# The quantization backend, exercised for real on a synthetic model. This is the check
# that catches a torchao release moving the selective-quantization API, which would
# otherwise surface only after the model had loaded.
"${PY_BIN}" -m pytest tests/test_torchao_backend.py -q --no-header \
    || { echo "ERROR: torchao backend tests failed; not spending GPU time." >&2; exit 1; }
