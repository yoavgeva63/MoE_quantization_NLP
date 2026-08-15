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
# These runs alternate between big transient logits and long-lived quantized weights,
# which fragments the caching allocator badly on 11GB cards. Expandable segments let the
# allocator grow a block instead of failing next to free-but-unusable memory.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}" logs

# Activate whichever environment exists. MOEQUANT_VENV wins; then a local .venv; then the
# venv in storage, which is where it lives on this cluster because the repo sits on a
# shared filesystem; then a PYTHONPATH-style .deps tree for hosts where `python -m venv`
# cannot bootstrap pip. The storage fallback matters: a job must not depend on the
# submitting shell having exported anything.
if [[ -n "${MOEQUANT_VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${MOEQUANT_VENV}/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -f "${STORAGE}/venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${STORAGE}/venv/bin/activate"
elif [[ -d .deps ]]; then
    export PYTHONPATH="${PWD}/.deps:${PWD}/src:${PYTHONPATH:-}"
else
    echo "ERROR: no Python environment found." >&2
    echo "  Looked for: \$MOEQUANT_VENV, ./.venv, ${STORAGE}/venv, ./.deps" >&2
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

import os

# The wheel only ships kernels for the architectures in get_arch_list(); anything older
# loads and reports cuda.is_available() == True, then fails on the first kernel launch.
# torch.cuda.is_bf16_supported() is not a usable guard here — it returns True on sm_61.
arches = sorted(int(a[3:]) for a in torch.cuda.get_arch_list()
                if a.startswith("sm_") and a[3:].isdigit())
min_arch = min(arches) if arches else 0

unusable, total_gb = [], 0.0
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    sm, gb = p.major * 10 + p.minor, p.total_memory / 1024**3
    total_gb += gb
    ok = sm >= min_arch
    print(f"  cuda:{i} {p.name} sm_{sm} {gb:.1f} GB {'ok' if ok else 'UNUSABLE'}")
    if not ok:
        unusable.append(f"cuda:{i} {p.name} (sm_{sm})")
print(f"{torch.cuda.device_count()} GPU(s), {total_gb:.1f} GB total VRAM")

if unusable:
    print(f"ERROR: this torch build needs sm_{min_arch}+; got:\n  - "
          + "\n  - ".join(unusable), file=sys.stderr)
    print("On studentkillable request the RTX 2080 Ti explicitly: "
          "--gres=gpu:geforce_rtx_2080:N (the TITAN Xp cards are sm_61).", file=sys.stderr)
    sys.exit(1)

need = float(os.environ.get("MOEQUANT_MIN_VRAM_GB", "0"))
if total_gb < need:
    print(f"ERROR: need ~{need:.0f} GB total VRAM, allocated {total_gb:.1f} GB. "
          "Raise the GPU count in --gres.", file=sys.stderr)
    sys.exit(1)
PY

# The quantization backend, exercised for real on a synthetic model. This is the check
# that catches a torchao release moving the selective-quantization API, which would
# otherwise surface only after the model had loaded.
"${PY_BIN}" -m pytest tests/test_torchao_backend.py -q --no-header \
    || { echo "ERROR: torchao backend tests failed; not spending GPU time." >&2; exit 1; }
