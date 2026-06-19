#!/usr/bin/env bash
# Build a second venv that can run the torch-family tests alongside the main env.
#
# WHY THIS EXISTS
#   The main dev env runs Python 3.14, but PyTorch publishes no 3.14 wheels, and
#   on Intel macOS (x86_64) PyTorch dropped x86_64 wheels after 2.2.2 (which only
#   has cp38–cp312 wheels). So the ~22 torch-gated tests (tests/ml/test_l5_advanced.py,
#   tests/ml/test_late_chunking.py, tests/test_finetune_llama.py,
#   tests/worm/test_worm_embedding.py) cannot run on 3.14 and skip there.
#
#   This script creates .venv-torch on Python 3.12 with torch 2.2.2 (the last
#   Intel-mac wheel) + numpy<2 (torch 2.2.2 is built against NumPy 1.x) + the ML
#   extras, so those tests run and pass. Keep using the 3.14 .venv for everything
#   else — this is purely a supplementary "torch matrix" env.
#
#   On Apple Silicon / Linux you can instead use a newer Python + newer torch;
#   the pins below are the Intel-mac lowest-common-denominator.
#
# USAGE
#   bash scripts/setup_torch_test_env.sh          # build the env
#   bash scripts/setup_torch_test_env.sh --run    # build, then run the torch tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-torch"
PY312="$(command -v python3.12 || true)"

if [[ -z "${PY312}" ]]; then
  echo "ERROR: python3.12 not found. Install it (e.g. 'brew install python@3.12')." >&2
  echo "       (Python 3.12 is required: torch 2.2.2 — the last Intel-mac wheel — is cp312.)" >&2
  exit 1
fi

echo ">> Creating venv at ${VENV_DIR} (Python: $(${PY312} --version 2>&1))"
"${PY312}" -m venv "${VENV_DIR}"
VPY="${VENV_DIR}/bin/python"

"${VPY}" -m pip install --quiet --upgrade pip

echo ">> Installing torch 2.2.2 + numpy<2 (NumPy-1.x ABI required by torch 2.2.2)"
"${VPY}" -m pip install 'torch==2.2.2' 'numpy<2'

echo ">> Installing na0s (editable) + ML test extras"
"${VPY}" -m pip install -e "${REPO_ROOT}" \
  'sentence-transformers>=2.2,<4' 'peft' 'pandas' 'pytest'

# Re-pin numpy<2 in case a dep pulled NumPy 2.x back in.
"${VPY}" -m pip install 'numpy<2' >/dev/null

echo ">> Sanity check:"
"${VPY}" - <<'PY'
import numpy, torch
print(f"   numpy {numpy.__version__} | torch {torch.__version__}")
assert numpy.__version__.startswith("1."), "torch 2.2.2 needs numpy<2"
torch.tensor(numpy.float32(1.5))  # raises if the ABI mismatch is back
print("   torch<->numpy OK")
PY

TORCH_TESTS=(
  tests/ml/test_l5_advanced.py
  tests/ml/test_late_chunking.py
  tests/test_finetune_llama.py
  tests/worm/test_worm_embedding.py
)

echo
echo "Done. Run the torch-family tests with:"
echo "  PYTHONPATH=src ${VENV_DIR}/bin/python -m pytest ${TORCH_TESTS[*]} -q"

if [[ "${1:-}" == "--run" ]]; then
  echo
  echo ">> Running torch-family tests..."
  cd "${REPO_ROOT}"
  PYTHONPATH=src "${VPY}" -m pytest "${TORCH_TESTS[@]}" -q
fi
