#!/usr/bin/env bash
# Reproducible environment setup for CytoDiff.
# Creates venv/, installs PyTorch (CUDA 13.0 wheels), the pinned stack, and the package (editable).
# Verified on: RTX 2080 (Turing, cc 7.5), driver 580, CUDA 13.0, Python 3.12.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo ">> Creating venv/"
"$PYTHON" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate

echo ">> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo ">> Installing PyTorch (cu130)"
pip install torch torchvision --index-url "$TORCH_INDEX"

echo ">> Installing pinned dependencies"
pip install -r requirements.txt

echo ">> Installing cytodiff (editable)"
pip install -e .

echo ">> GPU sanity check"
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo ">> Done. Activate with: source venv/bin/activate"
