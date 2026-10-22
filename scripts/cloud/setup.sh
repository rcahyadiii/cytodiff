#!/usr/bin/env bash
# One-time setup on a fresh 48 GB cloud GPU box (RunPod / Vast / Lambda).
# Assumes: Ubuntu-like, Python 3.11+, this project's code copied to $PWD (on the persistent volume,
# e.g. /workspace/cytodiff on RunPod), an NVIDIA GPU present.
# Data is fetched by run_all.sh (scripts/fetch_lunghist700.sh), so it does not need to be copied.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh

echo "== [1/5] GPU, Python and disk check =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || { echo "no GPU visible"; exit 1; }
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "Python 3.11+ is needed (scikit-image 0.26); found $(python3 -V)"; exit 1; }
FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
[ "$FREE_GB" -ge 150 ] || echo "  WARNING: ${FREE_GB} GB free in $PWD; the full run needs about 150 GB."
echo "  caches: HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME"

echo "== [2/5] system tools (tmux, bsdtar) =="
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  missing=()
  command -v tmux >/dev/null || missing+=(tmux)
  command -v bsdtar >/dev/null || missing+=(libarchive-tools)
  if [ ${#missing[@]} -gt 0 ]; then apt-get update -qq && apt-get install -y -qq "${missing[@]}"; fi
fi
command -v tmux >/dev/null || echo "  tmux not found: install it, or use screen / nohup for the long run."

echo "== [3/5] venv + deps =="
if ! python3 -m venv venv; then
  # Some images ship Python without the venv module.
  PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
  apt-get update -qq && apt-get install -y -qq "python${PYV}-venv"
  python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
# PyTorch CUDA build: cu130 needs a CUDA 13 driver; cu126 runs on any CUDA 12 driver but has no
# Blackwell kernels, so Blackwell GPUs on a CUDA 12 driver get cu129. TORCH_INDEX overrides.
DRV_CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+$' || echo 12)
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV_CUDA" -ge 13 ]; then CU=cu130; elif [ "$CC_MAJOR" -ge 10 ]; then CU=cu129; else CU=cu126; fi
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/$CU}"
echo "  driver CUDA $DRV_CUDA, compute capability $CC_MAJOR.x -> $TORCH_INDEX"
pip install torch torchvision --index-url "$TORCH_INDEX"
pip install -r requirements.txt
pip install -e .

echo "== [4/5] Hugging Face auth (SD 3 and UNI are gated) =="
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$HF_HOME/token" ]; then
    echo "  using the saved login in $HF_HOME/token"
  else
    echo "  Set HF_TOKEN (read token from hf.co/settings/tokens) and re-run this script."
  fi
else
  python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
  echo "  logged in (token saved in $HF_HOME/token)."
fi

echo "== [5/5] sanity =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch does not see the GPU"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()   # fails here if this torch build has no kernels for the GPU
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB | bf16 ok")
PY
echo "setup done. Next, inside tmux:  SMOKE=1 bash scripts/cloud/run_all.sh   then   bash scripts/cloud/run_all.sh"
