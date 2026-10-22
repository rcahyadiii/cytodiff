# Sourced by setup.sh, run_all.sh and run_model.sh (from the project root).
# Keeps model weights and the pip and torch caches inside the project folder, so they land on the
# persistent volume (RunPod: /workspace) instead of the small container disk, which is wiped when a
# pod stops. Every script then also finds the same Hugging Face login ($HF_HOME/token).
CYTODIFF_CACHE="${CYTODIFF_CACHE:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$CYTODIFF_CACHE/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CYTODIFF_CACHE/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CYTODIFF_CACHE/pip}"
