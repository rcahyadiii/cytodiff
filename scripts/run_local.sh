#!/usr/bin/env bash
# Local full chain for one model (train 10k -> CFG sweep -> FID/KID). SD 1.5 / 2.1 fit the 8 GB RTX 2080;
# SDXL / SD 3 need scripts/cloud/run_model.sh on a 24 GB+ GPU.
# Usage:  bash scripts/run_local.sh sd15          (GEN_BATCH=4 for sd21: batch 8 OOMs on 8 GB)
set -u
MODEL=${1:?usage: run_local.sh <sd15|sd21>}
GEN_BATCH=${GEN_BATCH:-8}
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=venv/bin/python
mkdir -p outputs/logs

echo "[$MODEL] manifest (patient-level, deterministic) $(date '+%T')"
$V scripts/prepare_data.py --skip-preprocess > outputs/logs/prepare_data.log 2>&1 || { echo "manifest failed; see outputs/logs/prepare_data.log"; exit 1; }
echo "[$MODEL] train (cache+10k) $(date '+%T')"
$V -u scripts/train.py --config "configs/$MODEL.yaml" --save-every 5000 > "outputs/logs/train_$MODEL.log" 2>&1
echo "[$MODEL] train rc=$? $(date '+%T')"
echo "[$MODEL] sweep (batch $GEN_BATCH) $(date '+%T')"
$V -u scripts/generate.py --config "configs/$MODEL.yaml" --num-images 200 --batch "$GEN_BATCH" --seed 0 > "outputs/logs/sweep_$MODEL.log" 2>&1
echo "[$MODEL] sweep rc=$? images=$(find "experiments/samples/$MODEL" -name '*.png' 2>/dev/null | wc -l) $(date '+%T')"
echo "[$MODEL] eval $(date '+%T')"
$V -u scripts/evaluate.py --model "$MODEL" > "outputs/logs/eval_$MODEL.log" 2>&1
echo "[$MODEL] eval rc=$? $(date '+%T')"
grep -E "MACRO|best CFG" "outputs/logs/eval_$MODEL.log"
echo "[$MODEL] COMPLETE $(date '+%T')"
