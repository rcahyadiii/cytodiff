#!/usr/bin/env bash
# Full stage-1d..stage-5 run for one model on a cloud GPU: cache -> SMOKE GATE -> train -> sweep -> eval.
# The 5-step smoke gate catches config/shape/OOM errors for ~1 min of compute before the ~hours-long
# full run, so a bug never wastes a full rental.
#
# Usage:  bash scripts/cloud/run_model.sh sdxl        # or sd3
#         GEN_BATCH=2 bash scripts/cloud/run_model.sh sd3   # smaller gen batch if OOM
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh
MODEL="${1:?usage: run_model.sh <sdxl|sd3|sd15|sd21>}"
CONFIG="configs/${MODEL}.yaml"
[ -f "$CONFIG" ] || { echo "no $CONFIG"; exit 1; }
VENV=venv/bin/python
GEN_BATCH="${GEN_BATCH:-4}"
# Overrides used by run_all.sh SMOKE=1 (defaults = the real run):
NUM_IMAGES="${NUM_IMAGES:-200}"   # images per class per CFG
MAX_STEPS="${MAX_STEPS:-}"        # empty = the config's 10,000 steps
CFG_LIST="${CFG_LIST:-}"          # empty = the full CFG grid
N_BOOT="${N_BOOT:-1000}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_DIR"

echo "===== $MODEL : data prep (patient-level manifest, deterministic) ====="
# Always rebuild: a stale image-level manifest from an earlier sync must never be reused.
if ls data/processed/ADC/*.png >/dev/null 2>&1; then
  $VENV scripts/prepare_data.py --skip-preprocess
else
  $VENV scripts/prepare_data.py
fi

echo "===== $MODEL : stage 1d cache (this is the T5-heavy step for sd3) ====="
$VENV scripts/train.py --config "$CONFIG" --cache-only

echo "===== $MODEL : SMOKE GATE (5 steps) ====="
# The smoke run writes to its own folder, so its 5-step adapter can never be mistaken for a
# finished training run when an interrupted run is resumed.
if ! $VENV scripts/train.py --config "$CONFIG" --skip-cache --max-steps 5 --save-every 100000 --run-name "${MODEL}_smoke" 2>&1 | tee $LOG_DIR/smoke_${MODEL}.log | grep -q "saved LoRA adapter"; then
  echo "!! SMOKE GATE FAILED for $MODEL — inspect $LOG_DIR/smoke_${MODEL}.log; NOT starting full run."
  exit 2
fi
# Guard: smoke must have produced >0 trainable params (catches wrong LoRA target_modules).
grep -qE "trainable [1-9]" $LOG_DIR/smoke_${MODEL}.log || { echo "!! 0 trainable LoRA params — fix target_modules in $CONFIG"; exit 3; }
# Guard: NaN/Inf loss (fp16 overflow on big denoisers — switch that config to mixed_precision: bf16).
if grep -qiE "loss (nan|inf)" $LOG_DIR/smoke_${MODEL}.log; then
  echo "!! SMOKE GATE FAILED: NaN/Inf loss for $MODEL — set mixed_precision: bf16 in $CONFIG"; exit 4
fi
echo "  smoke gate PASSED"
rm -rf "${CYTODIFF_EXPERIMENTS_DIR:-experiments}/checkpoints/${MODEL}_smoke"

echo "===== $MODEL : full training ====="
$VENV -u scripts/train.py --config "$CONFIG" --skip-cache --save-every 2000 ${MAX_STEPS:+--max-steps "$MAX_STEPS"} 2>&1 | tee $LOG_DIR/train_${MODEL}.log

echo "===== $MODEL : CFG sweep (batch=$GEN_BATCH) ====="
$VENV -u scripts/generate.py --config "$CONFIG" --num-images "$NUM_IMAGES" --batch "$GEN_BATCH" --seed 0 ${CFG_LIST:+--cfg $CFG_LIST} 2>&1 | tee $LOG_DIR/sweep_${MODEL}.log

echo "===== $MODEL : KID/FID (per class, patient-bootstrap CIs) ====="
$VENV -u scripts/evaluate.py --model "$MODEL" --n-boot "$N_BOOT" ${CFG_LIST:+--cfg $CFG_LIST} 2>&1 | tee $LOG_DIR/eval_${MODEL}.log
grep -E "MACRO|best CFG" $LOG_DIR/eval_${MODEL}.log || true
echo "===== $MODEL DONE. Copy back: experiments/results/metrics_${MODEL}.csv, experiments/features/, experiments/checkpoints/${MODEL}/final/, experiments/samples/${MODEL}/ ====="
