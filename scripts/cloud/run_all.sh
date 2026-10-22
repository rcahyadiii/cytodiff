#!/usr/bin/env bash
# One-command v2 run on a single cloud GPU (24 GB+, 48 GB recommended):
#   access checks -> data -> patient-level split -> zero-shot baselines FIRST (each base model
#   without LoRA: 9-scale CFG sweep, evaluation) -> the same four models with LoRA at native
#   resolution (train seed 0, sweep, evaluation) -> extra training seeds for the two best models
#   -> color transfer -> real-vs-real baseline -> paired comparison incl. LoRA vs zero-shot,
#   color statistics, memorization, label check, VAE reconstruction -> results archive.
#
# Resumable: re-running skips finished stages (trained adapter, complete sweep, metrics file).
# Records library versions (environment.txt) and wall time per stage (timings.csv) in the log dir.
#
# SMOKE=1 runs every stage at toy size (20 training steps, 8 images per class, CFG 3 and 5, one
# extra seed, 50 bootstrap draws) into experiments_smoke/ and outputs/logs_smoke/, so the whole
# pipeline is exercised on the GPU in a short session before the real run. Model downloads and
# latent caches are shared with the real run, so nothing is wasted.
#
# Usage (after scripts/cloud/setup.sh, with HF_TOKEN set and the SD 3 and UNI licenses accepted):
#   SMOKE=1 bash scripts/cloud/run_all.sh                  # first: quick end-to-end check
#   bash scripts/cloud/run_all.sh                          # then: the real run
#   MODELS="sd15 sd21" SEEDS="1" bash scripts/cloud/run_all.sh
#   ZS_CFGS="3 5 7.5 10" bash scripts/cloud/run_all.sh    # narrower zero-shot grid (default: full)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh
VENV=venv/bin/python
MODELS="${MODELS:-sd15 sd21 sdxl sd3}"
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  EXP=experiments_smoke; LOGS=outputs/logs_smoke
  SEEDS="${SEEDS:-1}"; GRID="3 5"; ZS_CFGS="3 5"; NUM=8; MAX_STEPS=20; NBOOT=50; VAE_LIMIT="--limit 8"
else
  EXP=experiments; LOGS=outputs/logs
  SEEDS="${SEEDS:-1 2}"; GRID=""; ZS_CFGS="${ZS_CFGS:-}"; NUM=200; MAX_STEPS=""; NBOOT=1000; VAE_LIMIT=""
fi
export CYTODIFF_EXPERIMENTS_DIR="$PWD/$EXP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ZS_RUNS=$(for m in $MODELS; do printf "%s_zeroshot " "$m"; done)
mkdir -p "$LOGS"
TIMINGS=$LOGS/timings.csv
[ -f "$TIMINGS" ] || echo "stage,start_utc,end_utc,seconds" > "$TIMINGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run_all.log"; }
timed() {  # timed <stage-name> <command...>
  local name=$1; shift
  local t0; t0=$(date -u +%s); log "== $name =="
  "$@"
  local t1; t1=$(date -u +%s); echo "$name,$t0,$t1,$((t1 - t0))" >> "$TIMINGS"
}
gen_batch() { case $1 in sd15|sd21) echo 8 ;; sdxl) echo 4 ;; sd3) echo 2 ;; *) echo 4 ;; esac; }
n_png() { find "$1" -name '*.png' 2>/dev/null | wc -l | tr -d ' '; }
has_metrics() { [ -f "$EXP/results/metrics_$1.csv" ] && grep -q ",macro," "$EXP/results/metrics_$1.csv"; }
adapter() { echo "$EXP/checkpoints/$1/final/pytorch_lora_weights.safetensors"; }
n_grid() { if [ -n "$1" ]; then echo "$1" | wc -w | tr -d ' '; else $VENV -c "from cytodiff.constants import CFG_SWEEP; print(len(CFG_SWEEP))"; fi; }
grid_args=(); if [ -n "$GRID" ]; then read -r -a grid_args <<< "--cfg $GRID"; fi
zs_args=(); if [ -n "$ZS_CFGS" ]; then read -r -a zs_args <<< "--cfg $ZS_CFGS"; fi
SWEEP_N=$(( $(n_grid "$GRID") * NUM * 2 ))
ZS_SWEEP_N=$(( $(n_grid "$ZS_CFGS") * NUM * 2 ))
log "mode: $([ "$SMOKE" = 1 ] && echo SMOKE || echo FULL); outputs in $EXP, logs in $LOGS"

# ---------------------------------------------------------------- environment and access
{
  date -u; nvidia-smi; $VENV -V
  echo "code sha256: $(find src scripts configs -type f -not -path '*/__pycache__/*' | sort | xargs sha256sum | sha256sum | cut -c1-16)"
  $VENV -m pip freeze
} > "$LOGS/environment.txt" 2>&1
log "== access checks (gated models: SD 3, UNI) =="
$VENV - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for repo in ("stabilityai/stable-diffusion-3-medium-diffusers", "MahmoodLab/UNI", "owkin/phikon",
             "stable-diffusion-v1-5/stable-diffusion-v1-5", "SfinOe/stable-diffusion-v2-1",
             "stabilityai/stable-diffusion-xl-base-1.0"):
    try:
        api.auth_check(repo)
    except Exception as e:
        raise SystemExit(f"!! no access to {repo}: {type(e).__name__}. Set HF_TOKEN and accept the "
                         f"model's license on huggingface.co, then re-run.")
    print(f"  access ok: {repo}")
PY

# ---------------------------------------------------------------- data and split
stage_data() {
  bash scripts/fetch_lunghist700.sh
  if [ "$(n_png data/processed/ADC)" = 280 ] && [ "$(n_png data/processed/SCC)" = 260 ]; then
    $VENV scripts/prepare_data.py --skip-preprocess
  else
    $VENV scripts/prepare_data.py
  fi
  $VENV - <<'PY'
import pandas as pd
d = pd.read_csv("data/splits/manifest.csv")
got = {f"{c} {s}": int(n) for (c, s), n in d.groupby(["class", "split"]).size().items()}
want = {"ADC eval": 83, "ADC train": 197, "SCC eval": 77, "SCC train": 183}
assert got == want, f"unexpected split {got}, expected {want}"
assert (d.groupby("patient_id")["split"].nunique() == 1).all(), "a patient is on both sides"
print("  patient-level split verified:", got)
PY
}
timed data stage_data

# Keep earlier (v1) outputs apart: training and generation overwrite files in place.
if [ "$EXP" = experiments ] && [ -d experiments ] && [ ! -f experiments/.v2 ]; then
  old="experiments_v1_$(date +%Y%m%d_%H%M%S)"; mv experiments "$old"; log "moved old experiments to $old"
fi
mkdir -p "$EXP/results" && touch "$EXP/.v2"

# ---------------------------------------------------------------- zero-shot baselines first
# Each base model without LoRA, same captions, resolution and CFG grid: the reference that the
# LoRA runs below are compared against (compare_models.py -> lora_vs_zeroshot.csv).
for m in $MODELS; do
  run="${m}_zeroshot"
  if has_metrics "$run"; then log "$run: finished earlier, skipping"; continue; fi
  if [ "$(n_png "$EXP/samples/$run")" != "$ZS_SWEEP_N" ]; then
    timed "sweep_$run" $VENV -u scripts/generate.py --config "configs/$m.yaml" --zero-shot --run-name "$run" \
      --num-images "$NUM" --batch "$(gen_batch "$m")" --seed 0 ${zs_args[@]+"${zs_args[@]}"}
  fi
  timed "eval_$run" $VENV -u scripts/evaluate.py --model "$run" --n-boot "$NBOOT" ${zs_args[@]+"${zs_args[@]}"}
done

# ---------------------------------------------------------------- four models with LoRA, seed 0
for m in $MODELS; do
  gb=$(gen_batch "$m")
  if has_metrics "$m"; then log "$m: finished earlier, skipping"; continue; fi
  if [ ! -f "$(adapter "$m")" ]; then
    timed "model_$m" env GEN_BATCH="$gb" NUM_IMAGES="$NUM" MAX_STEPS="$MAX_STEPS" CFG_LIST="$GRID" \
      N_BOOT="$NBOOT" LOG_DIR="$LOGS" bash scripts/cloud/run_model.sh "$m"
  else
    if [ "$(n_png "$EXP/samples/$m")" != "$SWEEP_N" ]; then
      timed "sweep_$m" $VENV -u scripts/generate.py --config "configs/$m.yaml" --num-images "$NUM" --batch "$gb" \
        --seed 0 ${grid_args[@]+"${grid_args[@]}"}
    fi
    timed "eval_$m" $VENV -u scripts/evaluate.py --model "$m" --n-boot "$NBOOT" ${grid_args[@]+"${grid_args[@]}"}
  fi
done

# ---------------------------------------------------------------- ranking (Inception macro KID)
read -r TOP1 CFG1 TOP2 CFG2 < <(MODELS="$MODELS" $VENV - <<'PY'
import os
from cytodiff.eval.evaluate import best_cfg_of, RESULTS_DIR
import pandas as pd
rows = []
for m in os.environ["MODELS"].split():
    cfg = best_cfg_of(m)
    d = pd.read_csv(RESULTS_DIR / f"metrics_{m}.csv")
    d = d[(d["class"] == "macro") & (d["backbone"] == "inception") & (d["cfg"] == cfg)]
    rows.append((float(d["kid"].iloc[0]), m, cfg))
rows.sort()
print(rows[0][1], f"{rows[0][2]:g}", rows[1][1], f"{rows[1][2]:g}")
PY
)
log "best: $TOP1 (CFG $CFG1); runner-up: $TOP2 (CFG $CFG2)"

# ---------------------------------------------------------------- extra training seeds (top two)
for pair in "$TOP1 $CFG1" "$TOP2 $CFG2"; do
  read -r m cfg <<< "$pair"
  gb=$(gen_batch "$m")
  for s in $SEEDS; do
    run="${m}_s${s}"
    if has_metrics "$run"; then log "$run: finished earlier, skipping"; continue; fi
    if [ ! -f "$(adapter "$run")" ]; then
      timed "train_$run" bash -o pipefail -c "$VENV -u scripts/train.py --config configs/$m.yaml --skip-cache --seed $s --run-name $run --save-every 100000 ${MAX_STEPS:+--max-steps $MAX_STEPS} 2>&1 | tee $LOGS/train_$run.log"
    fi
    # Same sampling seed (0) as the seed-0 run, so only the training seed differs.
    if [ "$(n_png "$EXP/samples/$run")" != $((NUM * 2)) ]; then
      timed "sample_$run" $VENV -u scripts/generate.py --config "configs/$m.yaml" --run-name "$run" --cfg "$cfg" \
        --num-images "$NUM" --batch "$gb" --seed 0
    fi
    timed "eval_$run" $VENV -u scripts/evaluate.py --model "$run" --cfg "$cfg" --n-boot "$NBOOT"
  done
done

# ---------------------------------------------------------------- color transfer on the top two
CT_RUNS=""
for pair in "$TOP1 $CFG1" "$TOP2 $CFG2"; do
  read -r m cfg <<< "$pair"
  for space in rgb lab; do
    ct="${m}_ct_${space}"; CT_RUNS="$CT_RUNS $ct"
    if has_metrics "$ct"; then log "$ct: finished earlier, skipping"; continue; fi
    timed "ct_$ct" $VENV scripts/colortransfer.py --model "$m" --cfg "$cfg" --space "$space"
    timed "eval_$ct" $VENV -u scripts/evaluate.py --model "$ct" --cfg "$cfg" --n-boot "$NBOOT"
  done
done

# ---------------------------------------------------------------- analyses
timed real_baseline $VENV scripts/evaluate.py --real-baseline --n-boot "$NBOOT"
timed compare bash -o pipefail -c "$VENV scripts/compare_models.py --models $MODELS --extra $ZS_RUNS $CT_RUNS --n-boot $NBOOT 2>&1 | tee $LOGS/compare_models.log"
for cls in ADC SCC; do
  timed "color_stats_$cls" $VENV scripts/color_stats.py --cls "$cls" --sets real real_all "$TOP1:$CFG1" "${TOP1}_ct_rgb:$CFG1" "${TOP1}_ct_lab:$CFG1"
done
timed memorization bash -o pipefail -c "$VENV scripts/memorization.py --runs $MODELS $ZS_RUNS 2>&1 | tee $LOGS/memorization.log"
timed label_check bash -o pipefail -c "$VENV scripts/label_check.py --runs $MODELS $ZS_RUNS 2>&1 | tee $LOGS/label_check.log"
timed vae_reconstruction bash -o pipefail -c "$VENV scripts/vae_reconstruction.py --models $MODELS $VAE_LIMIT 2>&1 | tee $LOGS/vae_reconstruction.log"

# LoRA parameter counts from the training logs (all four models, same LoRA settings).
LOGS="$LOGS" EXP="$EXP" $VENV - <<'PY'
import os, re
from pathlib import Path
import pandas as pd
rows = []
for log in sorted(Path(os.environ["LOGS"]).glob("train_*.log")):
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"(\S+) \((\w+)\): trainable ([\d,]+)/([\d,]+)", line)
        if m:
            rows.append({"log": log.name, "model": m.group(1), "family": m.group(2),
                         "lora_params": int(m.group(3).replace(",", "")), "denoiser_params": int(m.group(4).replace(",", ""))})
            break
d = pd.DataFrame(rows)
if not d.empty:
    d["percent"] = 100 * d["lora_params"] / d["denoiser_params"]
d.to_csv(Path(os.environ["EXP"]) / "results" / "lora_params.csv", index=False)
print(d.to_string(index=False))
PY

if [ "$SMOKE" = 1 ]; then
  log "SMOKE RUN PASSED: every stage ran end to end. Start the real run with: bash scripts/cloud/run_all.sh"
  exit 0
fi

# ---------------------------------------------------------------- results archive
OUT="cytodiff_v2_results_$(date +%Y%m%d_%H%M).tar.gz"
MODELS="$MODELS" TOP1="$TOP1" $VENV - <<'PY' > "$LOGS/archive_files.txt"
import os
from pathlib import Path
from cytodiff.constants import EXPERIMENTS_DIR
from cytodiff.eval.evaluate import RESULTS_DIR, best_cfg_of, gen_paths
root = Path.cwd()
models, top1 = os.environ["MODELS"].split(), os.environ["TOP1"]
runs = models + [p.stem.removeprefix("metrics_") for p in RESULTS_DIR.glob("metrics_*.csv")]
files = {str(p.relative_to(root)) for p in EXPERIMENTS_DIR.glob("checkpoints/*/final/*")}   # every adapter
for cls in ("ADC", "SCC"):
    files.update(str(p.relative_to(root)) for p in gen_paths(top1, best_cfg_of(top1), cls))   # released set
    for run in dict.fromkeys(runs):                                     # first 8 per class per run (figures)
        files.update(str(p.relative_to(root)) for p in gen_paths(run, best_cfg_of(run), cls)[:8])
print("\n".join(sorted(files)))
PY
tar -czf "$OUT" "$EXP/results" "$EXP/features" "$LOGS" data/splits/manifest.csv -T "$LOGS/archive_files.txt"
log "DONE. Download $OUT ($(du -h "$OUT" | cut -f1)); see RERUN.md."
