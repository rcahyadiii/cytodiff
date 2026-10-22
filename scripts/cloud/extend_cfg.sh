#!/usr/bin/env bash
# Extend the guidance grid below its lowest value (1.5) for every run whose best scale is that lowest
# value, then redo the stages that depend on the best scale. Started while run_all.sh is still running,
# it first waits for the main run to finish.
#
#   EXTRA_CFGS (default "1 1.25", written as %g). Guidance 1 means no classifier-free guidance: the
#   pipelines then use the conditional prediction only and ignore the negative prompt. It is the natural
#   lower end of the scale; below 1 the caption, and with it the class, loses its effect.
#
# Steps: wait for run_all.sh -> find runs (LoRA and zero-shot) whose best scale is the lowest grid
# value -> generate EXTRA_CFGS for them with the same captions, sampling seed and batch size ->
# evaluate them on the extended grid -> if the best scale of one of the two best models moved, set
# aside its seed samples and color-transfer outputs made at the old scale (experiments/superseded_cfg/)
# -> rerun run_all.sh, which is resumable: it trains nothing, remakes those seed samples and color
# transfers at the new best scale, reruns all analyses, and writes a new results archive.
# If the extension changes WHICH two models are best, it stops before run_all.sh, because extra seeds
# for a new model would need new training runs.
#
# Usage (on the pod, in a second tmux window):
#   bash scripts/cloud/extend_cfg.sh 2>&1 | tee outputs/logs/extend_cfg.log
#   DRY_RUN=1 bash scripts/cloud/extend_cfg.sh     # print the plan without generating anything
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/cloud/env.sh
VENV="${VENV:-venv/bin/python}"
MODELS="${MODELS:-sd15 sd21 sdxl sd3}"
SEEDS="${SEEDS:-1 2}"
EXTRA_CFGS="${EXTRA_CFGS:-1 1.25}"
DRY_RUN="${DRY_RUN:-0}"
EXP="${EXP:-experiments}"; LOGS="${LOGS:-outputs/logs}"; NUM=200; NBOOT=1000
export CYTODIFF_EXPERIMENTS_DIR="$PWD/$EXP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"
TIMINGS=$LOGS/timings.csv
log() { if [ "$DRY_RUN" = 1 ]; then echo "[dry run] $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run_all.log"; fi; }
run() { if [ "$DRY_RUN" = 1 ]; then echo "  [dry run] $*"; else "$@"; fi; }
timed() {  # timed <stage-name> <command...>
  local name=$1; shift
  local t0; t0=$(date -u +%s); log "== $name =="
  run "$@"
  local t1; t1=$(date -u +%s); [ "$DRY_RUN" = 1 ] || echo "$name,$t0,$t1,$((t1 - t0))" >> "$TIMINGS"
}
gen_batch() { case $1 in sd15|sd21) echo 8 ;; sdxl) echo 4 ;; sd3) echo 2 ;; *) echo 4 ;; esac; }
n_png() { find "$1" -name '*.png' 2>/dev/null | wc -l | tr -d ' '; }
ranking() {  # prints "TOP1 CFG1 TOP2 CFG2", the same rule as run_all.sh
  MODELS="$MODELS" $VENV - <<'PY'
import os
import pandas as pd
from cytodiff.eval.evaluate import RESULTS_DIR, best_cfg_of
rows = []
for m in os.environ["MODELS"].split():
    cfg = best_cfg_of(m)
    d = pd.read_csv(RESULTS_DIR / f"metrics_{m}.csv")
    d = d[(d["class"] == "macro") & (d["backbone"] == "inception") & (d["cfg"] == cfg)]
    rows.append((float(d["kid"].iloc[0]), m, cfg))
rows.sort()
print(rows[0][1], f"{rows[0][2]:g}", rows[1][1], f"{rows[1][2]:g}")
PY
}
STASH="$EXP/superseded_cfg/$(date +%Y%m%d_%H%M%S)"
stash() {  # move paths (relative to the project) under $STASH, keeping their layout
  local p
  for p in "$@"; do
    [ -e "$p" ] || continue
    run mkdir -p "$STASH/$(dirname "$p")"
    run mv "$p" "$STASH/$p"
  done
}

# ---------------------------------------------------------------- wait for the main run
if [ "$DRY_RUN" != 1 ]; then
  if pgrep -f "scripts/cloud/run_all.sh" >/dev/null; then
    echo "run_all.sh is still running; checking again every 5 minutes..."
    while pgrep -f "scripts/cloud/run_all.sh" >/dev/null; do sleep 300; done
  fi
  grep -q "DONE. Download" "$LOGS/run_all.log" \
    || { echo "!! run_all.sh did not finish. Resume it first: bash scripts/cloud/run_all.sh"; exit 1; }
fi

# ---------------------------------------------------------------- runs whose best scale is the lowest
read -r OLD_TOP1 OLD_CFG1 OLD_TOP2 OLD_CFG2 < <(ranking)
[ -n "${OLD_CFG2:-}" ] || { echo "!! could not rank the models from experiments/results"; exit 1; }
EDGE=$(MODELS="$MODELS" EXTRA="$EXTRA_CFGS" $VENV - <<'PY'
import os
import numpy as np
import pandas as pd
from cytodiff.constants import CFG_SWEEP
from cytodiff.eval.evaluate import PRIMARY_BACKBONE, RESULTS_DIR, best_cfg_of
low, extra = min(CFG_SWEEP), [float(c) for c in os.environ["EXTRA"].split()]
edge = []
for m in os.environ["MODELS"].split():
    for run in (m, f"{m}_zeroshot"):
        d = pd.read_csv(RESULTS_DIR / f"metrics_{run}.csv")
        d = d[(d["class"] == "macro") & (d["backbone"] == PRIMARY_BACKBONE)]
        done = all(np.isclose(d["cfg"], c).any() for c in extra)     # already extended earlier
        if np.isclose(best_cfg_of(run), low) and not done:
            edge.append(run)
print(" ".join(edge))
PY
)
log "== CFG extension (${EXTRA_CFGS}): runs whose best scale is the lowest grid value: ${EDGE:-none} =="
if [ -z "$EDGE" ]; then log "no run has its best scale at the lowest grid value; nothing to extend"; exit 0; fi
GRID_ALL=$(EXTRA="$EXTRA_CFGS" $VENV -c "import os; from cytodiff.constants import CFG_SWEEP; \
g = sorted(set(map(float, os.environ['EXTRA'].split())) | set(CFG_SWEEP)); print(' '.join(f'{c:g}' for c in g))")

for r in $EDGE; do
  m=${r%_zeroshot}; zs=(); [ "$r" = "$m" ] || zs=(--zero-shot)
  todo=""
  for c in $EXTRA_CFGS; do [ "$(n_png "$EXP/samples/$r/cfg_$c")" = $((NUM * 2)) ] || todo="$todo $c"; done
  if [ -n "$todo" ]; then
    # shellcheck disable=SC2086
    timed "ext_sweep_$r" $VENV -u scripts/generate.py --config "configs/$m.yaml" ${zs[@]+"${zs[@]}"} \
      --run-name "$r" --num-images "$NUM" --batch "$(gen_batch "$m")" --seed 0 --cfg $todo
  fi
  # shellcheck disable=SC2086
  timed "ext_eval_$r" $VENV -u scripts/evaluate.py --model "$r" --n-boot "$NBOOT" --cfg $GRID_ALL
done

# ---------------------------------------------------------------- did the best scale of the top two move?
if [ "$DRY_RUN" = 1 ]; then
  TOP1=$OLD_TOP1; CFG1=$OLD_CFG1; TOP2=$OLD_TOP2; CFG2=$OLD_CFG2
  for r in $EDGE; do   # in a dry run, show what happens if every extended run moves to the lowest new scale
    lowest=$(echo $EXTRA_CFGS | tr ' ' '\n' | sort -g | head -1)
    if [ "$r" = "$TOP1" ]; then CFG1=$lowest; fi
    if [ "$r" = "$TOP2" ]; then CFG2=$lowest; fi
  done
else
  read -r TOP1 CFG1 TOP2 CFG2 < <(ranking)
fi
log "best before: $OLD_TOP1 (CFG $OLD_CFG1), $OLD_TOP2 (CFG $OLD_CFG2); after: $TOP1 (CFG $CFG1), $TOP2 (CFG $CFG2)"
if [ "$(printf '%s\n' "$TOP1" "$TOP2" | sort | tr '\n' ' ')" != "$(printf '%s\n' "$OLD_TOP1" "$OLD_TOP2" | sort | tr '\n' ' ')" ]; then
  log "!! the two best models changed; their extra seeds would need new training runs. Stopped before run_all.sh."
  exit 3
fi
old_cfg() { if [ "$1" = "$OLD_TOP1" ]; then echo "$OLD_CFG1"; else echo "$OLD_CFG2"; fi; }
for pair in "$TOP1 $CFG1" "$TOP2 $CFG2"; do
  read -r m cfg <<< "$pair"
  old=$(old_cfg "$m")
  if [ "$cfg" != "$old" ]; then
    log "$m: best scale moved from $old to $cfg; setting aside seed and color-transfer outputs made at $old in $STASH"
    for s in $SEEDS; do stash "$EXP/samples/${m}_s$s" "$EXP/results/metrics_${m}_s$s.csv"; done
    for space in rgb lab; do stash "$EXP/samples/${m}_ct_$space" "$EXP/results/metrics_${m}_ct_$space.csv"; done
  fi
done
for a in cytodiff_v2_results_*.tar.gz; do stash "$a"; done   # the new archive replaces it

# ---------------------------------------------------------------- redo everything that depends on the best scale
log "== rerunning run_all.sh: seed samples and color transfer at the new best scales, analyses, archive =="
run bash scripts/cloud/run_all.sh
