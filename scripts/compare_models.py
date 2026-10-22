#!/usr/bin/env python
"""Rank the backbones at their best CFG, with paired bootstrap tests and seed variability.

For each feature extractor, every model is scored at its own best CFG (Inception macro KID). Each
bootstrap replicate uses the SAME resample of held-out real images (patients drawn with
replacement) for all models, so model differences are paired. Outputs, in experiments/results/:

  model_ranking.csv    macro KID and macro FD per model, with 95 % intervals
  model_pairwise.csv   for each pair: difference in macro KID and macro FD, 95 % interval, and
                       the share of replicates in which the first model scores lower (better)
  seed_variability.csv macro KID / FD per training seed for models with extra seed runs
                       (folders <model>_s1, <model>_s2, ...), evaluated at the seed-0 best CFG
  lora_vs_zeroshot.csv for each model with a <model>_zeroshot run: the same backbone without LoRA
                       vs with LoRA, each at its own best CFG; paired difference in macro KID and
                       FD with 95 % interval and the share of replicates in which LoRA is better

Intervals: for a single run, KID and FD use normal bootstrap intervals, the point estimate +- 1.96
bootstrap standard deviations, because their resampled values are shifted upward (KID: duplicated
patients raise the share of within-patient pairs; FD: repeated images). Paired KID differences share
the real resample, so that shift cancels and their percentile intervals are used; paired FD
differences keep normal intervals.

Extra runs (zero-shot baselines, color-transfer variants) are added to the ranking with --extra.

Usage:
    venv/bin/python scripts/compare_models.py --models sd15 sd21 sdxl sd3 \
        --extra sd15_zeroshot sd21_zeroshot sdxl_zeroshot sd3_zeroshot
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from cytodiff.constants import CLASSES
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import N_BOOT, RESULTS_DIR, SEED, best_cfg_of, gen_features, real_features
from cytodiff.eval.metrics import (
    BACKBONES, KernelSet, bootstrap_fd, cluster_resamples, default_device, frechet_distance,
    image_resamples, normal_ci,
)


def ci(x: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap interval (used for paired KID differences)."""
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap comparison of backbones.")
    ap.add_argument("--models", nargs="+", required=True, help="base runs to rank")
    ap.add_argument("--extra", nargs="*", default=[], help="extra runs shown in the ranking")
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES), choices=BACKBONES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    device = default_device()
    manifest = pd.read_csv(MANIFEST_PATH)
    runs = args.models + args.extra
    cfgs = {r: best_cfg_of(r) for r in runs}
    ranking, pairs, lora_gain = [], [], []

    for backbone in args.backbones:
        real = {c: real_features(manifest, c, device, backbone) for c in CLASSES}
        real_idx = {c: cluster_resamples(real[c][1], args.n_boot, SEED + k) for k, c in enumerate(CLASSES)}
        point, kid_draws, fd_point, fd_draws = {}, {}, {}, {}
        for r_i, run in enumerate(runs):
            kp, kd, fp, fdd = [], [], [], []
            for k, cls in enumerate(CLASSES):
                fr, fg = real[cls][0], gen_features(run, cfgs[run], cls, device, backbone)
                gen_idx = image_resamples(len(fg), args.n_boot, SEED + 1000 * (r_i + 1) + k)
                ks = KernelSet(fr, fg)
                kp.append(ks.kid())
                kd.append([ks.kid(i, j) for i, j in zip(real_idx[cls], gen_idx)])
                fp.append(frechet_distance(fr, fg))
                fdd.append(bootstrap_fd(fr, fg, real_idx[cls], gen_idx))
            point[run], kid_draws[run] = float(np.mean(kp)), np.mean(kd, axis=0)
            fd_point[run], fd_draws[run] = float(np.mean(fp)), np.mean(fdd, axis=0)

        order = sorted(runs, key=point.get)
        for rank, run in enumerate(order, 1):
            ranking.append({"backbone": backbone, "rank": rank, "model": run, "cfg": cfgs[run],
                            "baseline_or_variant": run in args.extra,
                            "macro_kid": point[run], "kid_ci_low": normal_ci(point[run], kid_draws[run])[0],
                            "kid_ci_high": normal_ci(point[run], kid_draws[run])[1],
                            "macro_fd": fd_point[run], "fd_ci_low": normal_ci(fd_point[run], fd_draws[run])[0],
                            "fd_ci_high": normal_ci(fd_point[run], fd_draws[run])[1]})
        base_order = [r for r in order if r in args.models]
        for a, b in combinations(base_order, 2):
            dk, df_ = kid_draws[a] - kid_draws[b], fd_draws[a] - fd_draws[b]
            pairs.append({"backbone": backbone, "better": a, "worse": b,
                          "kid_diff": point[a] - point[b], "kid_diff_ci_low": ci(dk)[0], "kid_diff_ci_high": ci(dk)[1],
                          "p_better_kid": float((dk < 0).mean()),
                          "fd_diff": fd_point[a] - fd_point[b],
                          "fd_diff_ci_low": normal_ci(fd_point[a] - fd_point[b], df_)[0],
                          "fd_diff_ci_high": normal_ci(fd_point[a] - fd_point[b], df_)[1],
                          "p_better_fd": float((df_ < 0).mean())})
        for m in args.models:
            zs = f"{m}_zeroshot"
            if zs not in point:
                continue
            dk, df_ = kid_draws[m] - kid_draws[zs], fd_draws[m] - fd_draws[zs]
            lora_gain.append({"backbone": backbone, "model": m, "lora_cfg": cfgs[m], "zeroshot_cfg": cfgs[zs],
                              "kid_zeroshot": point[zs], "kid_lora": point[m], "kid_diff": point[m] - point[zs],
                              "kid_diff_ci_low": ci(dk)[0], "kid_diff_ci_high": ci(dk)[1],
                              "p_lora_better_kid": float((dk < 0).mean()),
                              "fd_zeroshot": fd_point[zs], "fd_lora": fd_point[m], "fd_diff": fd_point[m] - fd_point[zs],
                              "fd_diff_ci_low": normal_ci(fd_point[m] - fd_point[zs], df_)[0],
                              "fd_diff_ci_high": normal_ci(fd_point[m] - fd_point[zs], df_)[1],
                              "p_lora_better_fd": float((df_ < 0).mean())})

    # Seed variability: <model>_s<k> runs evaluated at the seed-0 best CFG.
    seed_rows = []
    for m in args.models:
        for s in range(0, 10):
            run = m if s == 0 else f"{m}_s{s}"
            path = RESULTS_DIR / f"metrics_{run}.csv"
            if not path.exists():
                continue
            d = pd.read_csv(path)
            d = d[(d["class"] == "macro") & np.isclose(d["cfg"], cfgs[m])]
            for _, row in d.iterrows():
                seed_rows.append({"model": m, "seed": s, "cfg": cfgs[m], "backbone": row["backbone"],
                                  "macro_kid": row["kid"], "macro_fd": row["fd"]})
    seeds = pd.DataFrame(seed_rows)
    multi = seeds.groupby("model")["seed"].nunique() if not seeds.empty else pd.Series(dtype=int)
    seeds = seeds[seeds["model"].isin(multi[multi > 1].index)] if not seeds.empty else seeds

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking, pairs, lora_gain = pd.DataFrame(ranking), pd.DataFrame(pairs), pd.DataFrame(lora_gain)
    ranking.to_csv(RESULTS_DIR / "model_ranking.csv", index=False)
    pairs.to_csv(RESULTS_DIR / "model_pairwise.csv", index=False)
    lora_gain.to_csv(RESULTS_DIR / "lora_vs_zeroshot.csv", index=False)
    seeds.to_csv(RESULTS_DIR / "seed_variability.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 200)
    print(ranking.to_string(index=False)); print(); print(pairs.to_string(index=False))
    if not lora_gain.empty:
        print("\nLoRA vs zero-shot (negative difference = LoRA closer to the real images):")
        print(lora_gain.to_string(index=False))
    if not seeds.empty:
        print("\nSeed variability (mean +- SD over seeds):")
        print(seeds.groupby(["model", "backbone"])[["macro_kid", "macro_fd"]].agg(["mean", "std", "count"]).to_string())
        top = [m for m in ranking[ranking["backbone"] == args.backbones[0]]["model"] if m in multi[multi > 1].index][:2]
        if len(top) == 2:
            a, b = top
            w = seeds[seeds["backbone"] == args.backbones[0]].pivot(index="seed", columns="model", values="macro_kid")
            w = w.dropna(subset=[a, b])
            print(f"\n{a} < {b} (macro KID, {args.backbones[0]}) in {(w[a] < w[b]).sum()} of {len(w)} seeds")
    print(f"\n  wrote model_ranking.csv, model_pairwise.csv, seed_variability.csv, lora_vs_zeroshot.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
