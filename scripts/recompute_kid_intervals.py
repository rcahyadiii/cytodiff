#!/usr/bin/env python
"""Recompute single-run KID intervals as normal bootstrap intervals from the cached features.

The v2 cloud run reported percentile intervals for the KID of a single run. Resampling patients
raises the share of within-patient pairs, which shifts resampled KID values upward, so a percentile
interval could even exclude its own estimate (real-versus-real baseline). The pipeline now reports a
normal interval (estimate +- 1.96 bootstrap SD) instead; see cytodiff.eval.metrics.

This script rebuilds exactly the bootstrap draws of the cloud run (same features, resampling
functions, seeds, and run order), checks that they reproduce the stored percentile intervals, and
only then replaces the KID interval columns of:
    metrics_<run>.csv, baseline_real.csv, model_ranking.csv
Point estimates, FD, and paired differences are unchanged. The original files are kept in
experiments/results/_percentile_kid_intervals/. No images or GPU are needed.

Usage:
    python scripts/recompute_kid_intervals.py --models sd15 sd21 sdxl sd3 \
        --extra sd15_zeroshot sd21_zeroshot sdxl_zeroshot sd3_zeroshot \
                sd3_ct_rgb sd3_ct_lab sd21_ct_rgb sd21_ct_lab
    (--models/--extra in the order run_all.sh passed them to compare_models.py)
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from cytodiff.constants import CLASSES
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import FEATURES_DIR, N_BOOT, RESULTS_DIR, SEED, best_cfg_of
from cytodiff.eval.metrics import (
    KernelSet, bootstrap_kid, cluster_resamples, image_resamples, normal_ci,
)

BACKUP = RESULTS_DIR / "_percentile_kid_intervals"
TOL = 1e-6


def real_cached(manifest: pd.DataFrame, backbone: str, cls: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(FEATURES_DIR / backbone / f"real_{split}_{cls}.npz")
    patients = manifest.set_index("path").loc[z["names"], "patient_id"].astype(str).to_numpy()
    return z["feats"], patients


def gen_cached(run: str, cfg: float, cls: str, backbone: str) -> np.ndarray:
    return np.load(FEATURES_DIR / backbone / run / f"cfg_{cfg:g}_{cls}.npz")["feats"]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def backup(path) -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    if not (BACKUP / path.name).exists():
        shutil.copy2(path, BACKUP / path.name)


def redo_metrics_file(path, manifest, n_boot: int) -> tuple[pd.DataFrame, int]:
    """evaluate_model(): real resample seed SEED+k, generated SEED+100+k, macro = mean of classes."""
    d = pd.read_csv(path)
    checked = 0
    for backbone in d["backbone"].unique():
        real = {c: real_cached(manifest, backbone, c, "eval") for c in CLASSES}
        real_idx = {c: cluster_resamples(real[c][1], n_boot, SEED + k) for k, c in enumerate(CLASSES)}
        for cfg in d.loc[d["backbone"] == backbone, "cfg"].unique():
            boots, kids = {}, {}
            for k, cls in enumerate(CLASSES):
                ks = KernelSet(real[cls][0], gen_cached(d["model"].iloc[0], cfg, cls, backbone))
                kids[cls] = ks.kid()
                boots[cls] = bootstrap_kid(ks, real_idx[cls], image_resamples(len(ks.kyy), n_boot, SEED + 100 + k))
            boots["macro"] = np.mean([boots[c] for c in CLASSES], axis=0)
            kids["macro"] = float(np.mean([kids[c] for c in CLASSES]))
            for cls in (*CLASSES, "macro"):
                m = (d["backbone"] == backbone) & np.isclose(d["cfg"], cfg) & (d["class"] == cls)
                if not m.any():
                    continue
                row = d.loc[m].iloc[0]
                lo, hi = np.percentile(boots[cls], [2.5, 97.5])
                if not (close(row["kid"], kids[cls]) and close(row["kid_ci_low"], lo) and close(row["kid_ci_high"], hi)):
                    raise SystemExit(f"!! {path.name} {backbone} cfg={cfg:g} {cls}: draws do not reproduce "
                                     f"(kid {row['kid']:.6g} vs {kids[cls]:.6g}, low {row['kid_ci_low']:.6g} vs {lo:.6g})")
                checked += 1
                d.loc[m, ["kid_ci_low", "kid_ci_high"]] = normal_ci(row["kid"], boots[cls])
    return d, checked


def redo_baseline(manifest, n_boot: int) -> tuple[pd.DataFrame, int]:
    """evaluate_real_baseline(): real seed SEED+k; training images resampled by patient, SEED+200+k."""
    path = RESULTS_DIR / "baseline_real.csv"
    d = pd.read_csv(path)
    n = 0
    for backbone in d["backbone"].unique():
        boots, kids = {}, {}
        for k, cls in enumerate(CLASSES):
            fr, gr = real_cached(manifest, backbone, cls, "eval")
            ft, gt = real_cached(manifest, backbone, cls, "train")
            ks = KernelSet(fr, ft)
            kids[cls] = ks.kid()
            boots[cls] = bootstrap_kid(ks, cluster_resamples(gr, n_boot, SEED + k), cluster_resamples(gt, n_boot, SEED + 200 + k))
        boots["macro"] = np.mean([boots[c] for c in CLASSES], axis=0)
        kids["macro"] = float(np.mean([kids[c] for c in CLASSES]))
        for cls in (*CLASSES, "macro"):
            m = (d["backbone"] == backbone) & (d["class"] == cls)
            row = d.loc[m].iloc[0]
            lo, hi = np.percentile(boots[cls], [2.5, 97.5])
            if not (close(row["kid"], kids[cls]) and close(row["kid_ci_low"], lo) and close(row["kid_ci_high"], hi)):
                raise SystemExit(f"!! baseline {backbone} {cls}: draws do not reproduce")
            d.loc[m, ["kid_ci_low", "kid_ci_high"]] = normal_ci(row["kid"], boots[cls])
            n += 1
    return d, n


def redo_ranking(manifest, runs: list[str], n_boot: int) -> tuple[pd.DataFrame, int]:
    """compare_models.py: real seed SEED+k; run r_i's generated images SEED+1000*(r_i+1)+k."""
    path = RESULTS_DIR / "model_ranking.csv"
    d = pd.read_csv(path)
    cfgs = {r: best_cfg_of(r) for r in runs}
    n = 0
    for backbone in d["backbone"].unique():
        real = {c: real_cached(manifest, backbone, c, "eval") for c in CLASSES}
        real_idx = {c: cluster_resamples(real[c][1], n_boot, SEED + k) for k, c in enumerate(CLASSES)}
        for r_i, run in enumerate(runs):
            kp, kd = [], []
            for k, cls in enumerate(CLASSES):
                ks = KernelSet(real[cls][0], gen_cached(run, cfgs[run], cls, backbone))
                gen_idx = image_resamples(len(ks.kyy), n_boot, SEED + 1000 * (r_i + 1) + k)
                kp.append(ks.kid())
                kd.append([ks.kid(i, j) for i, j in zip(real_idx[cls], gen_idx)])
            point, draws = float(np.mean(kp)), np.mean(kd, axis=0)
            m = (d["backbone"] == backbone) & (d["model"] == run)
            row = d.loc[m].iloc[0]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            if not (close(row["macro_kid"], point) and close(row["kid_ci_low"], lo) and close(row["kid_ci_high"], hi)):
                raise SystemExit(f"!! ranking {backbone} {run}: draws do not reproduce (check the run order)")
            d.loc[m, ["kid_ci_low", "kid_ci_high"]] = normal_ci(point, draws)
            n += 1
    return d, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if (BACKUP / "model_ranking.csv").exists():
        raise SystemExit(f"Already converted (backup in {BACKUP}); restore the originals from there to rerun.")
    manifest = pd.read_csv(MANIFEST_PATH)
    out = {}   # verify everything first; write only if every interval reproduces
    for path in sorted(RESULTS_DIR.glob("metrics_*.csv")):
        out[path], n = redo_metrics_file(path, manifest, args.n_boot)
        print(f"  {path.name}: {n} intervals reproduced")
    out[RESULTS_DIR / "baseline_real.csv"], n = redo_baseline(manifest, args.n_boot)
    print(f"  baseline_real.csv: {n} intervals reproduced")
    out[RESULTS_DIR / "model_ranking.csv"], n = redo_ranking(manifest, args.models + args.extra, args.n_boot)
    print(f"  model_ranking.csv: {n} intervals reproduced")
    for path, d in out.items():
        backup(path)
        d.to_csv(path, index=False)
    print(f"  all KID intervals converted to normal intervals; originals kept in {BACKUP}")


if __name__ == "__main__":
    main()
