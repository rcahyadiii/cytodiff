#!/usr/bin/env python
"""Do the generated images carry their intended class?

A linear probe (logistic regression on standardized pathology-foundation-model features) is
trained on REAL training images and tested on
  (a) held-out real images: the reference accuracy of the probe itself, and
  (b) each run's synthetic images at its best CFG: label fidelity (the "GAN-test" of
      Shmelkov et al., 2018); low accuracy means the images do not show their caption's subtype.
The reverse, training on a run's synthetic images and testing on held-out real images, is the
Classification Accuracy Score (Ravuri and Vinyals, 2019).
Accuracy and balanced accuracy with 95 % bootstrap intervals (held-out real images resampled by
patient, synthetic images by image). Output: experiments/results/label_check.csv

Usage:
    venv/bin/python scripts/label_check.py --runs sd15 sd21 sdxl sd3 sd3_zeroshot
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cytodiff.constants import CLASSES
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import RESULTS_DIR, best_cfg_of, gen_features, real_features
from cytodiff.eval.metrics import cluster_resamples, default_device, image_resamples

N_BOOT = 1000


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def stats(y, pred, groups, seed) -> dict:
    correct = (y == pred)
    def bal(idx):
        return float(np.mean([correct[idx][y[idx] == c].mean() for c in np.unique(y[idx])]))
    reps = (cluster_resamples(groups, N_BOOT, seed) if groups is not None
            else image_resamples(len(y), N_BOOT, seed))
    acc = np.array([correct[i].mean() for i in reps])
    bacc = np.array([bal(i) for i in reps])
    full = np.arange(len(y))
    return {"accuracy": float(correct.mean()), "acc_ci_low": np.percentile(acc, 2.5), "acc_ci_high": np.percentile(acc, 97.5),
            "balanced_accuracy": bal(full), "bacc_ci_low": np.percentile(bacc, 2.5), "bacc_ci_high": np.percentile(bacc, 97.5),
            "n_test": len(y)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear-probe label check and Classification Accuracy Score.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--backbones", nargs="*", default=["uni", "phikon"])
    args = ap.parse_args()

    device = default_device()
    df = pd.read_csv(MANIFEST_PATH)
    rows = []
    for backbone in args.backbones:
        def real(split):
            parts = [real_features(df, c, device, backbone, split) for c in CLASSES]
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([np.full(len(p[0]), k) for k, p in enumerate(parts)])
            g = np.concatenate([p[1] for p in parts])
            return x, y, g
        x_tr, y_tr, _ = real("train")
        x_te, y_te, g_te = real("eval")
        clf = probe().fit(x_tr, y_tr)
        rows.append({"backbone": backbone, "train_on": "real", "test_on": "real_heldout", "run": "",
                     **stats(y_te, clf.predict(x_te), g_te, 1)})
        for k, run in enumerate(args.runs):
            cfg = best_cfg_of(run)
            parts = [gen_features(run, cfg, c, device, backbone) for c in CLASSES]
            x_s = np.concatenate(parts)
            y_s = np.concatenate([np.full(len(p), j) for j, p in enumerate(parts)])
            rows.append({"backbone": backbone, "train_on": "real", "test_on": "synthetic", "run": run, "cfg": cfg,
                         **stats(y_s, clf.predict(x_s), None, 10 + k)})
            cas = probe().fit(x_s, y_s)
            rows.append({"backbone": backbone, "train_on": "synthetic", "test_on": "real_heldout", "run": run, "cfg": cfg,
                         **stats(y_te, cas.predict(x_te), g_te, 100 + k)})
            print(f"  [{backbone}] {run}: label fidelity {rows[-2]['accuracy']:.3f}, CAS {rows[-1]['accuracy']:.3f}")
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "label_check.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(out.to_string(index=False))
    print(f"\n  wrote {RESULTS_DIR / 'label_check.csv'}")


if __name__ == "__main__":
    main()
