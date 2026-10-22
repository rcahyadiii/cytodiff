#!/usr/bin/env python
"""Label fidelity across the guidance grid, from cached features (no images or GPU needed).

label_check.py tests each run only at its best CFG. This script applies the same linear probe
(standardized features, logistic regression, trained on REAL training images) to the generated
images of every CFG, so that class fidelity can be set against the Inception macro KID that chose
the best CFG. Output: experiments/results/label_vs_cfg.csv with, per run, backbone, and CFG:
  accuracy (+ 95 % image-bootstrap interval)  probe accuracy on the synthetic images
  macro_kid_inception                         the macro KID that selected the best CFG
  kid_own_class, kid_cross_class              KID (this backbone) of generated ADC/SCC against held-out
                                              real images of the same / the other class (class means)
  cos_paired, cos_unpaired                    mean cosine similarity of ADC image i and SCC image i,
                                              which share their starting noise, and of ADC i, SCC j

Usage:
    python scripts/label_vs_cfg.py --runs sd15 sd21 sdxl sd3 sd15_zeroshot sd21_zeroshot sdxl_zeroshot sd3_zeroshot
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cytodiff.constants import CLASSES
from cytodiff.eval.evaluate import FEATURES_DIR, PRIMARY_BACKBONE, RESULTS_DIR
from cytodiff.eval.metrics import KernelSet

N_BOOT = 1000


def probe():   # identical to scripts/label_check.py
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def real(backbone: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    parts = [np.load(FEATURES_DIR / backbone / f"real_{split}_{c}.npz")["feats"] for c in CLASSES]
    return np.concatenate(parts), np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--backbones", nargs="*", default=["uni", "phikon"])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for backbone in args.backbones:
        clf = probe().fit(*real(backbone, "train"))
        held = {c: np.load(FEATURES_DIR / backbone / f"real_eval_{c}.npz")["feats"] for c in CLASSES}
        for run in args.runs:
            kid = pd.read_csv(RESULTS_DIR / f"metrics_{run}.csv")
            kid = kid[(kid["class"] == "macro") & (kid["backbone"] == PRIMARY_BACKBONE)].set_index("cfg")["kid"]
            cfgs = sorted({float(m.group(1)) for p in (FEATURES_DIR / backbone / run).glob("cfg_*_ADC.npz")
                           if (m := re.match(r"cfg_([\d.]+)_ADC", p.stem))})
            for cfg in cfgs:
                parts = [np.load(FEATURES_DIR / backbone / run / f"cfg_{cfg:g}_{c}.npz")["feats"] for c in CLASSES]
                x = np.concatenate(parts)
                y = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
                correct = clf.predict(x) == y
                boot = [correct[rng.integers(0, len(y), len(y))].mean() for _ in range(N_BOOT)]
                gen = dict(zip(CLASSES, parts))
                own = np.mean([KernelSet(held[c], gen[c]).kid() for c in CLASSES])
                cross = np.mean([KernelSet(held[o], gen[c]).kid() for c, o in zip(CLASSES, CLASSES[::-1])])
                a, b = unit(parts[0]), unit(parts[1])
                rows.append({"run": run, "backbone": backbone, "cfg": cfg, "accuracy": correct.mean(),
                             "acc_ci_low": np.percentile(boot, 2.5), "acc_ci_high": np.percentile(boot, 97.5),
                             "macro_kid_inception": kid.get(cfg, np.nan), "kid_own_class": own,
                             "kid_cross_class": cross, "cos_paired": float(np.sum(a * b, 1).mean()),
                             "cos_unpaired": float((a @ b.T).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "label_vs_cfg.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(out.pivot_table(index=["run", "cfg"], columns="backbone", values="accuracy").to_string())
    print(f"\n  wrote {RESULTS_DIR / 'label_vs_cfg.csv'}")


if __name__ == "__main__":
    main()
