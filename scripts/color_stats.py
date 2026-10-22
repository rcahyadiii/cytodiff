#!/usr/bin/env python
"""Color statistics of real and generated image sets (Table VII of the paper).

between-image spread : standard deviation across images of each image's mean RGB color,
                       averaged over the three channels
within-image contrast: mean over images of the per-image standard deviation of pixel values
Images are downsampled to 256x256 first. A set is 'real' (held-out images of the class),
'real_all' (every image of the class), or '<model>:<cfg>' for generated samples.

Usage:
    venv/bin/python scripts/color_stats.py --cls ADC --sets real real_all sd3:5 sd3_ct_rgb:5 sd3_ct_lab:5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from PIL import Image

from cytodiff.constants import PROJECT_ROOT
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import RESULTS_DIR, gen_paths


def stats(paths) -> tuple[float, float]:
    means, stds = [], []
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.float64)
        means.append(a.mean((0, 1)))
        stds.append(a.std((0, 1)).mean())
    return float(np.std(means, axis=0).mean()), float(np.mean(stds))


def main() -> None:
    ap = argparse.ArgumentParser(description="Color statistics of image sets.")
    ap.add_argument("--cls", required=True, choices=["ADC", "SCC"])
    ap.add_argument("--sets", nargs="+", required=True)
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST_PATH)
    rows = []
    for s in args.sets:
        if s in ("real", "real_all"):
            sub = df[df["class"] == args.cls]
            sub = sub[sub["eval_reference"]] if s == "real" else sub
            paths = [PROJECT_ROOT / p for p in sub["path"]]
        else:
            model, cfg = s.split(":")
            paths = gen_paths(model, float(cfg), args.cls)
        if not paths:
            raise SystemExit(f"no images for set {s!r}")
        spread, contrast = stats(paths)
        rows.append({"set": s, "class": args.cls, "n": len(paths),
                     "between_image_spread": spread, "within_image_contrast": contrast})
        print(f"  {s:16s} n={len(paths):3d}  spread={spread:5.1f}  contrast={contrast:5.1f}")
    out = RESULTS_DIR / f"color_stats_{args.cls}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
