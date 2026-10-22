#!/usr/bin/env python
"""Memorization check: how close is each synthetic image to its nearest TRAINING image?

For every synthetic image (each run at its best CFG), the 5 nearest training images of the same
class are found by cosine similarity of pathology-foundation-model features (UNI by default);
among those candidates the closest by LPIPS (AlexNet) is kept, and SSIM is reported for that pair.
The same is done for every held-out real image: that gives the similarity between DIFFERENT
patients' real tissue and the training set, the baseline a non-memorizing generator should not
beat. A synthetic image whose LPIPS to its nearest training image is below the smallest value
seen for any held-out real image is flagged as a possible copy.

Outputs in experiments/results/:
  memorization_pairs.csv      one row per query image (synthetic or held-out real)
  memorization_summary.csv    per run and class: LPIPS/SSIM/cosine quantiles, flagged count
  memorization_closest_<run>.png   the 8 closest synthetic/training pairs of each run

Usage:
    venv/bin/python scripts/memorization.py --runs sd15 sd21 sdxl sd3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from cytodiff.constants import CLASSES, PROJECT_ROOT
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import RESULTS_DIR, best_cfg_of, gen_features, gen_paths, real_features, real_set
from cytodiff.eval.metrics import default_device, load_eval_image

SIDE = 256          # LPIPS / SSIM resolution
TOP_K = 5


def small(path: Path) -> np.ndarray:
    return np.asarray(load_eval_image(path).resize((SIDE, SIDE), Image.Resampling.BICUBIC), dtype=np.uint8)


def to_lpips_tensor(arrs: list[np.ndarray], device: str) -> torch.Tensor:
    x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x.to(device)


def nearest(query_feats, query_paths, train_feats, train_imgs, train_paths, train_pat, lpips_net, device):
    q = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
    t = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)
    sims = q @ t.T
    rows = []
    for i, path in enumerate(query_paths):
        cand = np.argsort(-sims[i])[:TOP_K]
        qa = small(path)
        with torch.no_grad():
            d = lpips_net(to_lpips_tensor([qa] * len(cand), device),
                          to_lpips_tensor([train_imgs[c] for c in cand], device)).flatten().cpu().numpy()
        best = cand[int(np.argmin(d))]
        rows.append({"image": str(Path(path).relative_to(PROJECT_ROOT)),
                     "nn_train": str(Path(train_paths[best]).relative_to(PROJECT_ROOT)),
                     "nn_patient": train_pat[best], "cosine": float(sims[i, best]),
                     "lpips": float(d.min()),
                     "ssim": float(structural_similarity(qa, train_imgs[best], channel_axis=2, data_range=255))})
    return rows


def closest_grid(pairs: pd.DataFrame, out: Path, n: int = 8) -> None:
    top = pairs.nsmallest(n, "lpips")
    tile = 160
    grid = Image.new("RGB", (tile * len(top), tile * 2 + 18), "white")
    draw = ImageDraw.Draw(grid)
    for k, (_, r) in enumerate(top.iterrows()):
        grid.paste(load_eval_image(PROJECT_ROOT / r["image"]).resize((tile, tile)), (k * tile, 0))
        grid.paste(load_eval_image(PROJECT_ROOT / r["nn_train"]).resize((tile, tile)), (k * tile, tile))
        draw.text((k * tile + 4, 2 * tile + 3), f"LPIPS {r['lpips']:.3f}", fill="black")
    grid.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Nearest-training-image memorization check.")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--search-backbone", default="uni", choices=["uni", "phikon", "inception"])
    args = ap.parse_args()

    import lpips
    device = default_device()
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    df = pd.read_csv(MANIFEST_PATH)
    rows = []
    for cls in CLASSES:
        train_paths, train_pat = real_set(df, cls, "train")
        train_feats, _ = real_features(df, cls, device, args.search_backbone, "train")
        train_imgs = [small(p) for p in train_paths]
        held_paths, _ = real_set(df, cls, "eval")
        held_feats, _ = real_features(df, cls, device, args.search_backbone, "eval")
        for r in nearest(held_feats, held_paths, train_feats, train_imgs, train_paths, train_pat, lpips_net, device):
            rows.append({"run": "real_heldout", "cfg": float("nan"), "class": cls, **r})
        for run in args.runs:
            cfg = best_cfg_of(run)
            paths = gen_paths(run, cfg, cls)
            feats = gen_features(run, cfg, cls, device, args.search_backbone)
            for r in nearest(feats, paths, train_feats, train_imgs, train_paths, train_pat, lpips_net, device):
                rows.append({"run": run, "cfg": cfg, "class": cls, **r})
            print(f"  {run} {cls}: done")

    pairs = pd.DataFrame(rows)
    floor = pairs[pairs["run"] == "real_heldout"].groupby("class")["lpips"].min()
    pairs["below_heldout_min"] = pairs.apply(lambda r: r["run"] != "real_heldout" and r["lpips"] < floor[r["class"]], axis=1)
    summary = pairs.groupby(["run", "class"]).agg(
        n=("lpips", "size"), lpips_median=("lpips", "median"), lpips_p05=("lpips", lambda x: x.quantile(0.05)),
        lpips_min=("lpips", "min"), ssim_median=("ssim", "median"), ssim_p95=("ssim", lambda x: x.quantile(0.95)),
        ssim_max=("ssim", "max"), cosine_median=("cosine", "median"), cosine_max=("cosine", "max"),
        flagged=("below_heldout_min", "sum")).reset_index()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(RESULTS_DIR / "memorization_pairs.csv", index=False)
    summary.to_csv(RESULTS_DIR / "memorization_summary.csv", index=False)
    for run in args.runs:
        closest_grid(pairs[pairs["run"] == run], RESULTS_DIR / f"memorization_closest_{run}.png")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(summary.to_string(index=False))
    print(f"\n  wrote memorization_pairs.csv, memorization_summary.csv and closest-pair figures in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
