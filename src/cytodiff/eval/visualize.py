"""Stage 4 qualitative evaluation — real vs. synthetic comparison grids (proposal §3.8.2).

Builds a per-class figure: top rows = real eval images, bottom rows = synthetic images from the
chosen model/CFG. Saved to experiments/results/. Selection is seeded for reproducibility.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cytodiff.constants import CLASSES, EXPERIMENTS_DIR, PROJECT_ROOT
from cytodiff.data.split import MANIFEST_PATH

RESULTS_DIR = EXPERIMENTS_DIR / "results"


def _real_paths(df: pd.DataFrame, cls: str) -> list[Path]:
    sub = df[(df["class"] == cls) & (df["eval_reference"])]
    return [PROJECT_ROOT / p for p in sub["path"]]


def _gen_paths(model_name: str, cfg, cls: str) -> list[Path]:
    return sorted((EXPERIMENTS_DIR / "samples" / model_name / f"cfg_{cfg}" / cls).glob("*.png"))


def make_comparison_grid(model_name: str, cfg, n: int = 8, classes=CLASSES, seed: int = 0) -> list[Path]:
    df = pd.read_csv(MANIFEST_PATH)
    rng = random.Random(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for cls in classes:
        real = _real_paths(df, cls)
        gen = _gen_paths(model_name, cfg, cls)
        if not real or not gen:
            print(f"  [skip] {cls}: real={len(real)} gen={len(gen)}")
            continue
        k = min(n, len(real), len(gen))
        real_sel = rng.sample(real, k)
        gen_sel = rng.sample(gen, k)

        fig, axes = plt.subplots(2, k, figsize=(1.6 * k, 3.6))
        for j in range(k):
            axes[0, j].imshow(Image.open(real_sel[j]).convert("RGB"))
            axes[1, j].imshow(Image.open(gen_sel[j]).convert("RGB"))
            for r in (0, 1):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, 0].set_ylabel("REAL", fontsize=11)
        axes[1, 0].set_ylabel("SYNTH", fontsize=11)
        fig.suptitle(f"{model_name}  {cls}  (CFG {cfg})  —  top: real, bottom: synthetic", fontsize=11)
        fig.tight_layout()
        out = RESULTS_DIR / f"grid_{model_name}_cfg{cfg}_{cls}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    return written
