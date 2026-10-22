"""Stage 6 — histogram-matching color transfer (proposal §3.7).

Per-channel CDF matching of each generated image to a class-matched REAL reference, to pull the
synthetic palette back toward the authentic H&E staining. Two colour spaces are offered
(RGB and CIELAB); the lower-FID one is kept. Reference selection follows §3.7.3: a single fixed
reference per class, or a fresh random reference per image.

Leakage note: colour references are drawn from the TRAIN split (manifest split=="train"), while
KID/FID are computed against the held-out eval_reference rows. With the patient-level split these
are disjoint for both classes (different patients).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from skimage.exposure import match_histograms

from cytodiff.constants import CLASSES, EXPERIMENTS_DIR, PROJECT_ROOT
from cytodiff.data.split import MANIFEST_PATH

SAMPLES_DIR = EXPERIMENTS_DIR / "samples"


def match_one(src_rgb: np.ndarray, ref_rgb: np.ndarray, space: str = "rgb") -> np.ndarray:
    """Match src's colour distribution to ref. Both uint8 HxWx3 RGB; returns uint8 RGB."""
    if space == "rgb":
        out = match_histograms(src_rgb, ref_rgb, channel_axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)
    if space == "lab":
        s = skcolor.rgb2lab(src_rgb / 255.0)
        r = skcolor.rgb2lab(ref_rgb / 255.0)
        m = match_histograms(s, r, channel_axis=-1)
        return np.clip(skcolor.lab2rgb(m) * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown colour space: {space!r}")


def _train_refs(df: pd.DataFrame, cls: str) -> list[Path]:
    sub = df[(df["class"] == cls) & (df["split"] == "train")]
    return [PROJECT_ROOT / p for p in sub["path"]]


def _find_cfg_dir(model_name: str, cfg: float, cls: str) -> Path | None:
    # Directory naming is inconsistent across the codebase: the full CFG_SWEEP
    # writes integer-valued CFGs as `cfg_5` while `--cfg 5` writes `cfg_5.0`.
    # Try both so colour transfer finds samples regardless of how they were made.
    for tag in (str(int(cfg)) if float(cfg).is_integer() else None, str(cfg)):
        if tag is None:
            continue
        d = SAMPLES_DIR / model_name / f"cfg_{tag}" / cls
        if d.is_dir() and any(d.glob("*.png")):
            return d
    return None


def apply_to_model(
    model_name: str,
    cfg: float,
    space: str = "rgb",
    strategy: str = "random_per_image",
    seed: int = 0,
    out_model: str | None = None,
) -> str:
    """Colour-transfer every generated image for (model, cfg). Writes a new sample tree whose
    model name is out_model (default f'{model}_ct_{space}'), so the standard evaluate.py can score
    it directly. Returns that new model name."""
    df = pd.read_csv(MANIFEST_PATH)
    out_model = out_model or f"{model_name}_ct_{space}"
    rng = random.Random(seed)

    for cls in CLASSES:
        src_dir = _find_cfg_dir(model_name, cfg, cls)
        if src_dir is None:
            print(f"  [skip] no samples for {model_name} cfg={cfg} {cls}")
            continue
        gen = sorted(src_dir.glob("*.png"))
        refs = _train_refs(df, cls)
        if not refs:
            raise SystemExit(f"no train references for class {cls}")
        fixed_ref = np.asarray(Image.open(refs[0]).convert("RGB")) if strategy == "fixed" else None

        # Write in float-style (cfg_5.0) so `evaluate.py --model <out> --cfg 5` matches.
        out_dir = SAMPLES_DIR / out_model / f"cfg_{cfg}" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for g in gen:
            src = np.asarray(Image.open(g).convert("RGB"))
            ref = fixed_ref if fixed_ref is not None else np.asarray(
                Image.open(rng.choice(refs)).convert("RGB")
            )
            # references are 512 masters; match generated size for a per-pixel-safe transfer
            if ref.shape[:2] != src.shape[:2]:
                ref = np.asarray(Image.fromarray(ref).resize(src.shape[1::-1], Image.BICUBIC))
            out = match_one(src, ref, space=space)
            Image.fromarray(out).save(out_dir / g.name)
        print(f"  {out_model} cfg={cfg} {cls}: {len(gen)} images "
              f"({space}, {strategy}) -> {out_dir}")
    return out_model
