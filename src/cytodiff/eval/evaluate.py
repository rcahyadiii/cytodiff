"""Stage 4b/5 — evaluate a run across the CFG sweep and pick its best operating point.

For each CFG, class and feature extractor (inception, uni, phikon), generated images are compared
with the held-out real images of that class (patient-level split). Per row of
experiments/results/metrics_<run>.csv:
    kid, kid_ci_low, kid_ci_high   unbiased full-set KID with a 95 % normal bootstrap interval
                                   (patient-cluster resampling of the real images)
    fd                             Frechet distance (= FID for inception); secondary
    precision, recall, density, coverage
plus 'macro' rows (equal-weight mean of the two classes; KID interval from the joint bootstrap).
The best CFG is chosen on the PRIMARY backbone's (inception) macro KID, fixed in advance. No pooled
set is formed, because real and generated class proportions differ.

evaluate_real_baseline() scores the real TRAINING images against the held-out ones the same way,
giving the value a perfect generator could reach (experiments/results/baseline_real.csv).
Features are cached under experiments/features/<backbone>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cytodiff.constants import CFG_SWEEP, CLASSES, EXPERIMENTS_DIR, PROJECT_ROOT
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.metrics import (
    BACKBONES, PRIMARY_BACKBONE, KernelSet, bootstrap_kid, cluster_resamples, default_device,
    extract_features, frechet_distance, image_resamples, normal_ci, prdc,
)

RESULTS_DIR = EXPERIMENTS_DIR / "results"
FEATURES_DIR = EXPERIMENTS_DIR / "features"
N_BOOT = 1000
SEED = 0


def real_set(df: pd.DataFrame, cls: str, split: str = "eval") -> tuple[list[Path], np.ndarray]:
    sub = df[(df["class"] == cls) & (df["split"] == split)]
    return [PROJECT_ROOT / p for p in sub["path"]], sub["patient_id"].astype(str).to_numpy()


def real_reference(df: pd.DataFrame, cls: str) -> tuple[list[Path], np.ndarray]:
    return real_set(df, cls, "eval")


def gen_paths(run: str, cfg: float, cls: str) -> list[Path]:
    # Sweeps write cfg_5 / cfg_7.5; older single-CFG runs wrote cfg_5.0. Accept both.
    for tag in dict.fromkeys((f"{cfg:g}", str(float(cfg)))):
        d = EXPERIMENTS_DIR / "samples" / run / f"cfg_{tag}" / cls
        if d.is_dir():
            return sorted(d.glob("*.png"))
    return []


def cached_features(paths: list[Path], cache: Path, backbone: str, device: str) -> np.ndarray:
    """Features for `paths`, reusing `cache` only if it was built from the same files, unchanged
    (same names, sizes, and modification times), so regenerated samples are never mis-scored."""
    names = np.array([str(p.relative_to(PROJECT_ROOT)) for p in paths])
    stamps = np.array([(p.stat().st_size, p.stat().st_mtime_ns) for p in paths], dtype=np.int64)
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["names"], names) and "stamps" in z and np.array_equal(z["stamps"], stamps):
            return z["feats"]
    feats = extract_features(paths, backbone=backbone, device=device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feats=feats, names=names, stamps=stamps)
    return feats


def real_features(df: pd.DataFrame, cls: str, device: str, backbone: str = PRIMARY_BACKBONE,
                  split: str = "eval") -> tuple[np.ndarray, np.ndarray]:
    paths, patients = real_set(df, cls, split)
    cache = FEATURES_DIR / backbone / f"real_{split}_{cls}.npz"
    return cached_features(paths, cache, backbone, device), patients


def gen_features(run: str, cfg: float, cls: str, device: str,
                 backbone: str = PRIMARY_BACKBONE) -> np.ndarray | None:
    paths = gen_paths(run, cfg, cls)
    if not paths:
        return None
    return cached_features(paths, FEATURES_DIR / backbone / run / f"cfg_{cfg:g}_{cls}.npz", backbone, device)


def score(fr: np.ndarray, real_groups: np.ndarray, fg: np.ndarray, real_idx: list[np.ndarray],
          gen_seed: int, n_boot: int, gen_groups: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """All metrics for one (real, generated) pair; returns (row fields, KID bootstrap draws)."""
    ks = KernelSet(fr, fg)
    gen_idx = (cluster_resamples(gen_groups, n_boot, gen_seed) if gen_groups is not None
               else image_resamples(len(fg), n_boot, gen_seed))
    boots = bootstrap_kid(ks, real_idx, gen_idx)
    kid = ks.kid()
    lo, hi = normal_ci(kid, boots)
    row = {"kid": kid, "kid_ci_low": lo, "kid_ci_high": hi, "fd": frechet_distance(fr, fg),
           **prdc(fr, fg), "n_real": len(fr), "n_real_patients": len(set(real_groups)), "n_gen": len(fg)}
    return row, boots


def _macro(rows: list[dict], boots: dict[str, np.ndarray], classes, base: dict) -> dict:
    by_cls = {r["class"]: r for r in rows}
    draws = np.mean([boots[c] for c in classes], axis=0)
    out = {**base, "class": "macro"}
    for k in ("kid", "fd", "precision", "recall", "density", "coverage"):
        out[k] = float(np.mean([by_cls[c][k] for c in classes]))
    out["kid_ci_low"], out["kid_ci_high"] = normal_ci(out["kid"], draws)
    for k in ("n_real", "n_real_patients", "n_gen"):
        out[k] = int(sum(by_cls[c][k] for c in classes))
    return out


def evaluate_model(
    run: str,
    cfgs: tuple[float, ...] = CFG_SWEEP,
    classes: tuple[str, ...] = CLASSES,
    backbones: tuple[str, ...] = BACKBONES,
    device: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    device = device or default_device()
    df = pd.read_csv(MANIFEST_PATH)
    rows = []
    for backbone in backbones:
        real = {c: real_features(df, c, device, backbone) for c in classes}
        # Real resamples depend only on class and seed: every run and CFG is scored against the
        # same bootstrap replicates of the reference set.
        real_idx = {c: cluster_resamples(real[c][1], n_boot, seed + k) for k, c in enumerate(classes)}
        for cfg in cfgs:
            cls_rows, boots = [], {}
            for k, cls in enumerate(classes):
                fg = gen_features(run, cfg, cls, device, backbone)
                if fg is None:
                    print(f"  [skip] no samples for {run} cfg={cfg:g} {cls}")
                    continue
                fields, boots[cls] = score(real[cls][0], real[cls][1], fg, real_idx[cls], seed + 100 + k, n_boot)
                row = {"model": run, "cfg": cfg, "class": cls, "backbone": backbone, **fields}
                cls_rows.append(row)
                print(f"  {run} cfg={cfg:g} {cls} [{backbone}]: KID={row['kid']:.4f} "
                      f"[{row['kid_ci_low']:.4f}, {row['kid_ci_high']:.4f}] FD={row['fd']:.1f} "
                      f"P={row['precision']:.2f} R={row['recall']:.2f} D={row['density']:.2f} C={row['coverage']:.2f}")
            rows += cls_rows
            if len(cls_rows) == len(classes):
                m = _macro(cls_rows, boots, classes, {"model": run, "cfg": cfg, "backbone": backbone})
                rows.append(m)
                print(f"  {run} cfg={cfg:g} MACRO [{backbone}]: KID={m['kid']:.4f} [{m['kid_ci_low']:.4f}, {m['kid_ci_high']:.4f}]")

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit(f"No samples found for {run}. Run scripts/generate.py for these CFGs first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = RESULTS_DIR / f"metrics_{run}.csv"
    out.to_csv(csv, index=False)
    print(f"  wrote {csv}")
    best = best_cfg(out)
    if best is not None:
        print(f"  >> best CFG for {run}: {best:g} (macro KID, {PRIMARY_BACKBONE})")
    return out


def best_cfg(metrics: pd.DataFrame) -> float | None:
    """CFG with the lowest macro KID on the primary backbone."""
    m = metrics[(metrics["class"] == "macro") & (metrics["backbone"] == PRIMARY_BACKBONE)]
    return None if m.empty else float(m.loc[m["kid"].idxmin(), "cfg"])


def best_cfg_of(run: str) -> float:
    cfg = best_cfg(pd.read_csv(RESULTS_DIR / f"metrics_{run}.csv"))
    if cfg is None:
        raise ValueError(f"metrics_{run}.csv has no macro rows for the primary backbone "
                         f"({PRIMARY_BACKBONE}); evaluate {run} with it first.")
    return cfg


def evaluate_real_baseline(
    classes: tuple[str, ...] = CLASSES,
    backbones: tuple[str, ...] = BACKBONES,
    device: str | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Real training images vs held-out real images: the floor a perfect generator could reach."""
    device = device or default_device()
    df = pd.read_csv(MANIFEST_PATH)
    rows = []
    for backbone in backbones:
        cls_rows, boots = [], {}
        for k, cls in enumerate(classes):
            fr, gr = real_features(df, cls, device, backbone, "eval")
            ft, gt = real_features(df, cls, device, backbone, "train")
            real_idx = cluster_resamples(gr, n_boot, seed + k)
            fields, boots[cls] = score(fr, gr, ft, real_idx, seed + 200 + k, n_boot, gen_groups=gt)
            cls_rows.append({"model": "real_train", "cfg": float("nan"), "class": cls, "backbone": backbone, **fields})
            print(f"  real train vs held-out {cls} [{backbone}]: KID={fields['kid']:.4f} FD={fields['fd']:.1f}")
        rows += cls_rows + [_macro(cls_rows, boots, classes, {"model": "real_train", "cfg": float("nan"),
                                                              "backbone": backbone})]
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "baseline_real.csv", index=False)
    print(f"  wrote {RESULTS_DIR / 'baseline_real.csv'}")
    return out
