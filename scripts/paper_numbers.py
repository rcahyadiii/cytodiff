#!/usr/bin/env python
"""Every number quoted in the text of the paper (paper_v2.tex) that is not in one of its tables,
recomputed from experiments/results, the cached features, and the stage timings.

Writes docs/paper_numbers.md (quantity, value, source) so that a reader can check each claim.

Usage: PYTHONPATH=src python scripts/paper_numbers.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cytodiff.constants import CLASSES, PROJECT_ROOT
from cytodiff.eval.evaluate import FEATURES_DIR, RESULTS_DIR
from cytodiff.eval.metrics import KernelSet

OUT = PROJECT_ROOT / "docs" / "paper_numbers.md"
TIMINGS = PROJECT_ROOT / "outputs" / "logs" / "timings.csv"


def macro(run: str, backbone: str, cfg: float, col: str) -> float:
    d = pd.read_csv(RESULTS_DIR / f"metrics_{run}.csv")
    return float(d[(d["class"] == "macro") & (d["backbone"] == backbone) & (d["cfg"] == cfg)][col].iloc[0])


def real(backbone: str, col: str) -> float:
    d = pd.read_csv(RESULTS_DIR / "baseline_real.csv")
    return float(d[(d["class"] == "macro") & (d["backbone"] == backbone)][col].iloc[0])


def class_kids(backbone: str) -> tuple[float, float]:
    """Mean KID of real training images against held-out images of the same and of the other class."""
    tr = {c: np.load(FEATURES_DIR / backbone / f"real_train_{c}.npz")["feats"] for c in CLASSES}
    ho = {c: np.load(FEATURES_DIR / backbone / f"real_eval_{c}.npz")["feats"] for c in CLASSES}
    own = np.mean([KernelSet(ho[c], tr[c]).kid() for c in CLASSES])
    cross = np.mean([KernelSet(ho[o], tr[c]).kid() for c, o in zip(CLASSES, CLASSES[::-1])])
    return float(own), float(cross)


def main() -> None:
    rows = []
    add = lambda q, v, s: rows.append((q, v, s))

    for b in ("uni", "phikon"):
        add(f"SD 3 vs. real-vs-real macro KID ({b})", f"{macro('sd3', b, 1.5, 'kid'):.4f} vs. {real(b, 'kid'):.4f}",
            f"metrics_sd3.csv, baseline_real.csv (macro, {b})")
    add("SD 3 recall and coverage on UNI (real-vs-real)",
        f"{macro('sd3', 'uni', 1.5, 'recall'):.4f}, {macro('sd3', 'uni', 1.5, 'coverage'):.4f} "
        f"({real('uni', 'recall'):.4f}, {real('uni', 'coverage'):.4f})", "metrics_sd3.csv, baseline_real.csv (macro, uni)")
    add("SD 3 Inception macro KID at guidance 1, 1.25, 2",
        ", ".join(f"{macro('sd3', 'inception', g, 'kid'):.4f}" for g in (1.0, 1.25, 2.0)), "metrics_sd3.csv (macro, inception)")

    seeds = pd.read_csv(RESULTS_DIR / "seed_variability.csv")
    seeds = seeds[seeds["backbone"] == "inception"]
    for m in ("sd3", "sd21"):
        add(f"{m} Inception macro KID, training seeds 0, 1, 2",
            ", ".join(f"{v:.4f}" for v in seeds[seeds["model"] == m].sort_values("seed")["macro_kid"]), "seed_variability.csv")

    pw = pd.read_csv(RESULTS_DIR / "model_pairwise.csv")
    sd3 = pw[pw["better"] == "sd3"]
    add("Share of paired replicates in which SD 3 has the lower KID (min over rivals and extractors)",
        f"{sd3['p_better_kid'].min():.4f}", "model_pairwise.csv (better = sd3, p_better_kid)")

    lab = pd.read_csv(RESULTS_DIR / "label_vs_cfg.csv")
    lab = lab[(lab["run"] == "sd3") & (lab["backbone"] == "uni")].set_index("cfg")
    add("SD 3 UNI probe accuracy at guidance 1.5, 3, 5, 7.5",
        ", ".join(f"{lab.loc[g, 'accuracy']:.4f}" for g in (1.5, 3.0, 5.0, 7.5)), "label_vs_cfg.csv (sd3, uni)")
    add("SD 3 Inception macro KID at guidance 3, 5, 7.5",
        ", ".join(f"{lab.loc[g, 'macro_kid_inception']:.4f}" for g in (3.0, 5.0, 7.5)), "label_vs_cfg.csv (sd3, uni)")
    worst = max(macro("sd3", "inception", g, "kid") for g in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0))
    ranking = pd.read_csv(RESULTS_DIR / "model_ranking.csv")
    others = ranking[(ranking["backbone"] == "inception") & ranking["model"].isin(["sd15", "sd21", "sdxl"])]["macro_kid"].min()
    add("Largest SD 3 Inception macro KID over guidance 1 to 5 vs. lowest best-scale KID of the other LoRA backbones",
        f"{worst:.4f} vs. {others:.4f}", "metrics_sd3.csv (macro, inception); model_ranking.csv (inception)")
    add("SD 3 at 1.5: UNI cosine similarity of ADC i and SCC i vs. unrelated pairs",
        f"{lab.loc[1.5, 'cos_paired']:.4f} vs. {lab.loc[1.5, 'cos_unpaired']:.4f}", "label_vs_cfg.csv (sd3, uni, 1.5)")
    own_u, cross_u = class_kids("uni")
    add("Cross-class / same-class UNI KID: SD 3 at 1.5 vs. real training images",
        f"{lab.loc[1.5, 'kid_cross_class'] / lab.loc[1.5, 'kid_own_class']:.4f} vs. {cross_u / own_u:.4f}",
        "label_vs_cfg.csv; cached UNI features of real images")
    own_i, cross_i = class_kids("inception")
    add("Inception KID between real ADC and SCC (training vs. held-out), and its ratio to real-vs-real",
        f"{cross_i:.4f} ({cross_i / real('inception', 'kid'):.1f} times)", "cached Inception features; baseline_real.csv")

    chk = pd.read_csv(RESULTS_DIR / "label_check.csv")
    ref = chk[(chk["train_on"] == "real") & (chk["test_on"] == "real_heldout")].set_index("backbone")["accuracy"]
    add("Probe accuracy on held-out real images (UNI, Phikon)", f"{ref['uni']:.4f}, {ref['phikon']:.4f}", "label_check.csv")

    mem = pd.read_csv(RESULTS_DIR / "memorization_summary.csv")
    med = lambda r: float(mem[mem["run"] == r]["lpips_median"].mean())
    add("Median LPIPS to the nearest training image: SD 3 vs. held-out real images (mean of the class medians)",
        f"{med('sd3'):.4f} vs. {med('real_heldout'):.4f}", "memorization_summary.csv")
    pairs = pd.read_csv(RESULTS_DIR / "memorization_pairs.csv")
    flagged = pairs[pairs["below_heldout_min"]]
    for r, g in flagged.groupby("run"):
        add(f"Flagged images of {r} and their largest SSIM to the neighbor", f"{len(g)}, {g['ssim'].max():.4f}",
            "memorization_pairs.csv (below_heldout_min)")

    rank = pd.read_csv(RESULTS_DIR / "model_ranking.csv")
    rank = rank[rank["backbone"] == "inception"].set_index("model")["macro_kid"]
    add("SD 3 macro KID after color transfer (RGB, CIELAB)", f"{rank['sd3_ct_rgb']:.4f}, {rank['sd3_ct_lab']:.4f}",
        "model_ranking.csv (inception)")
    add("GPU hours of the whole study", f"{pd.read_csv(TIMINGS)['seconds'].sum() / 3600:.1f}", "outputs/logs/timings.csv")

    OUT.parent.mkdir(exist_ok=True)
    lines = ["# Numbers quoted in the text of the paper", "",
             "Generated by `scripts/paper_numbers.py` from the result tables in the Zenodo artifacts",
             "(`results/`), the cached features, and the stage timings, with four decimals; the paper rounds these half up.", "",
             "| Quantity | Value | Source |", "|---|---|---|"]
    lines += [f"| {q} | {v} | {s} |" for q, v, s in rows]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
