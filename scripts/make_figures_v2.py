#!/usr/bin/env python
"""Figures for the v2 paper (reports/ieee_paper/), from experiments/results and the sample images.

  fig_cfg_v2.pdf   (a) Inception macro KID across guidance scales, LoRA (solid) and zero-shot
                   (dashed) runs, with the real-versus-real value; (b) label fidelity of the LoRA
                   runs (UNI linear probe trained on real images) across guidance scales.
  fig_qual_v2.jpg  held-out real images and the first two generated images per class of each
                   model (not hand-picked): SD 3 zero-shot, the four LoRA models at their best
                   scale, and SD 3 at guidance 3 (paper_v2_long.tex).
  fig_qual_v2_short.jpg  the same with one image per class (paper_v2.tex, 8 pages).

Usage: python scripts/make_figures_v2.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cytodiff.constants import PROJECT_ROOT
from cytodiff.eval.evaluate import RESULTS_DIR, best_cfg_of, gen_paths

OUT = PROJECT_ROOT / "reports" / "ieee_paper"
MODELS = {"sd15": "SD v1.5", "sd21": "SD v2.1", "sdxl": "SDXL", "sd3": "SD 3"}
COLORS = {"sd15": "tab:blue", "sd21": "tab:orange", "sdxl": "tab:green", "sd3": "tab:red"}
TICKS = [1, 1.5, 2, 3, 5, 7.5, 10, 20]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})


def macro(run: str, backbone: str = "inception") -> pd.Series:
    d = pd.read_csv(RESULTS_DIR / f"metrics_{run}.csv")
    return d[(d["class"] == "macro") & (d["backbone"] == backbone)].set_index("cfg")["kid"].sort_index()


def fig_cfg() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.35), constrained_layout=True)
    real = pd.read_csv(RESULTS_DIR / "baseline_real.csv")
    real_kid = real[(real["class"] == "macro") & (real["backbone"] == "inception")]["kid"].iloc[0]
    for m, name in MODELS.items():
        lora, zs = macro(m), macro(f"{m}_zeroshot")
        a.plot(lora.index, lora.values, "-o", ms=2.5, lw=1.1, color=COLORS[m], label=f"{name} LoRA")
        a.plot(zs.index, zs.values, ls=(0, (3, 1.5)), marker="x", ms=3, mew=0.8, lw=0.9, color=COLORS[m],
               alpha=0.85, label=f"{name} zero-shot")
        best = best_cfg_of(m)
        a.plot([best], [lora[best]], "*", ms=8, color=COLORS[m], mec="black", mew=0.4)
    a.axhline(real_kid, color="black", ls=":", lw=0.9, label="real vs. real")
    a.set(xscale="log", yscale="log", xlabel="Guidance scale", ylabel="Macro KID (Inception)")
    a.set_xticks(TICKS, [f"{t:g}" for t in TICKS]); a.minorticks_off()
    a.set_title("(a) Distribution distance (lower is better)", fontsize=8)
    a.legend(fontsize=6, ncol=2, frameon=False, loc="lower right", handlelength=2.6)

    lab = pd.read_csv(RESULTS_DIR / "label_vs_cfg.csv")
    lab = lab[lab["backbone"] == "uni"]
    chk = pd.read_csv(RESULTS_DIR / "label_check.csv")
    real_acc = chk[(chk["backbone"] == "uni") & (chk["test_on"] == "real_heldout") & (chk["train_on"] == "real")]["accuracy"].iloc[0]
    for m, name in MODELS.items():
        d = lab[lab["run"] == m].sort_values("cfg")
        b.plot(d["cfg"], d["accuracy"], "-o", ms=2.5, lw=1.1, color=COLORS[m], label=name)
        b.fill_between(d["cfg"], d["acc_ci_low"], d["acc_ci_high"], color=COLORS[m], alpha=0.12, lw=0)
    b.axhline(real_acc, color="black", ls=":", lw=0.9, label="held-out real images")
    b.axhline(0.5, color="gray", ls="-", lw=0.5, label="chance")
    b.set(xscale="log", xlabel="Guidance scale", ylabel="Probe accuracy on generated images", ylim=(0.4, 1.02))
    b.set_xticks(TICKS, [f"{t:g}" for t in TICKS]); b.minorticks_off()
    b.set_title("(b) Label fidelity of LoRA models (higher is better)", fontsize=8)
    b.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
    fig.savefig(OUT / "fig_cfg_v2.pdf")
    print("  wrote", OUT / "fig_cfg_v2.pdf")


def fig_qual(real_paths: dict[str, list[str]], per_class: int = 2, name: str = "fig_qual_v2.jpg") -> None:
    cols = [("Real\n(held-out)", None, None), ("SD 3 zero-shot", "sd3_zeroshot", None)]
    cols += [(f"{name} LoRA", m, None) for m, name in MODELS.items()]
    cols += [("SD 3 LoRA", "sd3", 3.0)]
    rows = [(c, k) for c in ("ADC", "SCC") for k in range(per_class)]
    tile = 256
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.16, 0.25 + 1.0 * len(rows)),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r, (cls, k) in enumerate(rows):
        for c, (title, run, cfg) in enumerate(cols):
            if run is None:
                path = PROJECT_ROOT / real_paths[cls][k]
            else:
                path = gen_paths(run, cfg if cfg is not None else best_cfg_of(run), cls)[k]
            ax = axes[r, c]
            ax.imshow(Image.open(path).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                label = title if run is None else f"{title}\nCFG {cfg if cfg is not None else best_cfg_of(run):g}"
                ax.set_title(label.replace(" LoRA, CFG 3", " LoRA"), fontsize=6.3, linespacing=1.1)
            if c == 0:
                ax.set_ylabel(cls, fontsize=7)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 90})
    print("  wrote", OUT / name)


if __name__ == "__main__":
    fig_cfg()
    real = {"ADC": ["data/processed/ADC/aca_bd_20x_301.png", "data/processed/ADC/aca_bd_40x_301.png"],
            "SCC": ["data/processed/SCC/scc_bd_20x_301.png", "data/processed/SCC/scc_bd_40x_301.png"]}
    fig_qual(real)
    fig_qual({c: v[:1] for c, v in real.items()}, per_class=1, name="fig_qual_v2_short.jpg")
