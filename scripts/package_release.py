#!/usr/bin/env python
"""Build the public release from the v2 results (after RERUN.md step 4).

Creates release/ with:
  cytodiff_code.zip             -> GitHub: src/, scripts/, configs/, requirements, README, RERUN.md, LICENSE
                                   (the same files as github.com/rcahyadiii/cytodiff; see code_files())
  zenodo/README.md              -> Zenodo dataset card: contents, which images to use, ranking,
                                   generation settings, intended use, licenses (numbers read from
                                   experiments/results, never typed by hand)
  zenodo/cytodiff_v2_artifacts.zip  every LoRA adapter, all result tables and figures, the
                                   patient-level split
  zenodo/LICENSES.md, zenodo/CITATION.cff
The generated images themselves (all runs, about 39 GB) stay on the GPU box and are zipped and
uploaded to the same Zenodo record by scripts/cloud/zenodo_upload.py, together with release/zenodo/.

--anonymous drops author names and e-mail addresses for double-blind review
(e.g. upload the code archive to anonymous.4open.science).

Usage:
    python scripts/package_release.py [--anonymous] [--dry-run] [--code-license MIT] [--github URL] [--cards-only]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RESULTS = EXP / "results"
OUT = ROOT / "release"
TITLE = "A Comparison of Stable Diffusion Backbones With and Without LoRA Fine-Tuning for Synthetic H&E Lung Cancer Histopathology"
DATASET_TITLE = ("Synthetic H&E lung adenocarcinoma and squamous cell carcinoma images from four Stable Diffusion "
                 "backbones, with and without LoRA fine-tuning (LungHist700)")
AUTHORS = [("Cahyadi", "Muhammad Ridwan"), ("Wijaya", "Rifki")]
ZENODO_DOI = "10.5281/zenodo.23520698"   # reserved on the draft record; resolves once it is published
GITHUB_URL = "https://github.com/rcahyadiii/cytodiff"
NAMES = {"sd15": "SD v1.5", "sd21": "SD v2.1", "sdxl": "SDXL", "sd3": "SD 3 Medium"}
REPOS = {"sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
         "sd21": "SfinOe/stable-diffusion-v2-1 (mirror of stabilityai/stable-diffusion-2-1)",
         "sdxl": "stabilityai/stable-diffusion-xl-base-1.0", "sd3": "stabilityai/stable-diffusion-3-medium-diffusers"}
RES = {"sd15": 512, "sd21": 768, "sdxl": 1024, "sd3": 1024}
BATCH = {"sd15": 8, "sd21": 8, "sdxl": 4, "sd3": 2}

LICENSES = """# Licenses of the released components

| Component | License / terms |
|---|---|
| Code (github.com/rcahyadiii/cytodiff, `cytodiff_code.zip`) | {code_license} |
| Real training data | LungHist700 (Diosdado et al., Sci. Data 11, 1088, 2024; doi:10.6084/m9.figshare.25459174), CC BY 4.0; not redistributed here |
| SD v1.5 adapter and images | Subject to the CreativeML Open RAIL-M license of the base model |
| SD v2.1 and SDXL adapters and images | Subject to the CreativeML Open RAIL++-M license of the base models |
| SD 3 Medium adapters and images | Subject to the Stability AI license of Stable Diffusion 3 Medium |
| Synthetic images, result tables, split | Derived from LungHist700 (CC BY 4.0, attribution required); the terms of the generating model's license apply |

The synthetic images are for research and visualization. They are not validated by pathologists
and must not be used for clinical diagnosis, or for teaching without expert validation.
"""


def group_of(run: str) -> tuple[str, str]:
    """('lora' | 'zeroshot' | 'seeds' | 'colortransfer', base model) for a run name."""
    base = run.split("_")[0]
    if run.endswith("_zeroshot"):
        return "zeroshot", base
    if "_ct_" in run:
        return "colortransfer", base
    if run.rsplit("_", 1)[-1] in ("s1", "s2"):
        return "seeds", base
    return "lora", base


def zip_name(run: str) -> str:
    """images_<group>_<model>[_<variant>].zip, e.g. images_zeroshot_sd15.zip, images_colortransfer_sd3_rgb.zip.
    Must match scripts/cloud/zenodo_upload.py."""
    group, base = group_of(run)
    rest = run[len(base):].strip("_").replace("zeroshot", "").replace("ct_", "").strip("_")
    return f"images_{group}_{base}" + (f"_{rest}" if rest else "") + ".zip"


def macro(run: str) -> pd.DataFrame:
    m = pd.read_csv(RESULTS / f"metrics_{run}.csv")
    return m[(m["class"] == "macro") & (m["backbone"] == "inception")].set_index("cfg")


def image_counts() -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS.glob("metrics_*.csv")):
        run = p.stem.removeprefix("metrics_")
        m = pd.read_csv(p)
        m = m[(m["backbone"] == "inception") & m["class"].isin(["ADC", "SCC"])]
        group, base = group_of(run)
        rows.append({"run": run, "group": group, "model": base, "cfgs": m["cfg"].nunique(), "images": int(m["n_gen"].sum())})
    return pd.DataFrame(rows)


def dataset_card(anonymous: bool, github: str) -> str:
    ranking = pd.read_csv(RESULTS / "model_ranking.csv")
    ranking = ranking[ranking["backbone"] == "inception"]
    lab = pd.read_csv(RESULTS / "label_vs_cfg.csv")
    lab = lab[lab["backbone"] == "uni"].set_index(["run", "cfg"])["accuracy"]
    real = pd.read_csv(RESULTS / "baseline_real.csv")
    real = real[(real["class"] == "macro") & (real["backbone"] == "inception")].iloc[0]
    chk = pd.read_csv(RESULTS / "label_check.csv")
    real_acc = chk[(chk["backbone"] == "uni") & (chk["test_on"] == "real_heldout") & (chk["train_on"] == "real")]["accuracy"].iloc[0]
    counts = image_counts()
    best = ranking[~ranking["baseline_or_variant"]].iloc[0]
    curve = macro(best["model"])

    def pct(x):   # round half up, as in the paper (0.585 -> 59 %)
        return f"{int(100 * x + 0.5 + 1e-9)} %" if pd.notna(x) else ""

    rank_rows = "\n".join(
        f"| {r['model']} | {r['cfg']:g} | {r['macro_kid']:.3f} ({r['kid_ci_low']:.3f} to {r['kid_ci_high']:.3f}) | "
        f"{r['macro_fd']:.1f} | {pct(lab.get((r['model'], r['cfg'])))} |"
        for _, r in ranking[~ranking["model"].str.contains("_ct_")].iterrows())
    guidance_rows = "\n".join(f"| {c:g} | {row['kid']:.3f} | {pct(lab.get((best['model'], c)))} |" for c, row in curve.iterrows())
    count_rows = "\n".join(f"| `{zip_name(r.run)}` | {r.group} | {NAMES.get(r.model, r.model)} | {r.cfgs} | {r.images:,} |"
                           for r in counts.sort_values(["group", "run"]).itertuples())
    models = "\n".join(f"| {NAMES[m]} | `{REPOS[m]}` | {RES[m]} | {BATCH[m]} |" for m in NAMES)
    authors = "Anonymous" if anonymous else " and ".join(f"{g} {f}" for f, g in AUTHORS)
    return f"""# {DATASET_TITLE}

Companion data of the paper *{TITLE}* ({authors}). Dataset DOI: [{ZENODO_DOI}](https://doi.org/{ZENODO_DOI}).
Code: [{github.removeprefix('https://')}]({github}).
{counts['images'].sum():,} synthetic H&E images of lung adenocarcinoma (ADC) and squamous cell carcinoma (SCC),
generated by four Stable Diffusion backbones with LoRA fine-tuning and without it (zero-shot), at every
guidance scale tested, plus extra training seeds and color-transferred variants. **Not validated by
pathologists. For research only; never for clinical diagnosis.**

## Which images to use
- **Closest to real tissue:** `lora/{best['model']}/cfg_{best['cfg']:g}` (lowest KID against held-out patients). At this
  low guidance the images are only weakly class-specific: a linear probe that recognizes the class of
  {pct(real_acc)} of held-out real images recognizes {pct(lab.get((best['model'], best['cfg'])))} of them.
- **When the class matters:** `lora/{best['model']}/cfg_3` or `cfg_5`, with a stronger class signal at a KID that is
  still below the best of every other model (see the table below).
- ADC and SCC images with the same index start from the same noise, so at low guidance they look alike.

{NAMES.get(best['model'], best['model'])} with LoRA across guidance scales (Inception macro KID; label fidelity = UNI linear probe):

| Guidance | Macro KID | Label fidelity |
|---|---|---|
{guidance_rows}

## Ranking (each run at its best guidance scale)
Inception macro KID (mean of ADC and SCC) against images of held-out patients, with 95 % normal
bootstrap interval (patients resampled); real training vs. held-out images: {real['kid']:.3f}.

| Run | Guidance | Macro KID (95 % CI) | Macro FID | Label fidelity |
|---|---|---|---|---|
{rank_rows}

Full per-class results on Inception, UNI, and Phikon features, paired tests, seeds, memorization,
label fidelity, and autoencoder fidelity are in `cytodiff_v2_artifacts.zip` (`results/`).

## Contents
| File | Group | Model | Guidance scales | Images |
|---|---|---|---|---|
{count_rows}

- Inside each zip: `<group>/<run>/cfg_<guidance>/<ADC|SCC>/<class>_<index>.png`. Guidance 1 means no
  classifier-free guidance. Groups: `lora` (LoRA, training seed 0), `zeroshot` (base model, no LoRA),
  `seeds` (training seeds 1 and 2 of the two best models, at their best scale), `colortransfer`
  (histogram matching to a random real training image of the same class, RGB or CIELAB).
- `images_index.csv`: one row per image (zip, path, group, run, model, guidance, class, index).
- `cytodiff_v2_artifacts.zip`: the LoRA adapters (`adapters/<run>/pytorch_lora_weights.safetensors`), all
  result tables and figures (`results/`), and the patient-level split (`manifest.csv`).

## How the images were generated
| Model | Base weights (Hugging Face) | Resolution (px) | Sampling batch |
|---|---|---|---|
{models}

- LoRA rank 8, alpha 8, dropout 0.1 on the attention query, key, value, and output projections (image-token
  stream in SD 3); AdamW, learning rate 1e-4, cosine schedule, effective batch 4, 10,000 steps, BF16.
- Trained on 380 images of 27 patients of LungHist700 (patient-level split, `manifest.csv`); every model at its
  native resolution from the original 1200 x 1200 px center crops.
- Captions: "Hematoxylin and eosin stained lung adenocarcinoma histopathology, malignant glandular structures,
  photomicrograph, microscope view." and "Hematoxylin and eosin stained lung squamous cell carcinoma
  histopathology, keratinizing malignant epithelial nests, photomicrograph, microscope view."; negative prompt
  "blurry, low resolution, distorted nuclei, unrealistic colors, text overlay, frame, watermark".
- 50 sampling steps (DDIM for the U-Nets, flow matching for SD 3); 200 images per class and guidance scale;
  the noise generator of each batch is seeded with the index of its first image (sampling seed 0), so the
  same index gives the same starting noise in both classes and in every run of a model.
- Images are saved at the native resolution; the paper evaluates them after an antialiased resize to 512 px.

## Intended use and limits
- Intended: research on generative models and their evaluation for histopathology; visualization.
- Out of scope: clinical diagnosis; teaching or training clinical models without pathologist validation.
- Labels are the classes of the generating captions and are not verified by pathologists.
- Known limits: partial coverage of real tissue variety on pathology foundation model features; keratin
  pearls rarely present; magnification (20x / 40x) is not controlled.
- No generated image was found to copy a training image (nearest-neighbor search with UNI, LPIPS, SSIM).

## Provenance and licenses
Derived from LungHist700 (Diosdado et al., Sci. Data 11, 1088, 2024; doi:10.6084/m9.figshare.25459174;
CC BY 4.0; ethics approval CEIm Area de Salud Valladolid Este, PI 23-3167). The real images are not
redistributed. The terms of each generating model's license apply; see `LICENSES.md`.
"""


def citation(anonymous: bool, github: str) -> str:
    authors = "" if anonymous else "".join(f"  - family-names: {f}\n    given-names: {g}\n" for f, g in AUTHORS)
    return (f"cff-version: 1.2.0\nmessage: \"If you use this work, please cite it.\"\ntitle: \"{TITLE}\"\n"
            + (f"authors:\n{authors}" if authors else "authors:\n  - name: Anonymous\n")
            + f"repository-code: \"{github}\"\ndoi: \"{ZENODO_DOI}\"\n")


def code_files() -> list[Path]:
    """Public code: src/, scripts/, configs/, tests/ and the top-level docs, without internal or v1-only files."""
    skip = {"scripts/.gitkeep", "scripts/make_progress_slides.py", "scripts/make_seminar_slides.py",
            "scripts/make_compare_4models.py"}
    code = [p for d in ("src", "scripts", "configs", "tests") for p in (ROOT / d).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".egg-info" not in str(p)
            and "legacy_runs" not in p.parts and str(p.relative_to(ROOT)) not in skip]
    return code + [ROOT / f for f in ("pyproject.toml", "requirements.txt", "requirements-lock.txt", "README.md",
                                      "RERUN.md", "LICENSE", ".gitignore")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Package the code and the small artifacts for GitHub and Zenodo.")
    ap.add_argument("--anonymous", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the dataset card without writing archives")
    ap.add_argument("--code-license", default="MIT")
    ap.add_argument("--github", default=GITHUB_URL, help="URL of the code repository")
    ap.add_argument("--cards-only", action="store_true",
                    help="rewrite README.md, LICENSES.md, CITATION.cff only (keeps the zips, which may already be uploaded)")
    args = ap.parse_args()

    code = code_files()
    adapters = sorted(EXP.glob("checkpoints/*/final/*"))
    results = sorted(p for p in RESULTS.iterdir() if p.is_file() and p.suffix in (".csv", ".png"))
    counts = image_counts()
    print(f"code files: {len(code)}, adapters: {len(adapters)}, result files: {len(results)}")
    print(f"images expected on the GPU box: {counts['images'].sum():,} in {len(counts)} runs")
    if args.dry_run:
        print(dataset_card(args.anonymous, args.github))
        return

    zen = OUT / "zenodo"
    zen.mkdir(parents=True, exist_ok=True)
    licenses = LICENSES.format(code_license=args.code_license)
    (OUT / "LICENSES.md").write_text(licenses)
    (zen / "LICENSES.md").write_text("\n".join(l for l in licenses.splitlines() if "cytodiff_code.zip" not in l) + "\n")
    for d in (OUT, zen):
        (d / "CITATION.cff").write_text(citation(args.anonymous, args.github))
    (zen / "README.md").write_text(dataset_card(args.anonymous, args.github))
    if args.cards_only:
        print("  rewrote README.md, LICENSES.md, CITATION.cff")
        return
    with zipfile.ZipFile(OUT / "cytodiff_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in code:
            z.write(p, p.relative_to(ROOT))
        z.write(OUT / "LICENSES.md", "LICENSES.md")
        z.write(OUT / "CITATION.cff", "CITATION.cff")
    with zipfile.ZipFile(zen / "cytodiff_v2_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in adapters:   # experiments/checkpoints/<run>/final/<file> -> adapters/<run>/<file>
            z.write(p, Path("adapters") / p.parent.parent.name / p.name)
        for p in results:
            z.write(p, Path("results") / p.name)
        z.write(ROOT / "data" / "splits" / "manifest.csv", "manifest.csv")
        z.write(zen / "LICENSES.md", "LICENSES.md")
    for p in [OUT / "cytodiff_code.zip", *sorted(zen.iterdir())]:
        print(f"  wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")
    if args.anonymous:
        print("  anonymous build: check README.md and LICENSES.md for names before uploading.")


if __name__ == "__main__":
    main()
