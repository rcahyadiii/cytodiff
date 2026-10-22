"""Shared constants grounded in the proposal — single source of truth across stages."""

import os
from pathlib import Path

# --- Paths (repo-relative) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "dataset"                # class-mapped JPEGs: dataset/ADC, dataset/SCC
LUNGHIST700_DIR = RAW_DIR / "lunghist700_original"  # untouched 7-class LungHist700 download
LUNGHIST700_PATIENT_CSV = LUNGHIST700_DIR / "data.csv"  # official image -> patient_id table
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
# Overridable so a smoke run (scripts/cloud/run_all.sh with SMOKE=1) never touches the real outputs.
EXPERIMENTS_DIR = Path(os.environ.get("CYTODIFF_EXPERIMENTS_DIR", PROJECT_ROOT / "experiments"))

# --- Classes ---
CLASSES = ("ADC", "SCC")

# Per-class text labels. LungHist700 is H&E-stained lung HISTOPATHOLOGY (tissue sections).
# The "photomicrograph / microscope view" framing is kept from the captions of an earlier pilot
# study, where it helped.
CLASS_LABELS = {
    "ADC": "Hematoxylin and eosin stained lung adenocarcinoma histopathology, malignant glandular structures, photomicrograph, microscope view.",
    "SCC": "Hematoxylin and eosin stained lung squamous cell carcinoma histopathology, keratinizing malignant epithelial nests, photomicrograph, microscope view.",
}

# Shared negative prompt at inference (proposal §3.3.2).
NEGATIVE_PROMPT = (
    "blurry, low resolution, distorted nuclei, unrealistic colors, "
    "text overlay, frame, watermark"
)

# --- Master preprocessing target (proposal §3.2.2) ---
MASTER_RESOLUTION = 512          # 512x512 master: evaluation resolution for every model
# v2: each model trains and generates at its pretraining resolution; training images at 768/1024
# are made from the original 1200x1200 crops (never upscaled from the 512 master).
MODEL_NATIVE_RES = {"sd15": 512, "sd21": 768, "sdxl": 1024, "sd3": 1024}

# --- Model families (drive encoder/denoiser/loss dispatch across the pipeline) ---
#   unet_single : U-Net + one CLIP encoder, epsilon or v-prediction   (SD v1.5 / v2.1)
#   unet_dual   : U-Net + two CLIP encoders + pooled/time-id conditioning, epsilon  (SDXL)
#   mmdit       : SD3 transformer + three encoders (incl. T5), flow-matching loss   (SD 3)
MODEL_FAMILY = {
    "sd15": "unet_single",
    "sd21": "unet_single",
    "sdxl": "unet_dual",
    "sd3": "mmdit",
}


def model_family(name: str) -> str:
    try:
        return MODEL_FAMILY[name]
    except KeyError:
        raise ValueError(f"unknown model name {name!r}; add it to MODEL_FAMILY") from None

# --- Generation (proposal §3.5) ---
CFG_SWEEP = (1.5, 2, 3, 4, 5, 7.5, 10, 15, 20)   # v1 used 5..20; its optimum (5) sat on the grid edge
INFERENCE_STEPS = 50
FINAL_TARGET_PER_CLASS = 2000

# --- Dataset reality on disk (LungHist700, Diosdado et al., Sci. Data 2024) ---
# ADC = aca_bd + aca_md + aca_pd, SCC = scc_bd + scc_md + scc_pd; 'nor' (151) excluded.
# 1600x1200 JPEGs at 20x and 40x, ~1.08:1 ADC:SCC; 21 ADC and 21 SCC patients (2 have both).
# data.csv maps every image to its patient, so the split is made at patient level for BOTH
# classes (cytodiff.data.split): no patient appears in both training and evaluation.
DATASET_COUNTS = {"ADC": 280, "SCC": 260}
LUNGHIST700_CLASS_MAP = {
    "ADC": ("aca_bd", "aca_md", "aca_pd"),
    "SCC": ("scc_bd", "scc_md", "scc_pd"),
}

# --- Patient-level split ---
HELDOUT_FRAC = 0.30   # target share of each class's images held out for evaluation
SPLIT_SEED = 42
