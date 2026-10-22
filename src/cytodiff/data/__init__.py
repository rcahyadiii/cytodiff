"""Stage 1 — dataset preparation.

Modules:
    preprocess.py  - inspect/clean, center-crop, bicubic-resize to 512 RGB master  (§3.2.2)
    split.py       - patient/patch-wise train/eval split + manifest.csv            (§3.2.3)
    caption.py     - per-class label assignment; prompt-variant ablation hook       (§3.3.1)
    cache.py       - latent + text-embedding caching (SDXL/SD3 memory lever)        (§3.4)

Run end-to-end with: python scripts/prepare_data.py
"""
