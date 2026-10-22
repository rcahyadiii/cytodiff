#!/usr/bin/env python
"""Autoencoder reconstruction fidelity on LungHist700.

Every real image is encoded and decoded (latent mean) by each model's pretrained VAE at the same
512 px input, and compared with the original by PSNR, SSIM and LPIPS (AlexNet). This tests the
claim that SD 3's 16-channel latent preserves fine histological detail better than the 4-channel
latents of SD 1.5, SD 2.1 and SDXL. The VAEs are not fine-tuned, so all 540 images are used.
All VAEs run in fp32 (the SDXL VAE overflows in fp16).

Outputs in experiments/results/: vae_reconstruction.csv (per image and model) and
vae_reconstruction_summary.csv (means with 95 % patient-bootstrap intervals).

Usage:
    venv/bin/python scripts/vae_reconstruction.py --models sd15 sd21 sdxl sd3
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from cytodiff.constants import MASTER_RESOLUTION, PROJECT_ROOT, model_family
from cytodiff.data.cache import to_vae_input
from cytodiff.data.split import MANIFEST_PATH
from cytodiff.eval.evaluate import RESULTS_DIR
from cytodiff.eval.metrics import cluster_resamples, default_device, load_eval_image
from cytodiff.utils.config import load_config


def to_uint8(x: torch.Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="VAE encode-decode fidelity per model.")
    ap.add_argument("--models", nargs="+", default=["sd15", "sd21", "sdxl", "sd3"])
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (smoke test)")
    args = ap.parse_args()

    import lpips
    from diffusers import AutoencoderKL
    device = default_device()
    lpips_net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    df = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        df = df.groupby("class").head(args.limit // 2)
    rows = []
    for m in args.models:
        cfg = load_config(PROJECT_ROOT / "configs" / f"{m}.yaml")
        vae = AutoencoderKL.from_pretrained(cfg.model.pretrained, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
        ch = vae.config.latent_channels
        for _, r in df.iterrows():
            x = to_vae_input(load_eval_image(PROJECT_ROOT / r["path"]), MASTER_RESOLUTION).unsqueeze(0).to(device)
            with torch.no_grad():
                rec = vae.decode(vae.encode(x).latent_dist.mode()).sample
                lp = float(lpips_net(x, rec.clamp(-1, 1)).item())
            a, b = to_uint8(x[0]), to_uint8(rec[0])
            rows.append({"model": m, "family": model_family(m), "latent_channels": ch, "path": r["path"],
                         "class": r["class"], "patient_id": str(r["patient_id"]),
                         "psnr": peak_signal_noise_ratio(a, b, data_range=255),
                         "ssim": structural_similarity(a, b, channel_axis=2, data_range=255), "lpips": lp})
        print(f"  {m} ({ch}-channel latent): done {len(df)} images")
        del vae
        if device == "cuda":
            torch.cuda.empty_cache()

    per = pd.DataFrame(rows)
    summary = []
    for m, g in per.groupby("model", sort=False):
        reps = cluster_resamples(g["patient_id"].to_numpy(), 1000, 0)
        row = {"model": m, "latent_channels": int(g["latent_channels"].iloc[0]), "n_images": len(g)}
        for k in ("psnr", "ssim", "lpips"):
            v = g[k].to_numpy()
            draws = np.array([v[i].mean() for i in reps])
            row.update({k: v.mean(), f"{k}_ci_low": np.percentile(draws, 2.5), f"{k}_ci_high": np.percentile(draws, 97.5)})
        summary.append(row)
    summary = pd.DataFrame(summary)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(RESULTS_DIR / "vae_reconstruction.csv", index=False)
    summary.to_csv(RESULTS_DIR / "vae_reconstruction_summary.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(summary.to_string(index=False))
    print(f"\n  wrote vae_reconstruction.csv and vae_reconstruction_summary.csv in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
