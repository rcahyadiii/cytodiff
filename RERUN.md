# Re-run v2: patient-level split, native resolutions, and the review fixes

v1 split images (not patients), scored SCC against its own training images, and sampled SD 2.1 and
SD 3 at a different resolution from the one their LoRAs were trained at. A review of v1
(2026-10-15) asked for these and further changes. All four models are retrained on one cloud
GPU with `scripts/cloud/run_all.sh`.

## What changes in v2

| Review point | v2 |
|---|---|
| A. Image-level split (56 of 56 ADC reference images from training patients) | Patient-level split from LungHist700 `data.csv`, both classes; no patient on both sides |
| B. SCC reference = its training images | SCC has its own held-out patients |
| C. Pooled FID with mismatched class mix drove CFG choice | Per-class metrics and equal-weight macro average; best CFG by Inception macro KID |
| D. 56-image reference, rank-deficient covariance, overlapping KID subsets | Full-set KID with 95 % patient-cluster bootstrap intervals; FD secondary |
| 1. "SD 3 best" not tested statistically | Paired bootstrap (KID and FD, all extractors; FD intervals are normal bootstrap intervals because FD is biased under resampling); 3 training seeds for the two best models |
| 2. No memorization analysis | Nearest training image per synthetic image (UNI search, LPIPS, SSIM) vs held-out real baseline; closest-pair figures |
| 3. Optimum on the CFG grid edge | CFG grid 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20 |
| 4. FID without reference points | Real-vs-real baseline (train vs held-out); UNI and Phikon features; precision/recall and density/coverage |
| 5. Missing baselines | Zero-shot baseline for EVERY model (base model without LoRA, same captions, resolution and CFG grid), run first and compared with its LoRA version (`lora_vs_zeroshot.csv`); linear-probe label check and Classification Accuracy Score; GAN baseline not run (justified in the paper) |
| 7. Unequal comparison | Same GPU, BF16 and plain AdamW for all four; LoRA parameter counts for all four from the logs; image-stream-only LoRA in SD 3 reported as a confounder |
| 8. Reproducibility | Each model trains AND samples at its native resolution (SD 1.5 512, SD 2.1 768, SDXL 1024, SD 3 1024; training images made from the original 1200 px crops, never upscaled); all images resized to 512 px with antialiasing before feature extraction; training seeded (v1 was not); library versions and wall time per stage logged; final checkpoint at 10,000 steps |
| 9. No data/code availability | `scripts/package_release.py` builds the GitHub code archive and the Zenodo artifact archive |
| 10. 16-channel latent claim untested | VAE encode-decode PSNR, SSIM, LPIPS for all four VAEs |

New split (deterministic, seed 42):

| Class | Train | Held-out |
|---|---|---|
| ADC | 197 images / 13 patients | 83 images / 8 patients |
| SCC | 183 images / 15 patients | 77 images / 6 patients |

Held-out patients: 3, 4, 11, 12, 13, 17, 19, 21, 28, 31, 32, 34, 39. v1's manifest is kept as
`data/splits/manifest_imagesplit_v1.csv`.

## 1. Before renting the GPU

On huggingface.co, with the account whose token you will use:
- accept the license of **stabilityai/stable-diffusion-3-medium** (gated), and
- request access to **MahmoodLab/UNI** (automatic approval after accepting its terms).

`run_all.sh` checks this access first and stops within a minute if anything is missing.

## 2. Rent the GPU

- 48 GB VRAM recommended (A40 / A6000 / L40S); 24 GB may run out of memory for SD 3 and SDXL at 1024 px.
- At least 150 GB disk (about 45 GB of model weights, about 30 GB of samples).
- On RunPod: a 200 GB network volume, an On-Demand pod, one GPU type for the whole run; see
  `scripts/cloud/README.md`, section RunPod.
- Compute: this is a large run (four zero-shot sweeps and four LoRA models at native resolution,
  9 guidance scales x 400 images each, plus four extra seed runs). The script is resumable, so it can be spread over
  several sessions; `MODELS` and `SEEDS` select subsets.

## 3. Run

```
# on the Mac, inside cytodiff-project/
COPYFILE_DISABLE=1 tar --no-xattrs --no-fflags -czf cytodiff_code.tar.gz --exclude __pycache__ --exclude '*.egg-info' \
  src scripts configs pyproject.toml requirements.txt README.md RERUN.md
scp cytodiff_code.tar.gz user@BOX:/workspace/

# on the GPU box
mkdir -p /workspace/cytodiff && tar --no-same-owner -xzf /workspace/cytodiff_code.tar.gz -C /workspace/cytodiff
cd /workspace/cytodiff
export HF_TOKEN=hf_xxx
bash scripts/cloud/setup.sh
tmux new -s run
SMOKE=1 bash scripts/cloud/run_all.sh      # quick end-to-end check of every stage, toy size
bash scripts/cloud/run_all.sh              # the real run
```

**Run the smoke check first.** Many stages (SD 3 and SDXL at 1024 px, UNI features, the SD 3 VAE,
seed runs, the archive) have never run on a GPU. `SMOKE=1` exercises all of them at toy size
(20 training steps, 8 images per class, CFG 3 and 5, one extra seed, 50 bootstrap draws) and writes
to `experiments_smoke/` and `outputs/logs_smoke/`, so it never mixes with the real results. It ends
with "SMOKE RUN PASSED". Model downloads and latent caches are reused by the real run.

Stages: access checks; LungHist700 download and split verification; **zero-shot baselines first**
(each base model without LoRA: 9-scale CFG sweep and evaluation); then for each model a smoke gate,
10,000-step LoRA training (seed 0), the 9-scale CFG sweep and evaluation; three-seed runs of the two
best models at their best CFG; color transfer (RGB, CIELAB) on the two best models; real-vs-real
baseline; paired model comparison including LoRA vs zero-shot; color statistics; memorization (LoRA
and zero-shot runs); label check; VAE reconstruction; LoRA parameter counts; results archive
`cytodiff_v2_results_<date>.tar.gz`. `ZS_CFGS="3 5 7.5 10"` narrows the zero-shot grid if needed.

## 4. Download and unpack on the Mac

```
scp user@BOX:/workspace/cytodiff/cytodiff_v2_results_*.tar.gz ~/Downloads/
cd cytodiff-project
mv experiments experiments_v1_imagesplit       # keep v1 apart
tar -xzf ~/Downloads/cytodiff_v2_results_*.tar.gz
```

Main result files in `experiments/results/`: `metrics_<run>.csv`, `baseline_real.csv`,
`model_ranking.csv`, `model_pairwise.csv`, `lora_vs_zeroshot.csv`, `seed_variability.csv`, `color_stats_*.csv`,
`memorization_summary.csv` and `memorization_closest_<run>.png`, `label_check.csv`,
`vae_reconstruction_summary.csv`, `lora_params.csv`; logs, versions and timings in `outputs/logs/`.

After unpacking (done 2026-10-21; v1 outputs kept in `experiments_v1_imagesplit/` and `outputs/logs_v1_imagesplit/`),
three scripts run on the Mac from the cached features, without images or a GPU:
- `scripts/recompute_kid_intervals.py` reproduced every bootstrap draw of the cloud run and converted single-run KID
  intervals to normal intervals (the patient bootstrap shifts resampled KID upward); originals in
  `experiments/results/_percentile_kid_intervals/`.
- `scripts/label_vs_cfg.py` writes `label_vs_cfg.csv`: label fidelity, own- and cross-class KID, and ADC/SCC pair
  similarity at every guidance scale (exploratory, added after the results).
- `scripts/make_figures_v2.py` writes `reports/ieee_paper/fig_cfg_v2.pdf` and `fig_qual_v2.jpg`.
Evaluation also runs on Apple MPS or CPU from the cached features in `experiments/features/`.

## 5. Release

```
python scripts/package_release.py --code-license MIT            # or --anonymous for review
```

Upload `release/cytodiff_code.zip` to GitHub and `release/cytodiff_v2_artifacts.zip` to Zenodo,
then put both links in the paper's data and code availability statement.
