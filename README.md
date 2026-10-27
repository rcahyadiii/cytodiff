# CytoDiff

Code for the paper **A Comparison of Stable Diffusion Backbones With and Without LoRA Fine-Tuning for
Synthetic H&E Lung Cancer Histopathology** (Muhammad Ridwan Cahyadi and Rifki Wijaya, Telkom University).

Four Stable Diffusion backbones (v1.5, v2.1, SDXL, SD 3 Medium) are fine-tuned with LoRA on adenocarcinoma
(ADC) and squamous cell carcinoma (SCC) images of the public LungHist700 dataset, each at its native
resolution, and compared with their own zero-shot versions under a patient-level split.

- **Data** (all 32,800 generated images, the LoRA adapters, every result table, the patient-level split):
  [doi:10.5281/zenodo.23520698](https://doi.org/10.5281/zenodo.23520698)
- **Generated images are for research only.** They are not validated by pathologists, must not be used for
  teaching without that validation, and must never be used for clinical diagnosis.

## Results

Inception macro KID (mean of ADC and SCC) against images of held-out patients, each run at its best guidance
scale, with 95 % normal bootstrap interval (patients resampled). Real training vs. held-out images: 0.004.

| Model | LoRA: guidance | LoRA: macro KID | Zero-shot: macro KID |
|---|---|---|---|
| SD 3 Medium | 1.5 | **0.016** (0.006 to 0.027) | 0.203 |
| SD v2.1 | 4 | 0.099 (0.078 to 0.120) | 0.173 |
| SD v1.5 | 2 | 0.115 (0.105 to 0.126) | 0.233 |
| SDXL | 3 | 0.213 (0.199 to 0.227) | 0.256 |

- SD 3 is closest in distribution to the held-out patients: lower KID than every other model in every paired
  bootstrap comparison on Inception, UNI, and Phikon features, and lower than the runner-up with each of three
  training seeds. With 6 to 8 held-out patients per class and one split, these intervals are approximate.
- LoRA improves every backbone; SD 3 gains its lead only through adaptation, together with a 16-channel
  autoencoder that reconstructs histology far better (LPIPS 0.040 vs. 0.103 to 0.111).
- No near-duplicate of a training image was found: the 13 generated images closer to a training image than any
  held-out image (7 SD 3, 6 SD v1.5) were inspected and share stain and tissue type, not layout (SSIM <= 0.52).
- At the KID-optimal guidance (1.5), SD 3 images are only weakly class-specific (UNI probe 59 % vs. 90 % on
  real images); guidance 3 to 5 gives 74 to 80 % at a KID still below every other model's best.

## Dataset and split

[LungHist700](https://doi.org/10.1038/s41597-024-03944-3) (Diosdado et al., *Sci. Data* 11, 1088, 2024;
CC BY 4.0): 1600×1200 H&E images at 20× and 40×. The three adenocarcinoma grades map to ADC and the three
SCC grades to SCC (`LUNGHIST700_CLASS_MAP` in `src/cytodiff/constants.py`); the normal images are excluded.
The data are not redistributed here; `scripts/fetch_lunghist700.sh` downloads them from figshare.

The split is made **by patient** from LungHist700's `data.csv`, so no patient appears in both training and
evaluation (`src/cytodiff/data/split.py`, seed 42):

| Class | Images | Training | Held-out reference |
|---|---|---|---|
| ADC | 280 | 197 (13 patients) | 83 (8 patients) |
| SCC | 260 | 183 (15 patients) | 77 (6 patients) |

## Environment

```bash
bash scripts/setup_env.sh
source venv/bin/activate
```

Python 3.12 with diffusers 0.38, peft 0.19, transformers 5.8, accelerate 1.13, torchmetrics 1.9. The study
ran on one NVIDIA A40 (48 GB) with torch 2.14.1+cu130; `requirements-lock.txt` is the complete environment of
that machine (pip freeze), and `PYTHONPATH=src python -m pytest tests` checks KID and FD against direct
formulas, scipy, and torchmetrics; `scripts/cloud/README.md` sets up a cloud GPU
(24 GB or more). Evaluation also runs on Apple MPS or CPU. SD 3 Medium and the UNI feature extractor are
gated on Hugging Face and need an accepted license and a token.

## Running

```bash
bash scripts/cloud/run_all.sh     # on the GPU box: data, split, zero-shot sweeps, 4 LoRA models, seeds, analyses
```

Single steps:

```bash
bash scripts/fetch_lunghist700.sh                               # LungHist700 from figshare
venv/bin/python scripts/prepare_data.py                         # dataset/ -> data/processed + split manifest
bash scripts/cloud/run_model.sh sd3                             # one model: train -> guidance sweep -> evaluate
venv/bin/python scripts/compare_models.py --models sd15 sd21 sdxl sd3
venv/bin/python scripts/colortransfer.py --model sd3 --cfg 1.5 --space rgb
```

`RERUN.md` describes the full v2 protocol and how the results are unpacked; `scripts/package_release.py`
builds the Zenodo dataset card from the result tables.

## Layout

```
src/cytodiff/      package (pip install -e .): data, train (LoRA), generate, eval (KID, FD, PRDC), colortransfer
scripts/           command-line entry points; scripts/cloud/ runs the whole pipeline on a GPU box
configs/           per-model settings: sd15, sd21, sdxl, sd3
tests/             metric checks (KID, FD) against independent implementations
```

## Citation and license

Please cite the paper and the dataset; see `CITATION.cff`. The code is released under the MIT license
(`LICENSE`). The adapters and generated images are subject to the licenses of the base models
(CreativeML Open RAIL-M / RAIL++-M, Stability AI license for SD 3 Medium) and to the CC BY 4.0 terms of
LungHist700.
