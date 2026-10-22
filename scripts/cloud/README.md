# Cloud run: all four models on one GPU

All four backbones (SD 1.5, SD 2.1, SDXL, SD 3) are trained, sampled, and evaluated on one rented
GPU, so they share the same hardware. The whole v2 pipeline is one resumable command.

## 1. Rent a GPU
- 48 GB VRAM recommended (A40 / A6000 / L40S): SD 3 and SDXL train and sample at 1024 px in v2.
- At least 150 GB disk: about 45 GB of Hugging Face weights, about 30 GB of samples, plus caches.
- Ubuntu with Python 3.11+ and an NVIDIA driver for CUDA 12 or 13 (RunPod, Vast.ai, Lambda, and similar).
  On RunPod, follow the RunPod section below.
- Before starting: accept the SD 3 Medium license and request MahmoodLab/UNI access on
  huggingface.co with the account of your `HF_TOKEN` (see `RERUN.md` step 1).

## 2. Copy the code (not the data)
From the Mac, inside `cytodiff-project/`:
```
COPYFILE_DISABLE=1 tar --no-xattrs --no-fflags -czf cytodiff_code.tar.gz --exclude __pycache__ --exclude '*.egg-info' \
  src scripts configs pyproject.toml requirements.txt README.md RERUN.md
scp cytodiff_code.tar.gz user@BOX:/workspace/
ssh user@BOX 'mkdir -p /workspace/cytodiff && tar --no-same-owner -xzf /workspace/cytodiff_code.tar.gz -C /workspace/cytodiff'
```
The dataset is downloaded on the box from figshare by `scripts/fetch_lunghist700.sh`.

## 3. Setup
```
cd /workspace/cytodiff
export HF_TOKEN=hf_xxx        # SD 3 is gated (accept its license on huggingface.co first)
bash scripts/cloud/setup.sh
```

## 4. Run everything
Use `tmux` (or `screen`) so the run survives a dropped connection:
```
tmux new -s run
SMOKE=1 bash scripts/cloud/run_all.sh     # quick check of every stage at toy size
bash scripts/cloud/run_all.sh             # the real run
```
The full list of stages is in `RERUN.md` (step 3). If the run stops, run the same command again:
finished stages are skipped. `MODELS="sd15 sd21"` or `SEEDS="1"` restrict a session to a subset.

## 5. Bring back
```
scp user@BOX:/workspace/cytodiff/cytodiff_v2_results_*.tar.gz .
```
Then follow `RERUN.md` step 4 on the Mac.

## RunPod
1. **Storage.** Create a Network Volume of 200 GB (Storage, New Network Volume) in a Secure Cloud
   data center that lists the GPU you want. It is mounted at `/workspace` and outlives the pod.
   Keep the project, its venv, the model caches and all outputs in `/workspace/cytodiff`: the
   container disk is small (about 20 GB) and is wiped when the pod stops. `scripts/cloud/env.sh`
   points `HF_HOME`, `TORCH_HOME` and the pip cache to `/workspace/cytodiff/.cache`.
2. **Pod.** Deploy on that volume: one 48 GB GPU (A40, RTX A6000 or L40S), On-Demand (not Spot:
   a training run cannot resume mid-way, so a preempted pod restarts that model's training), any
   RunPod PyTorch template (Python 3.11+; `setup.sh` builds its own venv). Use the SAME GPU type for
   the whole run, including after a restart, because wall times are reported per stage.
3. **Copy the code.** `runpodctl` is preinstalled on pods; on the Mac:
   `brew install runpod/runpodctl/runpodctl`, then `runpodctl send cytodiff_code.tar.gz` and, in
   the pod's terminal, `cd /workspace && runpodctl receive <code>`, then unpack with
   `mkdir -p cytodiff && tar --no-same-owner -xzf cytodiff_code.tar.gz -C cytodiff` (the volume does
   not allow `chown`, so plain `tar -x` as root reports errors). (The JupyterLab file browser of
   the template also accepts uploads.)
4. **Run** steps 3 and 4 above in the pod's web terminal or over SSH. Detach from tmux with
   `Ctrl-b d`; reattach with `tmux attach -t run`.
5. **Bring back** with `runpodctl send cytodiff_v2_results_*.tar.gz` in the pod and
   `runpodctl receive <code>` on the Mac.
6. **Clean up** after the archive is safely on the Mac: terminate the pod and delete the network
   volume. The volume holds your Hugging Face token (`.cache/huggingface/token`); deleting the
   volume, or the token on huggingface.co, removes it.

## Guidance below 1.5
`scripts/cloud/extend_cfg.sh` extends the guidance grid to 1.25 and 1 (no guidance) for every run whose
best scale is the lowest grid value, then reruns `run_all.sh` for the stages that depend on the best
scale (seed samples, color transfer, analyses, archive; no training). Start it in a second tmux
session while the main run is going; it waits for the main run to finish. `DRY_RUN=1` prints the plan.

## Notes
- SDXL latents are cached with the VAE in fp32 (the fp16 SDXL VAE emits NaN latents); train in bf16.
- SD 3 is the highest-risk model (flow-matching loss and transformer LoRA targets); the smoke gate
  aborts within about a minute on a config, shape, or LoRA-target error.
- `run_model.sh <model>` still runs a single model if needed.
