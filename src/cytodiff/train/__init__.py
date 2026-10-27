"""Stage 3: LoRA fine-tuning.

Modules:
    dataset.py - CachedLatentDataset over the stage-1d latent/embedding cache
    lora.py    - U-Net load + LoRA injection (attn q/k/v/out) + adapter save
    train.py   - training loop (epsilon/v MSE, conditioning dropout for CFG)

Driven by a per-model YAML (sd15/sd21/sdxl/sd3). Run: python scripts/train.py --config configs/sd15.yaml
LoRA rank 8 for every model; AdamW lr 1e-4 with cosine decay, batch 1 x grad-accum 4, BF16 (v2 study).
Training runs a fixed 10,000 steps and the final checkpoint is used; there is no early stopping.
"""
