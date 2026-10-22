"""Denoiser loading, LoRA injection, and adapter saving across all four families.

Base denoiser is frozen; LoRA adapters go into the attention projections (proposal §3.4.2) and are
cast to fp32 for stable optimization. VAE + text encoders are NOT loaded here — their outputs come
from the stage-1d cache.

    unet_single / unet_dual : UNet2DConditionModel + DDPMScheduler (epsilon or v-prediction).
    mmdit (SD 3)            : SD3Transformer2DModel + FlowMatchEulerDiscreteScheduler.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig

from cytodiff.constants import model_family
from cytodiff.utils.config import Config


def load_denoiser_with_lora(cfg: Config, device: str = "cuda"):
    """Load the frozen base denoiser, inject LoRA, cast adapter params to fp32."""
    fam = model_family(cfg.model.name)
    if fam == "mmdit":
        from diffusers import SD3Transformer2DModel

        net = SD3Transformer2DModel.from_pretrained(
            cfg.model.pretrained, subfolder="transformer", torch_dtype=torch.float32
        )
    else:
        from diffusers import UNet2DConditionModel

        net = UNet2DConditionModel.from_pretrained(
            cfg.model.pretrained, subfolder="unet", torch_dtype=torch.float32
        )

    net.requires_grad_(False)
    lora = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        init_lora_weights="gaussian",
    )
    net.add_adapter(lora)
    if cfg.training.gradient_checkpointing:
        net.enable_gradient_checkpointing()
    for p in net.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA params fp32 even under autocast
    return net.to(device)


# Backward-compatible alias (older callers / tests referenced the U-Net-specific name).
load_unet_with_lora = load_denoiser_with_lora


def load_train_scheduler(cfg: Config):
    """Noise/flow scheduler for the training objective."""
    if model_family(cfg.model.name) == "mmdit":
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            cfg.model.pretrained, subfolder="scheduler"
        )
    from diffusers import DDPMScheduler

    return DDPMScheduler.from_pretrained(cfg.model.pretrained, subfolder="scheduler")


def trainable_params(net) -> list[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def count_trainable(net) -> tuple[int, int]:
    trn = sum(p.numel() for p in net.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in net.parameters())
    return trn, tot


def save_lora_adapter(net, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    net.save_lora_adapter(str(out_dir))  # diffusers >=0.30, loader-compatible export
    return out_dir
