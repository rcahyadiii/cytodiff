"""Inference pipeline builder for all four families, with a trained LoRA adapter (or none, for
the zero-shot baseline).

    unet_single (sd15/sd21): StableDiffusionPipeline, DDIM 50 steps, safety_checker off.
    unet_dual   (sdxl):      StableDiffusionXLPipeline, DDIM.
    mmdit       (sd3):       StableDiffusion3Pipeline, native flow-matching scheduler.

safety_checker is disabled where present — the NSFW filter blacks out legitimate cytology, which
would corrupt FID/KID.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cytodiff.constants import EXPERIMENTS_DIR, model_family

_ADAPTER_WEIGHT = "pytorch_lora_weights.safetensors"


def default_adapter_dir(model_name: str) -> Path:
    return EXPERIMENTS_DIR / "checkpoints" / model_name / "final"


def build_pipeline(pretrained: str, adapter_dir: Path | None, model_name: str, device: str = "cuda"):
    """adapter_dir=None loads the base model without LoRA (zero-shot baseline)."""
    fam = model_family(model_name)

    if fam == "mmdit":
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(pretrained, torch_dtype=torch.float16)
        # SD 3 LoRA lives on the transformer; keep the native flow-matching scheduler.
        if adapter_dir is not None:
            pipe.transformer.load_lora_adapter(str(adapter_dir), weight_name=_ADAPTER_WEIGHT, prefix=None)

    elif fam == "unet_dual":
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(pretrained, torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if adapter_dir is not None:
            pipe.unet.load_lora_adapter(str(adapter_dir), weight_name=_ADAPTER_WEIGHT, prefix=None)

    else:
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if adapter_dir is not None:
            pipe.unet.load_lora_adapter(str(adapter_dir), weight_name=_ADAPTER_WEIGHT, prefix=None)

    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
