"""Stage 4a: synthetic image generation across the guidance sweep.

For each class and each guidance scale, generate N images at the model's native (training)
resolution and save them to experiments/samples/<run>/cfg_<w>/<class>/<class>_<idx>.png.
The resolution is passed explicitly: in v1 the pipelines' defaults made SD 2.1 and SD 3 sample at
768 and 1024 px although their LoRAs were trained at 512 px.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cytodiff.constants import (
    CFG_SWEEP,
    CLASS_LABELS,
    CLASSES,
    EXPERIMENTS_DIR,
    INFERENCE_STEPS,
    NEGATIVE_PROMPT,
)
from cytodiff.generate.pipeline import build_pipeline, default_adapter_dir


def _sample_dir(run_name: str, cfg: float, cls: str) -> Path:
    return EXPERIMENTS_DIR / "samples" / run_name / f"cfg_{cfg:g}" / cls


def generate_for_model(
    model_name: str,
    pretrained: str,
    num_images: int,
    resolution: int,
    cfgs: tuple[float, ...] = CFG_SWEEP,
    classes: tuple[str, ...] = CLASSES,
    adapter_dir: Path | None = None,
    steps: int = INFERENCE_STEPS,
    batch: int = 8,
    seed: int = 0,
    device: str = "cuda",
    run_name: str | None = None,
    zero_shot: bool = False,
) -> None:
    """run_name: output folder and adapter run (default: model name). zero_shot: no LoRA."""
    run_name = run_name or model_name
    if zero_shot:
        adapter_dir = None
    else:
        adapter_dir = adapter_dir or default_adapter_dir(run_name)
    pipe = build_pipeline(pretrained, adapter_dir, model_name=model_name, device=device)

    for cfg in cfgs:
        for cls in classes:
            out_dir = _sample_dir(run_name, cfg, cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            prompt = CLASS_LABELS[cls]
            made = 0
            while made < num_images:
                n = min(batch, num_images - made)
                gen = torch.Generator(device=device).manual_seed(seed + made)
                images = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    num_images_per_prompt=n,
                    height=resolution,
                    width=resolution,
                    generator=gen,
                ).images
                for img in images:
                    img.save(out_dir / f"{cls}_{made:05d}.png")
                    made += 1
            print(f"  {run_name} cfg={cfg:g} {cls}: {made} images at {resolution}px -> {out_dir}")
