"""Stage 1d — latent + text-embedding caching (the SDXL / SD 3 memory lever).

Encoding every image to its VAE latent and every caption to its text embedding ONCE, to disk,
means training only needs the denoiser + LoRA params resident on the GPU. This is what makes
LoRA-training the big models plausible, and speeds up all four runs.

Cache layout:
    data/processed/cache_v2/<model>/latents/<stem>.pt   # {params, scaling, shift}, native resolution
    data/processed/cache_v2/<model>/text/<sha1>.pt      # single-encoder: Tensor [77,C]
                                                     # dual/mmdit: {emb, pooled}

Families (see constants.MODEL_FAMILY):
    unet_single (sd15/sd21): CLIP text encoder, 4-ch VAE, shift=0.
    unet_dual   (sdxl):      dual CLIP via SDXL pipeline.encode_prompt -> {emb[77,2048], pooled[1280]}.
    mmdit       (sd3):       triple encoder via SD3 pipeline.encode_prompt -> {emb, pooled}; 16-ch VAE w/ shift.

MEMORY: SD 3's T5-XXL (~10 GB fp16) does NOT fit the local 8 GB card even for caching — build SD 3
caches on the cloud GPU. SDXL's dual CLIP is lighter and usually fits 8 GB for caching.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cytodiff.constants import MASTER_RESOLUTION, PROCESSED_DIR, RAW_DIR, model_family

# v2: latents are built at each model's native resolution from the original crops. A new root
# keeps v1 caches (built from upscaled 512 masters) from ever being reused.
CACHE_ROOT = PROCESSED_DIR / "cache_v2"


# --- testable, model-agnostic helpers -------------------------------------------------

def to_vae_input(img: Image.Image, size: int) -> torch.Tensor:
    """PIL RGB -> CHW float tensor in [-1, 1] at (size, size), as SD's VAE expects (§3.2.2)."""
    if img.size != (size, size):
        img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    else:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t * 2.0 - 1.0


def training_image(rel_path: str, size: int) -> Image.Image:
    """Training image at `size`: the 512 master when size is 512, otherwise the original JPEG
    center-cropped and resized directly to `size` (no upscaling from the 512 master)."""
    if size == MASTER_RESOLUTION:
        from cytodiff.constants import PROJECT_ROOT
        return Image.open(PROJECT_ROOT / rel_path).convert("RGB")
    from cytodiff.data.preprocess import center_crop_to_square
    p = Path(rel_path)
    with Image.open(RAW_DIR / p.parent.name / f"{p.stem}.jpg") as im:
        return center_crop_to_square(im.convert("RGB")).resize((size, size), Image.Resampling.BICUBIC)


def caption_hash(caption: str) -> str:
    return hashlib.sha1(caption.encode("utf-8")).hexdigest()[:16]


def latent_cache_path(model_name: str, image_stem: str) -> Path:
    return CACHE_ROOT / model_name / "latents" / f"{image_stem}.pt"


def text_cache_path(model_name: str, caption: str) -> Path:
    return CACHE_ROOT / model_name / "text" / f"{caption_hash(caption)}.pt"


# --- per-family encoders --------------------------------------------------------------

def _cache_text_single(model_name, pretrained, captions, device, dtype):
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(
        pretrained, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    for caption in captions:
        out = text_cache_path(model_name, caption)
        if out.exists():
            continue
        ids = tok(caption, padding="max_length", truncation=True,
                  max_length=tok.model_max_length, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            emb = te(ids)[0].squeeze(0).cpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, out)  # bare Tensor (backward-compatible with existing sd15/sd21 caches)
    del te


def _cache_text_pipeline(model_name, pipe, captions, device):
    """SDXL / SD 3: use the pipeline's own encode_prompt so the dual/triple-encoder
    concatenation + pooling exactly matches what the denoiser expects at train/infer time."""
    import inspect

    # SDXL leaves prompt_2 optional; SD 3 requires prompt_2 AND prompt_3 explicitly.
    # Pass None for any that exist -> each pipeline reuses `prompt` across its encoders.
    accepted = inspect.signature(pipe.encode_prompt).parameters
    extra = {k: None for k in ("prompt_2", "prompt_3") if k in accepted}
    for caption in captions:
        out = text_cache_path(model_name, caption)
        if out.exists():
            continue
        with torch.no_grad():
            emb, _neg, pooled, _negpool = pipe.encode_prompt(
                prompt=caption, device=device, num_images_per_prompt=1,
                do_classifier_free_guidance=False, **extra,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"emb": emb.squeeze(0).cpu(), "pooled": pooled.squeeze(0).cpu()}, out)


def _cache_latents(model_name, vae, df, size, device, dtype):
    scaling = float(vae.config.scaling_factor)
    shift = float(getattr(vae.config, "shift_factor", None) or 0.0)
    # Encode in fp32: the SDXL VAE overflows in fp16 and emits NaN latents (well-known
    # fp16 bug; diffusers force_upcasts it). Safe/cheap for all models. Params stored fp32.
    vae = vae.to(torch.float32)
    for rel_path in df[df["split"] == "train"]["path"]:
        stem = Path(rel_path).stem
        out = latent_cache_path(model_name, stem)
        if out.exists():
            continue
        x = to_vae_input(training_image(rel_path, size), size).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            params = vae.encode(x).latent_dist.parameters.squeeze(0).float().cpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": params, "scaling": scaling, "shift": shift}, out)


# --- driver ---------------------------------------------------------------------------

def build_caches(model_name: str, pretrained: str, size: int, device: str = "cuda") -> None:
    import pandas as pd

    from cytodiff.data.split import MANIFEST_PATH

    fam = model_family(model_name)
    df = pd.read_csv(MANIFEST_PATH)
    dtype = torch.float16
    captions = sorted(df["label"].unique()) + [""]  # "" = CFG unconditional (§3.4.3)

    if fam == "unet_single":
        from diffusers import AutoencoderKL

        _cache_text_single(model_name, pretrained, captions, device, dtype)
        vae = AutoencoderKL.from_pretrained(
            pretrained, subfolder="vae", torch_dtype=dtype
        ).to(device).eval()
        _cache_latents(model_name, vae, df, size, device, dtype)

    elif fam in ("unet_dual", "mmdit"):
        # Load the pipeline on CPU, then move ONLY the components in use to the GPU
        # (never the denoiser — caching doesn't need it). This lets SDXL caching fit
        # 8 GB and lowers the cloud memory floor. SD 3's T5-XXL still needs >8 GB.
        if fam == "unet_dual":
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(pretrained, torch_dtype=dtype)
            encoders = [pipe.text_encoder, pipe.text_encoder_2]
        else:
            from diffusers import StableDiffusion3Pipeline
            pipe = StableDiffusion3Pipeline.from_pretrained(pretrained, torch_dtype=dtype)
            encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]

        for e in encoders:
            if e is not None:
                e.to(device)
        _cache_text_pipeline(model_name, pipe, captions, device)
        for e in encoders:
            if e is not None:
                e.to("cpu")
        torch.cuda.empty_cache()

        pipe.vae.to(device)
        _cache_latents(model_name, pipe.vae, df, size, device, dtype)

    else:
        raise ValueError(f"unhandled family {fam!r}")

    print(f"  cached latents + text embeddings for {model_name} ({fam}) -> {CACHE_ROOT / model_name}")
