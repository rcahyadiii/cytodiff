"""Stage 3: LoRA fine-tuning loop for all four model families.

Trains on cached latents/embeddings. Denoiser + objective dispatch on model family:
    unet_single (sd15/sd21) : add_noise, epsilon or v-prediction MSE.
    unet_dual   (sdxl)      : same, plus SDXL added_cond_kwargs (pooled text + add_time_ids).
    mmdit       (sd3)       : flow matching — noisy=(1-sigma)z0+sigma*noise, target=noise-z0.
Conditioning dropout swaps in the cached "" embedding for classifier-free guidance. Base denoiser
frozen; only LoRA params update.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from cytodiff.constants import EXPERIMENTS_DIR, model_family
from cytodiff.data.cache import text_cache_path
from cytodiff.train.dataset import CachedLatentDataset
from cytodiff.train.lora import (
    count_trainable,
    load_denoiser_with_lora,
    load_train_scheduler,
    save_lora_adapter,
    trainable_params,
)
from cytodiff.utils.config import Config


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch (CPU and CUDA); returns a CPU generator for data sampling.
    Fixes LoRA initialization, data order, timesteps and noise. GPU kernels may still be
    nondeterministic, so runs are seeded but not bit-identical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _cycle(dl):
    while True:
        for batch in dl:
            yield batch


def _sample_z0(batch: dict, device: str) -> torch.Tensor:
    """Sample a fresh latent from the cached distribution and apply the model's
    VAE normalization: z0 = (sample - shift) * scaling  (shift=0 for SD1/2/XL)."""
    params = batch["params"].to(device)
    scaling = batch["scaling"][0].item()
    shift = batch["shift"][0].item()
    mean, logvar = params.chunk(2, dim=1)
    std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
    sample = mean + std * torch.randn_like(mean)
    return (sample - shift) * scaling


def _get_sigmas(sched, timesteps, device, dtype, n_dim) -> torch.Tensor:
    """Map sampled flow-matching timesteps to their sigma values (SD 3)."""
    sigmas = sched.sigmas.to(device=device, dtype=dtype)
    schedule_t = sched.timesteps.to(device)
    idx = [(schedule_t == t).nonzero().item() for t in timesteps]
    s = sigmas[idx].flatten()
    while s.ndim < n_dim:
        s = s.unsqueeze(-1)
    return s


def _load_uncond(model_name: str, device: str):
    uc = torch.load(text_cache_path(model_name, ""), weights_only=True)
    if isinstance(uc, dict):
        return uc["emb"].to(device), uc["pooled"].to(device)
    return uc.to(device), None


def train(cfg: Config, max_steps: int | None = None, save_every: int = 1000, device: str = "cuda",
          seed: int | None = None, run_name: str | None = None) -> Path:
    """Train one LoRA adapter. `run_name` names the output folder (default: the model name);
    caches are always shared per model."""
    steps = max_steps or cfg.training.max_train_steps
    seed = cfg.training.seed if seed is None else seed
    run_name = run_name or cfg.model.name
    data_gen = seed_everything(seed)
    print(f"  run {run_name}: seed {seed}, {steps} steps, precision {cfg.training.mixed_precision}")
    # fp16 for SD1.5/2.1 (stable); bf16 for SDXL/SD3 (fp16 overflows -> NaN loss on the
    # large denoisers). bf16 has fp32-range exponent so it needs no GradScaler.
    prec = cfg.training.mixed_precision
    use_fp16, use_bf16 = prec == "fp16", prec == "bf16"
    amp_enabled = use_fp16 or use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    fam = model_family(cfg.model.name)
    is_mmdit, is_dual = fam == "mmdit", fam == "unet_dual"

    ds = CachedLatentDataset(cfg.model.name)
    if cfg.training.balanced_sampling:
        weights = ds.class_balanced_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=data_gen)
        dl = DataLoader(ds, batch_size=cfg.training.train_batch_size, sampler=sampler, drop_last=True)
        print(f"  balanced sampling: class counts {ds.rows['class'].value_counts().to_dict()}, ~50/50 per batch")
    else:
        dl = DataLoader(ds, batch_size=cfg.training.train_batch_size, shuffle=True, drop_last=True,
                        generator=data_gen)
    data_iter = _cycle(dl)

    net = load_denoiser_with_lora(cfg, device=device)
    noise_sched = load_train_scheduler(cfg)
    pred_type = getattr(noise_sched.config, "prediction_type", "flow_matching")
    num_train_t = noise_sched.config.num_train_timesteps
    trn, tot = count_trainable(net)
    print(f"  {cfg.model.name} ({fam}): trainable {trn:,}/{tot:,} ({100*trn/tot:.3f}%), pred={pred_type}")

    uncond_emb, uncond_pooled = _load_uncond(cfg.model.name, device)

    # SDXL conditioning: constant add_time_ids for square native-res crops
    add_time_ids = None
    if is_dual:
        r = cfg.model.native_resolution
        add_time_ids = torch.tensor([[r, r, 0, 0, r, r]], device=device, dtype=torch.float32)

    params = trainable_params(net)
    if cfg.training.use_8bit_adam:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    accum = cfg.training.gradient_accumulation_steps
    lr_sched = get_scheduler(cfg.training.lr_scheduler, opt,
                             num_warmup_steps=cfg.training.warmup_steps, num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # no-op passthrough for bf16/fp32

    out_root = EXPERIMENTS_DIR / "checkpoints" / run_name
    net.train()
    opt.zero_grad()
    running = 0.0
    for step in range(1, steps * accum + 1):
        batch = next(data_iter)
        z0 = _sample_z0(batch, device)
        emb = batch["text_emb"].to(device)
        pooled = batch["pooled"].to(device) if (is_dual or is_mmdit) else None

        if cfg.training.conditioning_dropout > 0:
            drop = torch.rand(emb.shape[0], device=device) < cfg.training.conditioning_dropout
            if drop.any():
                emb = emb.clone(); emb[drop] = uncond_emb
                if pooled is not None:
                    pooled = pooled.clone(); pooled[drop] = uncond_pooled

        noise = torch.randn_like(z0)
        bsz = z0.shape[0]

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            if is_mmdit:
                # logit-normal timestep sampling -> flow-matching target
                u = torch.sigmoid(torch.randn(bsz, device=device))
                idx = (u * num_train_t).long().clamp(max=num_train_t - 1)
                timesteps = noise_sched.timesteps.to(device)[idx]
                sigma = _get_sigmas(noise_sched, timesteps, device, z0.dtype, z0.ndim)
                noisy = (1.0 - sigma) * z0 + sigma * noise
                target = noise - z0
                pred = net(hidden_states=noisy, timestep=timesteps, encoder_hidden_states=emb,
                           pooled_projections=pooled, return_dict=False)[0]
            else:
                t = torch.randint(0, num_train_t, (bsz,), device=device).long()
                noisy = noise_sched.add_noise(z0, noise, t)
                target = noise if pred_type == "epsilon" else noise_sched.get_velocity(z0, noise, t)
                if is_dual:
                    added = {"text_embeds": pooled, "time_ids": add_time_ids.repeat(bsz, 1)}
                    pred = net(noisy, t, encoder_hidden_states=emb, added_cond_kwargs=added, return_dict=False)[0]
                else:
                    pred = net(noisy, t, encoder_hidden_states=emb, return_dict=False)[0]
            loss = F.mse_loss(pred.float(), target.float()) / accum

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss ({loss.item()}) at step {step} for {cfg.model.name}. "
                "Common causes: NaN cached latents (SDXL VAE fp16 bug -> cache in fp32), "
                "or fp16 overflow on a large denoiser (-> mixed_precision: bf16)."
            )
        scaler.scale(loss).backward()
        running += loss.item() * accum

        if step % accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            lr_sched.step()
            opt_step = step // accum
            if opt_step % 50 == 0:
                print(f"  step {opt_step}/{steps}  loss {running/50/accum:.4f}  lr {lr_sched.get_last_lr()[0]:.2e}")
                running = 0.0
            if opt_step % save_every == 0 and opt_step < steps:
                save_lora_adapter(net, out_root / f"step_{opt_step}")

    final = save_lora_adapter(net, out_root / "final")
    print(f"  saved LoRA adapter -> {final}")
    return final
