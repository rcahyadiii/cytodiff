#!/usr/bin/env python
"""Stage 3 CLI — cache (if needed) then LoRA fine-tune one model from its config.

Usage:
    venv/bin/python scripts/train.py --config configs/sd15.yaml
    venv/bin/python scripts/train.py --config configs/sd15.yaml --max-steps 3   # smoke test
    venv/bin/python scripts/train.py --config configs/sd15.yaml --cache-only
"""

from __future__ import annotations

import argparse

import torch

from cytodiff.data.cache import build_caches
from cytodiff.train.train import train
from cytodiff.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff stage 3: cache + LoRA fine-tune.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="override max_train_steps (smoke test)")
    ap.add_argument("--save-every", type=int, default=1000, help="optimizer steps between checkpoints")
    ap.add_argument("--cache-only", action="store_true", help="build caches and exit")
    ap.add_argument("--skip-cache", action="store_true", help="assume caches already built")
    ap.add_argument("--seed", type=int, default=None, help="override training.seed")
    ap.add_argument("--run-name", default=None, help="checkpoint folder name (default: model name)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — see project-environment memory for the driver fix.")

    if not cfg.fits_local_8gb:
        print(f"!! {cfg.model.name} is flagged fits_local_8gb=false — expect OOM on 8 GB; run on 24 GB+.")

    if not args.skip_cache and (cfg.data.cache_latents or cfg.data.cache_text_embeddings):
        print(f"== Caching latents + text embeddings for {cfg.model.name} ==")
        build_caches(cfg.model.name, cfg.model.pretrained, cfg.model.native_resolution)
    if args.cache_only:
        return

    print(f"== LoRA fine-tuning {cfg.model.name} ==")
    train(cfg, max_steps=args.max_steps, save_every=args.save_every, seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
