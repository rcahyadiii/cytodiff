#!/usr/bin/env python
"""Stage 4a CLI — generate synthetic images across the CFG sweep for one model.

Usage:
    venv/bin/python scripts/generate.py --config configs/sd15.yaml --num-images 200
    venv/bin/python scripts/generate.py --config configs/sd15.yaml --num-images 4 --cfg 7.5  # smoke
"""

from __future__ import annotations

import argparse

import torch

from cytodiff.constants import CFG_SWEEP
from cytodiff.generate.generate import generate_for_model
from cytodiff.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff stage 4a: generation + CFG sweep.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-images", type=int, default=200, help="images per class per CFG")
    ap.add_argument("--cfg", type=float, nargs="*", default=None, help="override CFG sweep")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (image i uses seed + i)")
    ap.add_argument("--run-name", default=None, help="adapter run and output folder (default: model name)")
    ap.add_argument("--zero-shot", action="store_true", help="base model without LoRA (baseline)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    cfg = load_config(args.config)
    cfgs = tuple(args.cfg) if args.cfg else CFG_SWEEP
    generate_for_model(
        cfg.model.name, cfg.model.pretrained,
        num_images=args.num_images, resolution=cfg.model.native_resolution, cfgs=cfgs,
        steps=args.steps, batch=args.batch, seed=args.seed,
        run_name=args.run_name, zero_shot=args.zero_shot,
    )


if __name__ == "__main__":
    main()
