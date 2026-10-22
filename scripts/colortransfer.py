#!/usr/bin/env python
"""Stage 6 CLI — apply histogram-matching colour transfer to a model's generated samples.

Writes a new sample tree named <model>_ct_<space> that scripts/evaluate.py can score directly,
so you get a clean before/after FID/KID comparison (proposal §3.7 / §3.8).

Usage:
    venv/bin/python scripts/colortransfer.py --model sd15 --cfg 5 --space rgb
    venv/bin/python scripts/colortransfer.py --model sd15 --cfg 5 --space lab --strategy fixed
"""

from __future__ import annotations

import argparse

from cytodiff.colortransfer import apply_to_model


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff stage 6: histogram colour transfer.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", type=float, required=True)
    ap.add_argument("--space", choices=["rgb", "lab"], default="rgb")
    ap.add_argument("--strategy", choices=["random_per_image", "fixed"], default="random_per_image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = apply_to_model(args.model, args.cfg, space=args.space,
                         strategy=args.strategy, seed=args.seed)
    print(f"  done -> evaluate with: scripts/evaluate.py --model {out} --cfg {args.cfg}")


if __name__ == "__main__":
    main()
