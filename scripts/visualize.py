#!/usr/bin/env python
"""Stage 4 qualitative CLI — build real vs. synthetic comparison grids.

Usage:
    venv/bin/python scripts/visualize.py --model sd15 --cfg 5 --n 8
"""

from __future__ import annotations

import argparse

from cytodiff.eval.visualize import make_comparison_grid


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff qualitative grids.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--cfg", default="5", help="CFG folder suffix (matches samples/<model>/cfg_<cfg>)")
    ap.add_argument("--n", type=int, default=8, help="images per row")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_comparison_grid(args.model, args.cfg, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
