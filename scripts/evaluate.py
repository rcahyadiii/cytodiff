#!/usr/bin/env python
"""Stage 4b/5 CLI — per-class KID (patient-bootstrap CIs), FD, precision/recall, density/coverage
across the CFG sweep, for Inception, UNI and Phikon features; reports the best CFG by macro KID on
Inception. Runs on CUDA, Apple MPS, or CPU.

Usage:
    venv/bin/python scripts/evaluate.py --model sd15
    venv/bin/python scripts/evaluate.py --model sd3_ct_rgb --cfg 5
    venv/bin/python scripts/evaluate.py --real-baseline          # real train vs held-out
"""

from __future__ import annotations

import argparse

from cytodiff.constants import CFG_SWEEP
from cytodiff.eval.evaluate import evaluate_model, evaluate_real_baseline
from cytodiff.eval.metrics import BACKBONES


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff stage 4b/5: distribution metrics.")
    ap.add_argument("--model", help="run name, i.e. the folder under experiments/samples (e.g. sd15, sd3_s1)")
    ap.add_argument("--real-baseline", action="store_true", help="score real train vs held-out images")
    ap.add_argument("--cfg", type=float, nargs="*", default=None, help="override CFG sweep")
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES), choices=BACKBONES)
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap replicates for KID intervals")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    args = ap.parse_args()

    if args.real_baseline:
        evaluate_real_baseline(backbones=tuple(args.backbones), device=args.device, n_boot=args.n_boot)
        return
    if not args.model:
        ap.error("--model is required unless --real-baseline is given")
    cfgs = tuple(args.cfg) if args.cfg else CFG_SWEEP
    evaluate_model(args.model, cfgs=cfgs, backbones=tuple(args.backbones), device=args.device, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
