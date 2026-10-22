#!/usr/bin/env python
"""Stage 1 CLI — standardize raw images, then build the patient-level train/eval manifest.

Usage:
    venv/bin/python scripts/prepare_data.py                # full run
    venv/bin/python scripts/prepare_data.py --overwrite    # re-standardize existing
    venv/bin/python scripts/prepare_data.py --skip-preprocess   # rebuild manifest only
"""

from __future__ import annotations

import argparse

from cytodiff.constants import HELDOUT_FRAC, MASTER_RESOLUTION, SPLIT_SEED
from cytodiff.data.preprocess import run_preprocess
from cytodiff.data.split import build_manifest, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description="CytoDiff stage 1: preprocess + split.")
    ap.add_argument("--size", type=int, default=MASTER_RESOLUTION, help="master square resolution")
    ap.add_argument("--heldout-frac", type=float, default=HELDOUT_FRAC,
                    help="target share of each class held out (by patient)")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED, help="split seed")
    ap.add_argument("--overwrite", action="store_true", help="re-standardize images that already exist")
    ap.add_argument("--skip-preprocess", action="store_true", help="only rebuild the manifest")
    args = ap.parse_args()

    if not args.skip_preprocess:
        print("== Stage 1a: standardization ==")
        run_preprocess(size=args.size, overwrite=args.overwrite)

    print("== Stage 1b/1c: split + captions -> manifest ==")
    df = build_manifest(heldout_frac=args.heldout_frac, seed=args.seed, write=True)
    summarize(df)
    print("Done.")


if __name__ == "__main__":
    main()
