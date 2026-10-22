"""Stage 1b — patient-level data split + manifest.

LungHist700 ships data.csv, which links every image to an anonymized patient id. Patients, not
images, are assigned to the held-out evaluation set, so no patient contributes images to both
training and evaluation. Both classes get a held-out set of about HELDOUT_FRAC of their images.
Constraints of the (deterministic, seeded) patient search:
  * every differentiation grade (well / moderately / poorly) of each class appears in both the
    training and the held-out partition;
  * a patient with both ADC and SCC images falls wholly on one side (it is one patient).

Manifest columns:
    path            relative-to-repo path of the 512 processed PNG
    class           ADC | SCC
    label           per-class caption (cytodiff.data.caption)
    patient_id      LungHist700 patient id (from data.csv)
    grade           bd | md | pd  (well / moderately / poorly differentiated)
    split           train | eval
    eval_reference  True for held-out images (the real FID/KID reference distribution)
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from cytodiff.constants import (
    CLASSES, HELDOUT_FRAC, LUNGHIST700_PATIENT_CSV, PROCESSED_DIR, PROJECT_ROOT, SPLIT_SEED,
    SPLITS_DIR,
)
from cytodiff.data.caption import caption_for

MANIFEST_PATH = SPLITS_DIR / "manifest.csv"
GRADES = ("bd", "md", "pd")


def load_patient_table(csv_path: Path = LUNGHIST700_PATIENT_CSV) -> dict[str, tuple[str, str]]:
    """Map an image stem (e.g. 'aca_bd_20x_0') to (patient_id, grade) from LungHist700 data.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. It ships inside the LungHist700 archive "
            "(https://doi.org/10.6084/m9.figshare.25459174) as data/data.csv."
        )
    df = pd.read_csv(csv_path, dtype=str)
    stems = df["superclass"] + "_" + df["subclass"] + "_" + df["resolution"] + "_" + df["image_id"]
    return dict(zip(stems, zip(df["patient_id"], df["subclass"])))


def select_heldout_patients(
    images: pd.DataFrame, frac: float = HELDOUT_FRAC, seed: int = SPLIT_SEED, iters: int = 200_000
) -> set[str]:
    """Pick the held-out patients. `images` has one row per image: class, patient_id, grade.

    Seeded random search over patient subsets; keeps the subset whose held-out share of each
    class (and, with a smaller weight, of each grade) is closest to `frac`, subject to every
    grade of every class being present on both sides. Deterministic for a given seed and input.
    """
    count = defaultdict(Counter)                    # patient -> class -> n images
    gcount = defaultdict(Counter)                   # patient -> (class, grade) -> n images
    grades = defaultdict(set)                       # (class, patient) -> grades
    for cls, patient, grade in zip(images["class"], images["patient_id"], images["grade"]):
        count[patient][cls] += 1
        gcount[patient][(cls, grade)] += 1
        grades[(cls, patient)].add(grade)
    total = {c: sum(count[p][c] for p in count) for c in CLASSES}
    gtotal = Counter()
    for p in gcount:
        gtotal.update(gcount[p])
    patients = sorted(count, key=lambda p: (len(p), p))
    all_grades = {c: {g for (cc, _), gs in grades.items() if cc == c for g in gs} for c in CLASSES}

    def grades_of(cls, ps):
        return {g for p in ps for g in grades.get((cls, p), ())}

    rng = random.Random(seed)
    best, best_err = None, float("inf")
    for _ in range(iters):
        held = set(rng.sample(patients, rng.randint(2, len(patients) - 2)))
        kept = set(patients) - held
        if any(grades_of(c, held) != all_grades[c] or grades_of(c, kept) != all_grades[c]
               for c in CLASSES):
            continue
        # Primary: each class's held-out share; secondary (weight 0.1): each grade's share.
        err = sum(abs(sum(count[p][c] for p in held) / total[c] - frac) for c in CLASSES)
        err += 0.1 * sum(abs(sum(gcount[p][k] for p in held) / gtotal[k] - frac) for k in gtotal)
        if err < best_err:
            best, best_err = held, err
    if best is None:
        raise RuntimeError("No patient split satisfies the grade constraints.")
    return best


def build_manifest(heldout_frac: float = HELDOUT_FRAC, seed: int = SPLIT_SEED,
                   write: bool = True) -> pd.DataFrame:
    """Scan processed images, assign a patient-level split, and (optionally) write manifest.csv."""
    table = load_patient_table()
    rows = []
    for cls in CLASSES:
        images = sorted((PROCESSED_DIR / cls).glob("*.png"))
        if not images:
            raise FileNotFoundError(
                f"No processed images for {cls} in {PROCESSED_DIR / cls}. Run preprocessing first."
            )
        label = caption_for(cls)
        for img in images:
            if img.stem not in table:
                raise KeyError(f"{img.stem} is not listed in {LUNGHIST700_PATIENT_CSV}")
            patient, grade = table[img.stem]
            rows.append({
                "path": str(img.relative_to(PROJECT_ROOT)), "class": cls, "label": label,
                "patient_id": patient, "grade": grade,
            })
    df = pd.DataFrame(rows)

    held = select_heldout_patients(df, frac=heldout_frac, seed=seed)
    df["split"] = df["patient_id"].map(lambda p: "eval" if p in held else "train")
    df["eval_reference"] = df["split"] == "eval"

    # Guard: no patient on both sides.
    sides = df.groupby("patient_id")["split"].nunique()
    assert (sides == 1).all(), "a patient appears in both training and evaluation"

    if write:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(MANIFEST_PATH, index=False)
        print(f"  wrote {MANIFEST_PATH} ({len(df)} rows)")
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print a per-class train/eval breakdown in images and patients."""
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        for split in ("train", "eval"):
            s = sub[sub["split"] == split]
            print(f"  {cls} {split:5s}: {len(s):3d} images from {s['patient_id'].nunique():2d} patients"
                  f"  (grades {dict(Counter(s['grade']))})")
    held = sorted(df.loc[df["split"] == "eval", "patient_id"].unique(), key=int)
    print(f"  held-out patients: {', '.join(held)}")
