"""Stage 1a: image standardization.

Raw JPEGs (~2560x2048) -> center-cropped square -> 512x512 RGB master copy (lossless PNG).
Per-model upscaling (e.g. 1024 for SDXL) happens at train time from this master, so all
models share identical underlying content. Pixel normalization to [-1, 1] is applied in the
dataloader, not baked into the saved files.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cytodiff.constants import CLASSES, MASTER_RESOLUTION, PROCESSED_DIR, RAW_DIR

RAW_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_raw_images(class_name: str):
    """Yield raw image paths for a class, skipping the .gitkeep placeholder."""
    class_dir = RAW_DIR / class_name
    for p in sorted(class_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS:
            yield p


def is_valid_image(path: Path) -> bool:
    """Return True if the file decodes without error (corruption check, §3.2.2)."""
    try:
        with Image.open(path) as im:
            im.verify()  # verify() consumes the file object, so we reopen to use it
        return True
    except Exception:
        return False


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Center-crop to the largest centered square (§3.2.2 aspect-ratio handling)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def standardize_image(src: Path, dst: Path, size: int = MASTER_RESOLUTION) -> None:
    """Open as RGB, center-crop square, bicubic-resize to size, save lossless PNG."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        im = center_crop_to_square(im)
        im = im.resize((size, size), Image.Resampling.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG")


def run_preprocess(size: int = MASTER_RESOLUTION, overwrite: bool = False) -> dict:
    """Standardize every raw image per class. Returns {class: {processed, skipped_corrupt}}."""
    report: dict[str, dict[str, int]] = {}
    for cls in CLASSES:
        out_dir = PROCESSED_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        processed = skipped = 0
        for src in iter_raw_images(cls):
            dst = out_dir / f"{src.stem}.png"
            if dst.exists() and not overwrite:
                processed += 1
                continue
            if not is_valid_image(src):
                skipped += 1
                print(f"  [skip corrupt] {src}")
                continue
            standardize_image(src, dst, size=size)
            processed += 1
        report[cls] = {"processed": processed, "skipped_corrupt": skipped}
        print(f"  {cls}: {processed} processed, {skipped} skipped (-> {out_dir})")
    return report
