# pre_process_image.py
"""
Pre‑process two sets of images for mapping:
 1) HWP에서 추출된 이미지
 2) 그 이미지의 원본

Usage
-----
python pre_process_image.py \
  path/to/extracted_src \
  path/to/processed_extracted \
  path/to/original_src \
  path/to/processed_original \
  --size 512

This version appends the original extension into the processed filename
so downstream scripts can recover it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def auto_crop(gray: np.ndarray, tol: int = 5) -> np.ndarray:
    mask = gray > tol
    if not mask.any():
        return gray
    coords = np.argwhere(mask)
    y0, x0 = coords.min(0)
    y1, x1 = coords.max(0) + 1
    return gray[y0:y1, x0:x1]


def preprocess(src: Path, dst: Path, size: int = 512) -> None:
    img = Image.open(src).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize((size, size), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    cropped = auto_crop(gray)
    pad_y = (size - cropped.shape[0]) // 2
    pad_x = (size - cropped.shape[1]) // 2
    gray = cv2.copyMakeBorder(
        cropped,
        pad_y,
        size - cropped.shape[0] - pad_y,
        pad_x,
        size - cropped.shape[1] - pad_x,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    # save using PIL to support unicode paths
    from PIL import Image as PILImage
    PILImage.fromarray(gray).save(dst, format="PNG")


def rename_extracted(fp: Path, root: Path) -> str:
    # keep original ext in filename
    return fp.stem + fp.suffix + ".png"


def rename_original(fp: Path, root: Path) -> str:
    rel = fp.relative_to(root)
    base = rel.with_suffix("")  # no ext
    # join path parts with '_' and append original ext
    name = base.as_posix().replace("/", "_") + fp.suffix + ".png"
    return name


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Preprocess extracted vs original images for mapping"
    )
    pa.add_argument("extracted_src", help="folder of HWP‑extracted images")
    pa.add_argument("dst_extracted", help="where to save processed extracted images")
    pa.add_argument("original_src", help="folder of corresponding original images")
    pa.add_argument("dst_original", help="where to save processed original images")
    pa.add_argument("--size", type=int, default=512, help="output square size")
    args = pa.parse_args()

    # Debug: print input arguments
    print(f"[ARGS] extracted_src={args.extracted_src}, dst_extracted={args.dst_extracted}")
    print(f"[ARGS] original_src={args.original_src}, dst_original={args.dst_original}, size={args.size}")

    src_ext = Path(args.extracted_src)
    dst_ext = Path(args.dst_extracted)
    src_ori = Path(args.original_src)
    dst_ori = Path(args.dst_original)
    size = args.size

    # Counters for logging
    cnt_ext = 0
    cnt_ori = 0

    # Process extracted images
    for fp in src_ext.rglob("*"):
        if fp.suffix.lower() not in IMG_EXTS:
            continue
        new_name = rename_extracted(fp, src_ext)
        dst = dst_ext / new_name
        print(f"[EXTRACT] {fp.name} -> {new_name}")
        preprocess(fp, dst, size)
        cnt_ext += 1
    print(f"[DONE] processed {cnt_ext} extracted images -> {dst_ext}")

    # Process original images
    for fp in src_ori.rglob("*"):
        if fp.suffix.lower() not in IMG_EXTS:
            continue
        new_name = rename_original(fp, src_ori)
        dst = dst_ori / new_name
        print(f"[ORIG]    {fp.relative_to(src_ori).as_posix()} -> {new_name}")
        preprocess(fp, dst, size)
        cnt_ori += 1
    print(f"[DONE] processed {cnt_ori} original images -> {dst_ori}")

if __name__ == "__main__":
    main()
    main()