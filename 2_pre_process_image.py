# 2_pre_process_image.py
"""
Pre-process two sets of images for mapping:
 1) HWP에서 추출된 이미지
 2) 그 이미지의 원본 (선택한 서브폴더만)

Usage
-----
python 2_pre_process_image.py \
  path/to/extracted_src \
  path/to/processed_extracted \
  path/to/original_src_subfolder \
  path/to/processed_original \
  --orig-root path/to/original_root \
  --size 512 \
  --min-pixels 0

This version performs only EXIF correction, grayscale conversion,
linear normalization, and aspect-preserving resize.
Originals keep full path information in flat filenames (underscores).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # ensure UTF-8 output

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

def preprocess(
    src: Path,
    dst: Path,
    size: int,
    min_pixels: int,
) -> bool:
    """
    - EXIF orientation correction
    - Grayscale conversion
    - Linear normalization 0-255
    - Aspect-preserving resize (long side -> size)
    """
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if w * h < min_pixels:
            print(f"[SKIP] {src.name} ({w}×{h} < {min_pixels})")
            return False

        # Convert to grayscale
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # Linear normalization
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Aspect-preserving resize
        if w >= h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        resized = cv2.resize(norm, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save as PNG
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(resized).save(str(dst), format="PNG")
        print(f"[OK]    {src.name} -> {dst.name}")
        return True

    except Exception as e:
        print(f"[FAIL]  {src.name}: {e}")
        return False


def rename_original(fp: Path, orig_root: Path) -> str:
    """
    Construct flat filename with full relative path parts from orig_root,
    joined by underscores, preserving original extension before .png
    """
    parts = fp.relative_to(orig_root).parts
    flat = '_'.join(parts[:-1] + (fp.stem,))
    return f"{orig_root.name}_{flat}{fp.suffix}.png"


def main() -> None:
    pa = argparse.ArgumentParser(description="Simplified image preprocessing for mapping")
    pa.add_argument("extracted_src", help="HWP-extracted images folder")
    pa.add_argument("dst_extracted", help="Processed extracted images output folder")
    pa.add_argument("original_src", help="Subfolder under original root to process")
    pa.add_argument("dst_original", help="Processed original images output folder")
    pa.add_argument("--orig-root", required=True, help="Original root folder for naming")
    pa.add_argument("--size", type=int, default=512, help="Max dimension for resize")
    pa.add_argument("--min-pixels", type=int, default=0, help="Minimum pixel area to process")
    args = pa.parse_args()

    src_ext = Path(args.extracted_src)
    dst_ext = Path(args.dst_extracted)
    src_ori = Path(args.original_src)
    dst_ori = Path(args.dst_original)
    orig_root = Path(args.orig_root)
    size = args.size
    min_px = args.min_pixels

    # Process extracted images
    print(f"[START] Preprocessing extracted images from {src_ext}")
    ext_files = [p for p in src_ext.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[INFO] Found {len(ext_files)} extracted images")
    cnt_ext = 0
    for idx, fp in enumerate(ext_files, start=1):
        print(f"[{idx}/{len(ext_files)}] Extracted: {fp.name}")
        dst = dst_ext / f"{fp.name}.png"
        if preprocess(fp, dst, size, min_px):
            cnt_ext += 1
    print(f"[DONE] Extracted processed: {cnt_ext}/{len(ext_files)}")

    # Process original images
    print(f"[START] Preprocessing originals in {src_ori} (root: {orig_root})")
    ori_files = [p for p in src_ori.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[INFO] Found {len(ori_files)} original images")
    cnt_ori = 0
    for idx, fp in enumerate(ori_files, start=1):
        rel = fp.relative_to(orig_root)
        print(f"[{idx}/{len(ori_files)}] Original: {rel}")
        filename = rename_original(fp, orig_root)
        dst = dst_ori / filename
        if preprocess(fp, dst, size, min_px):
            cnt_ori += 1
    print(f"[DONE] Originals processed: {cnt_ori}/{len(ori_files)} -> {dst_ori}")


if __name__ == "__main__":
    main()
