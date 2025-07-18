# mk_images_info.py
"""
Compute features for mapping and preserve original extension metadata.

Usage
-----
python mk_images_info.py \
  path/to/extracted_dir \
  path/to/originals_dir \
  extract.json \
  origin.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import base64

import cv2
import numpy as np
from PIL import Image
import imagehash

# supported extensions
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
# ORB feature extractor (max 1000 keypoints)
ORB = cv2.ORB_create(1000)


def compute_phash(pil_img: Image.Image) -> str:
    """Compute perceptual hash (pHash) of a PIL image."""
    return str(imagehash.phash(pil_img))


def compute_orb(img: np.ndarray) -> tuple[list[tuple[float, float]], str]:
    """Detect ORB keypoints & descriptors; return coords and base64-encoded descriptor."""
    kp, desc = ORB.detectAndCompute(img, None)
    coords = [(float(k.pt[0]), float(k.pt[1])) for k in kp]
    b64 = base64.b64encode(desc.tobytes()).decode() if desc is not None else None
    return coords, b64


def compute_color_hist(img: np.ndarray) -> list[float]:
    """Compute normalized grayscale histogram (256 bins)."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist.tolist()


def img_to_info(fp: Path) -> dict[str, object]:
    """Read image, compute and return feature dict including original extension metadata."""
    print(f"[INFO] Computing features for: {fp}")
    orig_ext = Path(fp.stem).suffix or fp.suffix
    pil_img = Image.open(fp).convert("L")
    arr = np.array(pil_img)
    # compute ORB once
    coords, orb_b64 = compute_orb(arr)
    return {
        "orig_ext":    orig_ext,
        "phash":       compute_phash(pil_img),
        "kp":          coords,
        "orb":         orb_b64,
        "color_hist":  compute_color_hist(arr),
    }


def process_dir(root: Path) -> dict[str, dict[str, object]]:
    """Process directory of pre-processed PNG images and return feature infos."""
    infos: dict[str, dict[str, object]] = {}
    png_files = list(root.rglob("*.png"))
    total = len(png_files)
    print(f"[STEP] Found {total} PNG files in {root}")
    for idx, fp in enumerate(png_files, start=1):
        key = fp.relative_to(root).as_posix()
        print(f"[STEP] ({idx}/{total}) Processing: {key}")
        try:
            infos[key] = img_to_info(fp)
        except Exception as e:
            print(f"[WARN] Skipping {key}: {e}")
    return infos


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Make extract.json & origin.json with image features"
    )
    pa.add_argument("extracted_dir", help="directory of pre-processed extracted images")
    pa.add_argument("originals_dir", help="directory of pre-processed original images")
    pa.add_argument("extract_json", help="output JSON filename for extracted images (in info_for_map)")
    pa.add_argument("origin_json", help="output JSON filename for original images (in info_for_map)")
    args = pa.parse_args()

    print(f"[ARGS] extracted_dir={args.extracted_dir}, originals_dir={args.originals_dir}")

    # Prepare output directory for JSON files
    out_dir = Path("info_for_map")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] JSON files will be saved under '{out_dir}'")

    # Process extracted images
    print(f"[STEP] Processing extracted images from: {args.extracted_dir}")
    ext_infos = process_dir(Path(args.extracted_dir))
    extract_path = out_dir / args.extract_json
    Path(extract_path).write_text(json.dumps(ext_infos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Saved {len(ext_infos)} extracted entries to {extract_path}")

    # Process original images
    print(f"[STEP] Processing original images from: {args.originals_dir}")
    ori_infos = process_dir(Path(args.originals_dir))
    origin_path = out_dir / args.origin_json
    Path(origin_path).write_text(json.dumps(ori_infos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Saved {len(ori_infos)} original entries to {origin_path}")

if __name__ == "__main__":
    main()
