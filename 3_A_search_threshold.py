# 3_A_search_threshold.py
"""
Threshold search script: identity-based distance distributions for pHash and Color Histogram
• Computes distances between processed extracted vs. processed original images of the same identity
  - pHash: gray_extracted ↔ gray_original
  - Hist: color_extracted ↔ color_original
• Saves CSVs and generates histograms & CDFs to suggest thresholds
• All results are stored in the 'identity_dist' directory

Usage:
  python 3_A_search_threshold.py \
    <gray_extracted_dir> \
    <gray_original_dir> \
    <color_extracted_dir> \
    <color_original_dir> \
    --mapping preprocess_mapping.json \
    [--dry_run]
"""
import argparse
import json
import csv
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np
import matplotlib.pyplot as plt

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute identity-based distance distributions and suggest thresholds"
    )
    parser.add_argument(
        "gray_extracted_dir",
        help="Directory of processed extracted gray images"
    )
    parser.add_argument(
        "gray_original_dir",
        help="Directory of processed original gray images"
    )
    parser.add_argument(
        "color_extracted_dir",
        help="Directory of processed extracted color images"
    )
    parser.add_argument(
        "color_original_dir",
        help="Directory of processed original color images"
    )
    parser.add_argument(
        "--mapping", required=True,
        help="Path to preprocess_mapping.json"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only compute/save distances; skip plotting and suggestions"
    )
    return parser.parse_args()


def load_mapping(path: str) -> dict:
    """Load JSON mapping from processed filename to metadata."""
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"[INFO] Loaded mapping from {path}, entries: {len(mapping)}")
    return mapping


def compute_phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img)


def compute_color_hist(path: Path) -> np.ndarray:
    with Image.open(path) as pil_img:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    chans = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0,256])
        cv2.normalize(h, h)
        chans.append(h.flatten())
    return np.concatenate(chans)


def main():
    args = parse_args()
    print(f"[INFO] Params: gray_ext={args.gray_extracted_dir}, gray_ori={args.gray_original_dir},")
    print(f"        color_ext={args.color_extracted_dir}, color_ori={args.color_original_dir}, mapping={args.mapping}, dry_run={args.dry_run}")
    print("-"*60)

    # Prepare output directory
    output_dir = Path("identity_dist")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Ensured output directory exists: {output_dir}")
    print("-"*60)

    id_map = load_mapping(args.mapping)
    gray_ext = {p.name: p for p in Path(args.gray_extracted_dir).rglob('*') if p.suffix.lower() in SUPPORTED_EXTS}
    gray_ori = {p.name: p for p in Path(args.gray_original_dir).rglob('*') if p.suffix.lower() in SUPPORTED_EXTS}
    color_ext = {p.name: p for p in Path(args.color_extracted_dir).rglob('*') if p.suffix.lower() in SUPPORTED_EXTS}
    color_ori = {p.name: p for p in Path(args.color_original_dir).rglob('*') if p.suffix.lower() in SUPPORTED_EXTS}
    print(f"[INFO] Found: gray_ext={len(gray_ext)}, gray_ori={len(gray_ori)}, color_ext={len(color_ext)}, color_ori={len(color_ori)}")
    print("-"*60)

    # PHASH: gray_extracted ↔ gray_original
    gray_items = [(fn, meta) for fn, meta in id_map.items() if meta.get("채널") == "gray"]
    phash_records = []
    print(f"[INFO] Computing pHash for {len(gray_items)} gray pairs...")
    for idx, (ext_name, meta) in enumerate(gray_items, 1):
        # use preprocessed original filename if available
        orig_rel = meta.get("원본_전체_경로")
        ori_name = meta.get("전처리_원본_파일명") or (Path(orig_rel).name if orig_rel else None)
        print(f"[INFO] pHash pair {idx}/{len(gray_items)}: {ext_name} vs {ori_name}")
        p_ext = gray_ext.get(ext_name)
        p_ori = gray_ori.get(ori_name) if ori_name else None
        if not p_ext or not p_ori:
            print(f"[WARN] Missing gray file for {ext_name} vs {ori_name}")
            continue
        dist = int(compute_phash(p_ext) - compute_phash(p_ori))
        phash_records.append((ext_name, ori_name, dist))
    phash_csv = output_dir / "phash_distances.csv"
    with phash_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["extracted_gray", "original_gray", "phash_dist"])
        w.writerows(phash_records)
    print(f"[SAVE] pHash distances saved to {phash_csv}, total: {len(phash_records)}")
    print("-"*60)

    # HIST: color_extracted ↔ color_original
    color_items = [(fn, meta) for fn, meta in id_map.items() if meta.get("채널") == "color"]
    hist_records = []
    print(f"[INFO] Computing Hist for {len(color_items)} color pairs...")
    for idx, (ext_name, meta) in enumerate(color_items, 1):
        orig_rel = meta.get("원본_전체_경로")
        ori_name = meta.get("전처리_원본_파일명") or (Path(orig_rel).name if orig_rel else None)
        print(f"[INFO] Hist pair {idx}/{len(color_items)}: {ext_name} vs {ori_name}")
        p_ext = color_ext.get(ext_name)
        p_ori = color_ori.get(ori_name) if ori_name else None
        if not p_ext or not p_ori:
            print(f"[WARN] Missing color file for {ext_name} vs {ori_name}")
            continue
        hd = float(cv2.compareHist(
            compute_color_hist(p_ext), compute_color_hist(p_ori), cv2.HISTCMP_BHATTACHARYYA
        ))
        hist_records.append((ext_name, ori_name, hd))
    hist_csv = output_dir / "hist_distances.csv"
    with hist_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["extracted_color", "original_color", "hist_dist"])
        w.writerows(hist_records)
    print(f"[SAVE] Hist distances saved to {hist_csv}, total: {len(hist_records)}")
    print("-"*60)

    if args.dry_run:
        print("[DONE] Dry run complete. Exiting before plotting.")
        return

    # Prepare value lists and check
    phash_vals = [d for *_, d in phash_records]
    if not phash_vals:
        print("[ERROR] No valid pHash distances found.")
        return
    hist_vals = [h for *_, h in hist_records]
    if not hist_vals:
        print("[ERROR] No valid Histogram distances found.")
        return

    # Plotting & thresholds
    print("[PLOT] Generating histograms and CDFs...")
    # pHash plot
    plt.figure()
    plt.hist(phash_vals, bins=50, log=True)
    p_hist_png = output_dir / "phash_histogram.png"
    plt.savefig(p_hist_png)
    sorted_p = sorted(phash_vals)
    cdf_p = np.linspace(0, 1, len(sorted_p))
    plt.figure()
    plt.plot(sorted_p, cdf_p)
    p_cdf_png = output_dir / "phash_cdf.png"
    plt.savefig(p_cdf_png)
    thresh_p = np.percentile(phash_vals, 98)
    print(f"[THRESHOLD] pHash 98% = {thresh_p}")
    print(f"[SAVE] pHash plots saved: {p_hist_png}, {p_cdf_png}")
    print("-"*60)
    # Hist plot
    plt.figure()
    plt.hist(hist_vals, bins=50, log=True)
    h_hist_png = output_dir / "histogram.png"
    plt.savefig(h_hist_png)
    sorted_h = sorted(hist_vals)
    cdf_h = np.linspace(0, 1, len(sorted_h))
    plt.figure()
    plt.plot(sorted_h, cdf_h)
    h_cdf_png = output_dir / "hist_cdf.png"
    plt.savefig(h_cdf_png)
    thresh_h = np.percentile(hist_vals, 97)
    print(f"[THRESHOLD] Hist 97% = {thresh_h}")
    print(f"[SAVE] Hist plots saved: {h_hist_png}, {h_cdf_png}")
    print("[DONE] Analysis complete.")

if __name__ == '__main__':
    main()
