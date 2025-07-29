# 3_A_search_threshold.py
"""
Threshold‑search script
=======================

*   Computes pHash (gray) and Bhattacharyya color‑histogram distances **only**
    between truthfully corresponding pairs:
         processed‑extracted  (BinData_BINxxxx)  ←→  processed‑original
*   Saves distance CSVs and, unless --dry_run, draws histograms / CDFs and
    prints percentile‑based threshold suggestions.

Pair‑finding logic
------------------
1.  Build a reverse index that maps each true‑original basename
    ('1.jpg', '163.bmp', …) **to** its processed filename
    ('003384_g.png', …) **per channel** using information in
    *preprocess_mapping.json*.
2.  For every extracted row we parse the 4‑digit hexadecimal ID embedded in
    “BinData_BINxxxx”. That hex → decimal string gives the original basename.
3.  Lookup the processed‑original via the index; if both processed files
    exist, we measure the distance.

Usage
-----
python 3_A_search_threshold.py \
    <gray_extracted_dir> <gray_original_dir> \
    <color_extracted_dir> <color_original_dir> \
    --mapping preprocess_mapping.json [--dry_run]
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp",
                  ".tif", ".tiff", ".gif"}
ORIG_CANDIDATE_EXTS = [".jpg", ".jpeg", ".png",
                       ".bmp", ".gif", ".tif", ".tiff"]
BIN_RE = re.compile(r"BinData_BIN([0-9A-Fa-f]{4})")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search thresholds for pHash / Hist.")
    p.add_argument("gray_extracted_dir")
    p.add_argument("gray_original_dir")
    p.add_argument("color_extracted_dir")
    p.add_argument("color_original_dir")
    p.add_argument("--mapping", required=True)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def is_extracted(orig_name: str) -> bool:
    """True = came from HWP BinData dump."""
    return "BinData_BIN" in orig_name


def compute_phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img)


def compute_color_hist(path: Path) -> np.ndarray:
    """96‑D HSV histogram."""
    with Image.open(path) as pil:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        cv2.normalize(h, h)
        feats.append(h.flatten())
    return np.concatenate(feats)


# --------------------------------------------------------------------------- #
#  Mapping loader + reverse index builder
# --------------------------------------------------------------------------- #
def load_mapping(path: str):
    with open(path, encoding="utf‑8") as f:
        mp = json.load(f)

    cnt = collections.Counter(
        "extracted" if is_extracted(v["원본_파일명"]) else "original"
        for v in mp.values()
    )
    u_ex = {v["원본_전체_경로"] for v in mp.values()
            if is_extracted(v["원본_파일명"])}
    u_ori = {v["원본_전체_경로"] for v in mp.values()
             if not is_extracted(v["원본_파일명"])}

    print(f"[INFO] Mapping rows                : {len(mp)}")
    print(f"[INFO] Row types (extracted / orig): {cnt['extracted']} / {cnt['original']}")
    print(f"[INFO] Unique source images         : {len(u_ex)} / {len(u_ori)}")
    print("-" * 60)
    return mp


def build_reverse_index(mapping: dict):
    """
    Returns two dicts:

        color_idx[basename] = processed_color_filename
        gray_idx [basename] = processed_gray_filename
    """
    color_idx, gray_idx = {}, {}
    for proc_name, meta in mapping.items():
        if is_extracted(meta["원본_파일명"]):
            continue
        base = meta["원본_파일명"]          # e.g. "163.jpg"
        if meta.get("채널") == "color":
            color_idx[base] = proc_name
        elif meta.get("채널") == "gray":
            gray_idx[base] = proc_name
    print(f"[INFO] Reverse index sizes (gray / color): "
          f"{len(gray_idx)} / {len(color_idx)}")
    print("-" * 60)
    return gray_idx, color_idx


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    print(f"[PARAM] gray_extracted_dir : {args.gray_extracted_dir}")
    print(f"[PARAM] gray_original_dir  : {args.gray_original_dir}")
    print(f"[PARAM] color_extracted_dir: {args.color_extracted_dir}")
    print(f"[PARAM] color_original_dir : {args.color_original_dir}")
    print(f"[PARAM] mapping            : {args.mapping}")
    print(f"[PARAM] dry_run            : {args.dry_run}")
    print("-" * 60)

    out_dir = Path("identity_dist")
    out_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output directory ready: {out_dir}")
    print("-" * 60)

    mapping = load_mapping(args.mapping)
    rev_gray, rev_color = build_reverse_index(mapping)

    # Index processed files on disk
    gray_ext_files = {p.name: p for p in Path(args.gray_extracted_dir).rglob("*")
                      if p.suffix.lower() in SUPPORTED_EXTS}
    gray_ori_files = {p.name: p for p in Path(args.gray_original_dir).rglob("*")
                      if p.suffix.lower() in SUPPORTED_EXTS}
    color_ext_files = {p.name: p for p in Path(args.color_extracted_dir).rglob("*")
                       if p.suffix.lower() in SUPPORTED_EXTS}
    color_ori_files = {p.name: p for p in Path(args.color_original_dir).rglob("*")
                       if p.suffix.lower() in SUPPORTED_EXTS}

    print(f"[INFO] Indexed files  (gray ext / ori): {len(gray_ext_files)} / {len(gray_ori_files)}")
    print(f"[INFO] Indexed files (color ext / ori): {len(color_ext_files)} / {len(color_ori_files)}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  pHash (gray channel)
    # ----------------------------------------------------------------------- #
    phash_records = []
    extracted_gray_rows = [
        (proc_name, meta)
        for proc_name, meta in mapping.items()
        if meta.get("채널") == "gray" and is_extracted(meta["원본_파일명"])
    ]
    print(f"[INFO] pHash: processing {len(extracted_gray_rows)} gray pairs")

    for idx, (proc_name, meta) in enumerate(extracted_gray_rows, 1):
        m = BIN_RE.search(meta.get("원본_전체_경로", ""))
        if not m:
            continue
        dec_id = str(int(m.group(1), 16))         # '00A3' → '163'
        ori_base = next(
            (dec_id + ext for ext in ORIG_CANDIDATE_EXTS if (dec_id + ext) in rev_gray),
            None
        )
        if ori_base is None:
            print(f"[WARN] No original basename found for {proc_name}")
            continue

        ori_proc_name = rev_gray[ori_base]
        p_ext = gray_ext_files.get(proc_name)
        p_ori = gray_ori_files.get(ori_proc_name)

        if not p_ext or not p_ori:
            print(f"[WARN] Missing gray file: {proc_name} / {ori_proc_name}")
            continue

        dist = int(compute_phash(p_ext) - compute_phash(p_ori))
        phash_records.append((proc_name, ori_proc_name, dist))

        if idx % 500 == 0 or idx == len(extracted_gray_rows):
            print(f"[PROGRESS] Gray {idx}/{len(extracted_gray_rows)}")

    phash_csv = out_dir / "phash_distances.csv"
    with phash_csv.open("w", newline="", encoding="utf‑8") as f:
        csv.writer(f).writerows([("extracted_gray", "original_gray", "phash_distance"),
                                 *phash_records])
    print(f"[SAVE] pHash rows written: {len(phash_records)}  →  {phash_csv}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  Hist‑color
    # ----------------------------------------------------------------------- #
    hist_records = []
    extracted_color_rows = [
        (proc_name, meta)
        for proc_name, meta in mapping.items()
        if meta.get("채널") == "color" and is_extracted(meta["원본_파일명"])
    ]
    print(f"[INFO] Hist: processing {len(extracted_color_rows)} color pairs")

    for idx, (proc_name, meta) in enumerate(extracted_color_rows, 1):
        m = BIN_RE.search(meta.get("원본_전체_경로", ""))
        if not m:
            continue
        dec_id = str(int(m.group(1), 16))
        ori_base = next(
            (dec_id + ext for ext in ORIG_CANDIDATE_EXTS if (dec_id + ext) in rev_color),
            None
        )
        if ori_base is None:
            print(f"[WARN] No original basename found for {proc_name}")
            continue

        ori_proc_name = rev_color[ori_base]
        p_ext = color_ext_files.get(proc_name)
        p_ori = color_ori_files.get(ori_proc_name)

        if not p_ext or not p_ori:
            print(f"[WARN] Missing color file: {proc_name} / {ori_proc_name}")
            continue

        dist = float(cv2.compareHist(
            compute_color_hist(p_ext),
            compute_color_hist(p_ori),
            cv2.HISTCMP_BHATTACHARYYA
        ))
        hist_records.append((proc_name, ori_proc_name, dist))

        if idx % 500 == 0 or idx == len(extracted_color_rows):
            print(f"[PROGRESS] Color {idx}/{len(extracted_color_rows)}")

    hist_csv = out_dir / "hist_distances.csv"
    with hist_csv.open("w", newline="", encoding="utf‑8") as f:
        csv.writer(f).writerows([("extracted_color", "original_color", "hist_distance"),
                                 *hist_records])
    print(f"[SAVE] Hist rows written : {len(hist_records)}  →  {hist_csv}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  Plotting / thresholds
    # ----------------------------------------------------------------------- #
    if args.dry_run:
        print("[DONE] Dry‑run complete – plotting skipped.")
        return

    # safety check
    if not phash_records or not hist_records:
        print("[ERROR] No distances collected – aborting plots.")
        return

    # pHash
    phash_vals = [d for *_, d in phash_records]
    plt.figure()
    plt.hist(phash_vals, bins=50, log=True)
    plt.title("pHash distance distribution")
    plt.xlabel("Hamming distance"); plt.ylabel("Frequency (log)")
    plt.savefig(out_dir / "phash_histogram.png")

    plt.figure()
    plt.plot(np.sort(phash_vals), np.linspace(0, 1, len(phash_vals)))
    plt.title("pHash CDF")
    plt.xlabel("Hamming distance"); plt.ylabel("CDF")
    plt.savefig(out_dir / "phash_cdf.png")
    thresh_p = np.percentile(phash_vals, 98)
    print(f"[THRESHOLD] Suggested pHash (98th pct): {thresh_p:.0f}")
    print("-" * 60)

    # Hist
    hist_vals = [d for *_, d in hist_records]
    plt.figure()
    plt.hist(hist_vals, bins=50, log=True)
    plt.title("Color‑hist distance distribution")
    plt.xlabel("Bhattacharyya distance"); plt.ylabel("Frequency (log)")
    plt.savefig(out_dir / "hist_histogram.png")

    plt.figure()
    plt.plot(np.sort(hist_vals), np.linspace(0, 1, len(hist_vals)))
    plt.title("Color‑hist CDF")
    plt.xlabel("Bhattacharyya distance"); plt.ylabel("CDF")
    plt.savefig(out_dir / "hist_cdf.png")
    thresh_h = np.percentile(hist_vals, 97)
    print(f"[THRESHOLD] Suggested Hist (97th pct): {thresh_h:.4f}")
    print("[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
