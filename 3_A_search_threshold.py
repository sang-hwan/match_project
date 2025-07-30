# 3_A_search_threshold.py
"""
Threshold‑search script (de‑identified mapping version)
======================================================

Purpose
-------
Given pre‑processed *extracted* images (from HWP BinData) and their
candidate *original* images, this script measures similarity‑distance
distributions for two metrics:

* Gray‑scale images → pHash Hamming distance
* Color images      → HSV‑histogram Bhattacharyya distance

It reads a mapping JSON file (`preprocess_mapping.json`) that records,
for each processed filename, its *track* (“low” or “high”), *channel*
(“gray” or “color”), and the absolute source path (`원본_전체_경로`).
If the source path contains **“BinData”** the file is treated as
*extracted*; otherwise it is treated as *original*.

For every (track, channel) pair, the script computes *all* cross‑product
pairs extracted × original, writes the raw distances to CSV, optionally
plots histograms and CDFs, and finally prints percentile‑based threshold
suggestions (default: pHash 98‑th, Hist 97‑th).

Because it looks at the entire distance distribution, the results also
include pairs that are **not** identical images.

Usage
-----
python 3_A_search_threshold.py \
    processed/extracted  processed/original \
    --mapping preprocess_mapping.json [--dry_run]

Arguments
---------
1. extracted_dir : root of processed/extracted
2. original_dir  : root of processed/original
3. --mapping     : path to preprocess_mapping.json
4. --dry_run     : skip plotting (CSV is still written)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute pHash / HSV‑histogram distance distributions "
                    "and suggest thresholds."
    )
    p.add_argument("extracted_dir", help="Root directory of processed/extracted")
    p.add_argument("original_dir",  help="Root directory of processed/original")
    p.add_argument("--mapping", required=True, help="preprocess_mapping.json")
    p.add_argument("--dry_run", action="store_true",
                   help="Skip plotting; still writes CSV files")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def norm_channel(ch: str) -> str:
    """Normalize channel labels to lowercase 'gray' or 'color'."""
    t = ch.strip().lower()
    if t in {"gray", "grey", "그레이", "gray_g"}:
        return "gray"
    if t in {"color", "colour", "컬러"}:
        return "color"
    return t  # fall through for unexpected values


def compute_phash(path: Path) -> imagehash.ImageHash:
    """Return the perceptual hash (pHash) of an image."""
    with Image.open(path) as img:
        return imagehash.phash(img)


def compute_color_hist(path: Path) -> np.ndarray:
    """Return a 96‑D HSV histogram feature (32 bins × 3 channels)."""
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
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args    = parse_args()
    ex_root = Path(args.extracted_dir).resolve()
    or_root = Path(args.original_dir).resolve()

    print(f"[PARAM] extracted_dir : {ex_root}")
    print(f"[PARAM] original_dir  : {or_root}")
    print(f"[PARAM] mapping       : {args.mapping}")
    print(f"[PARAM] dry_run       : {args.dry_run}")
    print("-" * 60)

    out_dir = Path("identity_dist")
    out_dir.mkdir(exist_ok=True)
    print(f"[INFO] Output directory ready: {out_dir}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  1) Load mapping & split into extracted / original groups
    # ----------------------------------------------------------------------- #
    with open(args.mapping, encoding="utf-8") as f:
        mp = json.load(f)

    extracted = defaultdict(list)  # {(track, channel): [Path, …]}
    original  = defaultdict(list)

    for proc_name, meta in mp.items():
        track   = meta.get("트랙", "").lower()          # "low" | "high"
        channel = norm_channel(meta.get("채널", ""))    # "gray" | "color"
        if not track or not channel:
            continue
        key = (track, channel)

        # Determine group by substring check
        is_ext = "BinData" in meta.get("원본_전체_경로", "")

        file_subpath = Path(track) / channel / proc_name  # e.g. low/gray/000001.png
        full_path    = (ex_root / file_subpath) if is_ext else (or_root / file_subpath)

        if not full_path.is_file():
            print(f"[WARN] File missing on disk: {full_path}")
            continue

        (extracted if is_ext else original)[key].append(full_path)

    # Stats
    count_dict = lambda d: {k: len(v) for k, v in d.items()}
    print(f"[INFO] Extracted file counts (track,channel): {count_dict(extracted)}")
    print(f"[INFO] Original  file counts (track,channel): {count_dict(original)}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  2) Compute distances
    # ----------------------------------------------------------------------- #
    phash_csv = out_dir / "phash_distances.csv"
    hist_csv  = out_dir / "hist_distances.csv"
    phash_rows, hist_rows = [], []

    total_pairs = sum(len(extracted[k]) * len(original[k])
                      for k in extracted if k in original)
    print(f"[INFO] Total pair candidates: {total_pairs:,}")
    print("-" * 60)

    processed_pairs = 0
    prog_step = max(1, total_pairs // 100)  # 1 % progress updates

    for (track, channel), ext_list in extracted.items():
        ori_list = original.get((track, channel), [])
        if not ori_list:
            continue

        if channel == "gray":
            # Pre‑compute pHash
            h_ext = {p: compute_phash(p) for p in ext_list}
            h_ori = {p: compute_phash(p) for p in ori_list}

            for p_ext, hash_ext in h_ext.items():
                for p_ori, hash_ori in h_ori.items():
                    dist = int(hash_ext - hash_ori)
                    phash_rows.append(
                        (track, channel, p_ext.as_posix(), p_ori.as_posix(), dist)
                    )
                    processed_pairs += 1
                    if processed_pairs % prog_step == 0:
                        perc = processed_pairs * 100 / total_pairs
                        print(f"[PROGRESS] {processed_pairs}/{total_pairs}  ({perc:5.1f} %)")

        elif channel == "color":
            # Pre‑compute color histograms
            h_ext = {p: compute_color_hist(p) for p in ext_list}
            h_ori = {p: compute_color_hist(p) for p in ori_list}

            for p_ext, feat_ext in h_ext.items():
                for p_ori, feat_ori in h_ori.items():
                    dist = float(cv2.compareHist(feat_ext, feat_ori,
                                                 cv2.HISTCMP_BHATTACHARYYA))
                    hist_rows.append(
                        (track, channel, p_ext.as_posix(), p_ori.as_posix(), dist)
                    )
                    processed_pairs += 1
                    if processed_pairs % prog_step == 0:
                        perc = processed_pairs * 100 / total_pairs
                        print(f"[PROGRESS] {processed_pairs}/{total_pairs}  ({perc:5.1f} %)")

    # ----------------------------------------------------------------------- #
    #  3) Save results
    # ----------------------------------------------------------------------- #
    with phash_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [("track", "channel", "extracted", "original", "phash_distance"),
             *phash_rows]
        )
    print(f"[SAVE] pHash rows : {len(phash_rows):,}  →  {phash_csv}")

    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(
            [("track", "channel", "extracted", "original", "hist_distance"),
             *hist_rows]
        )
    print(f"[SAVE] Hist rows  : {len(hist_rows):,}  →  {hist_csv}")
    print("-" * 60)

    # ----------------------------------------------------------------------- #
    #  4) Plot histograms & CDFs
    # ----------------------------------------------------------------------- #
    if args.dry_run:
        print("[DONE] Dry‑run complete – plotting skipped.")
        return

    # pHash
    if phash_rows:
        phash_vals = [row[-1] for row in phash_rows]
        plt.figure(); plt.hist(phash_vals, bins=60, log=True)
        plt.title("pHash distance distribution")
        plt.xlabel("Hamming distance"); plt.ylabel("Frequency (log)")
        plt.savefig(out_dir / "phash_histogram.png")

        plt.figure(); plt.plot(np.sort(phash_vals),
                               np.linspace(0, 1, len(phash_vals)))
        plt.title("pHash CDF")
        plt.xlabel("Hamming distance"); plt.ylabel("CDF")
        plt.savefig(out_dir / "phash_cdf.png")

        th_p = np.percentile(phash_vals, 98)
        print(f"[THRESHOLD] Suggested pHash (98th pct): {th_p:.0f}")

    # Hist
    if hist_rows:
        hist_vals = [row[-1] for row in hist_rows]
        plt.figure(); plt.hist(hist_vals, bins=60, log=True)
        plt.title("Color‑hist distance distribution")
        plt.xlabel("Bhattacharyya distance"); plt.ylabel("Frequency (log)")
        plt.savefig(out_dir / "hist_histogram.png")

        plt.figure(); plt.plot(np.sort(hist_vals),
                               np.linspace(0, 1, len(hist_vals)))
        plt.title("Color‑hist CDF")
        plt.xlabel("Bhattacharyya distance"); plt.ylabel("CDF")
        plt.savefig(out_dir / "hist_cdf.png")

        th_h = np.percentile(hist_vals, 97)
        print(f"[THRESHOLD] Suggested Hist (97th pct): {th_h:.4f}")

    print("[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
