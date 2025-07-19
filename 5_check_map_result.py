# 5_check_map_result.py
"""
Collect mapped pairs into a single folder for visual inspection of raw extracted and original images.

Usage
-----
python 5_check_map_result.py \
  map.json \
  raw_extracted_dir \
  raw_original_root_dir \
  out_dir
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Gather mapping pairs into a flat folder for inspection"
    )
    pa.add_argument(
        "map_json",
        help="JSON mapping file produced by 4_verify_mapping.py (key: processed original, value: processed extracted)"
    )
    pa.add_argument(
        "extracted_dir",
        help="Directory containing raw HWP/HWPX-extracted images"
    )
    pa.add_argument(
        "original_root",
        help="Root directory containing raw original images (may have subfolders)"
    )
    pa.add_argument(
        "out_dir",
        help="Directory to copy paired images for inspection"
    )
    args = pa.parse_args()

    # Load mapping
    print(f"[START] Loading mapping JSON from {args.map_json}...")
    with open(args.map_json, encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)
    print(f"[INFO] Loaded {len(mapping)} mapping entries")

    extracted_root = Path(args.extracted_dir)
    original_root = Path(args.original_root)
    dst_root = Path(args.out_dir)

    # Reset output folder (flat)
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(mapping)
    for idx, (orig_key, cand_key) in enumerate(mapping.items(), start=1):
        # Derive raw filenames
        raw_ex = Path(cand_key).stem    # e.g. '032_BinData_BIN001F.bmp'
        raw_orig = Path(orig_key).stem.split('_')[-1]  # e.g. '1.jpg'
        print(f"[{idx}/{total}] Pairing extracted='{raw_ex}' with original='{raw_orig}'")

        # Locate extracted file
        e_candidates = list(extracted_root.rglob(raw_ex))
        if not e_candidates:
            print(f"[WARN] Extracted not found: {raw_ex}")
            continue
        e_file = e_candidates[0]

        # Locate original file
        o_candidates = list(original_root.rglob(raw_orig))
        if not o_candidates:
            print(f"[WARN] Original not found: {raw_orig}")
            continue
        o_file = o_candidates[0]

        # Copy directly into dst_root
        ex_name = f"{raw_ex}_{raw_orig}_추출{e_file.suffix}"
        ori_name = f"{raw_ex}_{raw_orig}_원본{o_file.suffix}"
        out_ex = dst_root / ex_name
        out_ori = dst_root / ori_name
        shutil.copy(e_file, out_ex)
        shutil.copy(o_file, out_ori)
        print(f"[COPY] {ex_name}, {ori_name}")
        count += 1

    print(f"[DONE] Copied {count}/{total} pairs to {dst_root}")


if __name__ == "__main__":
    main()
