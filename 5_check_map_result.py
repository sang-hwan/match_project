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

    # Prepare sanitized root string for splitting
    root_sanit = original_root.as_posix().replace("/", "_").replace("\\", "_")

    # Reset output folder (flat)
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(mapping)

    for idx, (orig_key_flat, cand_key) in enumerate(mapping.items(), start=1):
        # cand_key sample: '044_BinData_BIN002B_bmp'
        raw_ex_key = Path(cand_key).name
        # split processed extracted key into base and extension token
        if "_" not in raw_ex_key:
            print(f"[WARN] Cannot parse extracted key: {raw_ex_key}")
            continue
        ex_base, ex_ext = raw_ex_key.rsplit("_", 1)
        ex_filename = f"{ex_base}.{ex_ext}"

        sanitized_orig = orig_key_flat  # e.g. 'target_data_..._23_jpg'
        print(f"[{idx}/{total}] Pairing extracted='{ex_filename}' with original key='{orig_key_flat}'")

        # Reconstruct actual raw original path
        if not orig_key_flat.startswith(root_sanit + "_"):
            print(f"[WARN] Cannot parse orig_key (unexpected prefix): {orig_key_flat}")
            continue
        suffix = orig_key_flat[len(root_sanit) + 1:]
        # suffix sample: '1. 냉동기_23_jpg'
        parts = suffix.rsplit("_", 2)
        if len(parts) != 3:
            print(f"[WARN] Cannot split orig_key into subdir, base, ext: {suffix}")
            continue
        subdir, base_o, ext_o = parts
        orig_filename = f"{base_o}.{ext_o}"
        orig_path = original_root / subdir / orig_filename

        if not orig_path.exists():
            print(f"[WARN] Original not found at: {orig_path}")
            continue

        # Locate the extracted file by exact filename
        e_candidates = list(extracted_root.rglob(ex_filename))
        if not e_candidates:
            print(f"[WARN] Extracted not found: {ex_filename}")
            continue
        e_file = e_candidates[0]
        o_file = orig_path

        # Copy with naming: <orig_key_flat>_<raw_ex_key>_추출.<ext> and _원본.<ext>
        prefix = f"{sanitized_orig}_{raw_ex_key}"
        ex_name = f"{prefix}_추출{e_file.suffix}"
        ori_name = f"{prefix}_원본{o_file.suffix}"
        shutil.copy(e_file, dst_root / ex_name)
        shutil.copy(o_file, dst_root / ori_name)
        print(f"[COPY] {ex_name}, {ori_name}")
        count += 1

    print(f"[DONE] Copied {count}/{total} pairs to {dst_root}")


if __name__ == "__main__":
    main()
