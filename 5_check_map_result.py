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
        help="JSON mapping file produced by 4_verify_mapping.py (key: orig_rel, value: cand_name)"
    )
    pa.add_argument(
        "extracted_dir",
        help="Directory containing raw HWP/HWPX-extracted images (images_output)"
    )
    pa.add_argument(
        "original_root",
        help="Root directory containing raw original images (target_data/자동등록 사진 모음)"
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

    # Reset output folder
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(mapping)

    for idx, (orig_key_flat, cand_name) in enumerate(mapping.items(), start=1):
        print(f"[{idx}/{total}] 처리: orig_key='{orig_key_flat}', cand='{cand_name}'")
        # Reconstruct raw extracted filename: replace '_bmp.png' -> '.bmp'
        raw_ex_filename = cand_name.replace('_bmp.png', '.bmp')
        # Locate extracted file
        ex_candidates = list(extracted_root.rglob(raw_ex_filename))
        if not ex_candidates:
            print(f"[WARN] Extracted not found: {raw_ex_filename}")
            continue
        e_file = ex_candidates[0]

        # Parse orig_key_flat: drop prefix, parse subdir, base, ext
        root_sanit = original_root.as_posix().replace('/', '_').replace('\\', '_')
        if not orig_key_flat.startswith(root_sanit + '_'):
            print(f"[WARN] Unexpected orig_key format: {orig_key_flat}")
            continue
        suffix = orig_key_flat[len(root_sanit)+1:]
        parts = suffix.rsplit('_', 2)
        if len(parts) != 3:
            print(f"[WARN] Cannot split orig_key suffix: {suffix}")
            continue
        subdir, base_o, ext_tok = parts
        ext_actual = ext_tok.split('.', 1)[0]
        orig_filename = f"{base_o}.{ext_actual}"
        orig_path = original_root / subdir / orig_filename
        if not orig_path.exists():
            print(f"[WARN] Original not found at: {orig_path}")
            continue

        # Copy files with clear naming
        # Changed: use stem of orig_key_flat to drop '.png' from prefix
        prefix = f"{Path(orig_key_flat).stem}_{Path(raw_ex_filename).stem}"
        ex_name = f"{prefix}_추출{e_file.suffix}"
        ori_name = f"{prefix}_원본{orig_path.suffix}"
        shutil.copy(e_file, dst_root / ex_name)
        shutil.copy(orig_path, dst_root / ori_name)
        print(f"[COPY] {ex_name}, {ori_name}")
        count += 1

    print(f"[DONE] Copied {count}/{total} pairs to {dst_root}")

if __name__ == "__main__":
    main()
