# mapping_valid.py
"""
Collect mapped pairs into a single folder for visual inspection.

Usage
-----
python mapping_valid.py \
  map.json \
  extracted_dir \
  original_root_dir \
  out_dir
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Gather mapping pairs for eyeballing"
    )
    pa.add_argument(
        "map_json",
        help="JSON mapping file produced by images_mapper.py (keys: extract_key, values: origin_key)"
    )
    pa.add_argument(
        "extracted_dir",
        help="Directory containing pre-processed extracted images"
    )
    pa.add_argument(
        "original_root",
        help="Root directory containing pre-processed original images (may have subfolders)"
    )
    pa.add_argument(
        "out_dir",
        help="Directory to copy paired images for inspection"
    )
    args = pa.parse_args()

    mapping = json.loads(Path(args.map_json).read_text())
    extracted_root = Path(args.extracted_dir)
    original_root = Path(args.original_root)
    dst_root = Path(args.out_dir)

    # reset output folder
    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    count = 0
    total = len(mapping)
    for idx, (e_key, o_key) in enumerate(mapping.items(), start=1):
        # find extracted image file by key (wildcard extension)
        ex_candidates = list(extracted_root.glob(f"{e_key}.*"))
        if not ex_candidates:
            print(f"[WARN] extracted file for '{e_key}' not found in {extracted_root}")
            continue
        e_file = ex_candidates[0]

        # find original image recursively by origin key
        ori_candidates = list(original_root.rglob(f"{o_key}.*"))
        if not ori_candidates:
            print(f"[WARN] original file for '{o_key}' not found under {original_root}")
            continue
        o_file = ori_candidates[0]

        # prepare output subfolder
        pair_dir = dst_root / f"{idx:03d}"
        pair_dir.mkdir()

        # copy with new filenames: key_key_extracted.ext, key_key_original.ext
        out_ex = pair_dir / f"{e_key}_{o_key}_extracted{e_file.suffix}"
        out_ori = pair_dir / f"{e_key}_{o_key}_original{o_file.suffix}"
        shutil.copy(e_file, out_ex)
        shutil.copy(o_file, out_ori)

        count += 1
        print(f"[COPY] {out_ex.name}, {out_ori.name}")

    print(f"[DONE] copied {count}/{total} pairs to {dst_root}")


if __name__ == "__main__":
    main()
