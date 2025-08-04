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
import argparse, json, shutil
from pathlib import Path

def resolve(path_str: str, hint_root: Path) -> Path | None:
    p = Path(path_str)
    if p.is_file():
        return p
    # fallback: look for same-named file under hint_root
    cand = list(hint_root.rglob(Path(path_str).name))
    return cand[0] if cand else None

def main() -> None:
    pa = argparse.ArgumentParser(
        description="Gather mapping pairs into a flat folder for inspection"
    )
    pa.add_argument("map_json")
    pa.add_argument("extracted_dir")
    pa.add_argument("original_root")
    pa.add_argument("out_dir")
    args = pa.parse_args()

    mapping: dict[str, str] = json.load(open(args.map_json, encoding="utf-8"))
    extracted_root, original_root, dst_root = map(Path, 
        (args.extracted_dir, args.original_root, args.out_dir))

    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    seen_extracted: set[Path] = set()
    copied = 0
    for idx, (orig_str, ex_str) in enumerate(mapping.items(), start=1):
        orig_path = resolve(orig_str, original_root)
        ex_path   = resolve(ex_str,   extracted_root)

        if not orig_path or not ex_path:
            print(f"[WARN] missing file — orig:{orig_str}, ex:{ex_str}")
            continue
        if ex_path in seen_extracted:
            print(f"[WARN] extracted image reused → {ex_path}")
        seen_extracted.add(ex_path)

        prefix = f"{idx:04d}_{orig_path.stem}"
        ori_name = f"{prefix}_원본{orig_path.suffix}"
        ex_name  = f"{prefix}_추출{ex_path.suffix}"
        shutil.copy(orig_path, dst_root / ori_name)
        shutil.copy(ex_path,  dst_root / ex_name)
        copied += 1
        print(f"[COPY] {ori_name} , {ex_name}")

    print(f"[DONE] Copied {copied}/{len(mapping)} pairs to '{dst_root}'")

if __name__ == "__main__":
    main()
