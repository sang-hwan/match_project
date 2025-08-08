# 5_check_map_result.py
"""
Collect mapped pairs into one folder for quick visual check.

Usage (RAW):
  python 5_check_map_result.py mapping_result.json images_output target_data out_pairs

Usage (processed low/gray):
  python 5_check_map_result.py mapping_result.json images_output target_data out_pairs_low_gray ^
    --mode processed --processed-root processed ^
    --track low --channel gray --preprocess-mapping preprocess_mapping.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────── Path & utils ───────────────
def norm(p: str) -> str:
    """Normalize path for comparisons (slash + lowercase)."""
    return str(Path(p)).replace("\\", "/").lower()

def is_extracted_identity(p: str) -> bool:
    return "bindata" in norm(p)

def resolve_raw(path_str: str, hint_root: Path) -> Optional[Path]:
    """
    RAW 해상: 1) 그대로 존재하면 사용  2) 없으면 basename으로 hint_root 하위 탐색
    """
    p = Path(path_str)
    if p.is_file():
        return p
    cand = list(hint_root.rglob(Path(path_str).name))
    return cand[0] if cand else None

def link_or_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "hardlink":
            os.link(src, dst)
            return
        if mode == "symlink":
            dst.symlink_to(src.resolve())
            return
    except Exception:
        pass  # fallback to copy
    shutil.copy2(src, dst)

# ─────────────── Preprocessed resolver ───────────────
def norm_channel(ch: str) -> str:
    t = str(ch).strip().lower()
    if t in {"gray", "grey", "그레이"} or t.endswith("_g"):
        return "gray"
    if t in {"color", "colour", "컬러"} or t.endswith("_c"):
        return "color"
    return t  # 그대로 둠

class PreprocIndex:
    """
    (origin, track, channel) -> preprocessed filename (e.g., 000123.png)
    """
    def __init__(self, mp: Dict[str, Dict]):
        self.idx: Dict[Tuple[str, str, str], str] = {}
        for out_name, meta in mp.items():
            o = norm(meta.get("원본_전체_경로", ""))
            tr = str(meta.get("트랙", "")).lower()
            ch = norm_channel(str(meta.get("채널", "")))
            if o and tr and ch:
                self.idx[(o, tr, ch)] = out_name  # last write wins

    def get(self, origin_path: str, track: str, channel: str) -> Optional[str]:
        return self.idx.get((norm(origin_path), track.lower(), channel.lower()))

def resolve_preprocessed(
    origin_identity: str,
    processed_root: Path,
    track: str,
    channel: str,
    pp_index: PreprocIndex,
) -> Optional[Path]:
    """
    processed/{extracted|original}/{track}/{channel}/<filename>
    """
    out_name = pp_index.get(origin_identity, track, channel)
    if not out_name:
        return None
    category = "extracted" if is_extracted_identity(origin_identity) else "original"
    p = processed_root / category / track / channel / out_name
    return p if p.is_file() else None

# ─────────────── Mapping normalize ───────────────
def unify_pairs(raw_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    (original_identity, extracted_identity) 리스트로 정규화.
    한쪽만 BinData면 방향 확정. 그 외엔 값 쪽이 BinData면 유지, 아니면 교환.
    """
    pairs: List[Tuple[str, str]] = []
    for k, v in raw_map.items():
        k_ex, v_ex = is_extracted_identity(k), is_extracted_identity(v)
        if v_ex and not k_ex:
            pairs.append((k, v))
        elif k_ex and not v_ex:
            pairs.append((v, k))
        else:
            pairs.append((k, v) if v_ex else (v, k))
    return pairs

# ─────────────── CLI ───────────────
def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser(
        description="Collect mapped pairs into a flat folder for inspection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # positional (backward-compatible)
    pa.add_argument("map_json", help="mapping_result.json (orig↔extracted)")
    pa.add_argument("extracted_dir", help="ROOT of RAW extracted images (e.g., images_output)")
    pa.add_argument("original_root", help="ROOT of RAW original images (folder tree)")
    pa.add_argument("out_dir", help="Destination folder")

    # options
    pa.add_argument("--mode", choices=["raw", "processed"], default="raw",
                    help="Copy RAW or PREPROCESSED files")
    pa.add_argument("--processed-root", default="processed",
                    help="Root containing processed/extracted|original")
    pa.add_argument("--preprocess-mapping", default="preprocess_mapping.json",
                    help="Mapping to resolve preprocessed filenames")
    pa.add_argument("--track", choices=["low", "high"], default="low")
    pa.add_argument("--channel", choices=["gray", "color"], default="gray")
    pa.add_argument("--link", choices=["copy", "hardlink", "symlink"], default="copy")
    pa.add_argument("--export-csv", default=None, help="Write manifest CSV (dst, src)")
    return pa.parse_args()

# ─────────────── Main ───────────────
def main() -> None:
    args = parse_args()

    mapping_raw: Dict[str, str] = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
    pairs = unify_pairs(mapping_raw)

    extracted_root = Path(args.extracted_dir)
    original_root  = Path(args.original_root)
    dst_root       = Path(args.out_dir)

    shutil.rmtree(dst_root, ignore_errors=True)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Preprocessed index (optional)
    pp_index: Optional[PreprocIndex] = None
    processed_root = Path(args.processed_root)
    if args.mode == "processed":
        try:
            mp = json.loads(Path(args.preprocess_mapping).read_text(encoding="utf-8"))
            pp_index = PreprocIndex(mp)
        except Exception as e:
            print(f"[WARN] failed to load preprocess mapping: {e}")

    seen_extracted: set[Path] = set()
    copied = 0
    rows_for_csv: List[Tuple[str, str]] = []

    for idx, (orig_id, ex_id) in enumerate(pairs, start=1):
        # Resolve src paths
        if args.mode == "processed" and pp_index is not None:
            orig_path = resolve_preprocessed(orig_id, processed_root, args.track, args.channel, pp_index) \
                        or resolve_raw(orig_id, original_root)
            ex_path   = resolve_preprocessed(ex_id,   processed_root, args.track, args.channel, pp_index) \
                        or resolve_raw(ex_id, extracted_root)
        else:
            orig_path = resolve_raw(orig_id, original_root)
            ex_path   = resolve_raw(ex_id, extracted_root)

        if not orig_path or not ex_path:
            print(f"[WARN] missing file — orig:{orig_id}, ex:{ex_id}")
            continue

        if ex_path in seen_extracted:
            print(f"[WARN] extracted image reused → {ex_path}")
        seen_extracted.add(ex_path)

        prefix   = f"{idx:04d}_{orig_path.stem}"
        dst_ori  = dst_root / f"{prefix}_원본{orig_path.suffix}"
        dst_ex   = dst_root / f"{prefix}_추출{ex_path.suffix}"

        try:
            link_or_copy(orig_path, dst_ori, args.link)
            link_or_copy(ex_path,  dst_ex,  args.link)
            rows_for_csv.append((str(dst_ori), str(orig_path)))
            rows_for_csv.append((str(dst_ex),  str(ex_path)))
            copied += 1
            print(f"[COPY] {dst_ori.name} , {dst_ex.name}")
        except Exception as e:
            print(f"[ERROR] copy failed: {e}")

    if args.export_csv:
        with open(args.export_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["dst_path", "src_path"])
            w.writerows(rows_for_csv)
        print(f"[SAVE] manifest: {args.export_csv}")

    print(f"[DONE] Copied {copied}/{len(pairs)} pairs → '{dst_root}'")

if __name__ == "__main__":
    main()
