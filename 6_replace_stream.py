# 6_replace_stream.py
"""
Replace *BinData* streams in an HWP (OLE compound) file with external images
according to a JSON mapping.

Key features
------------
* Keeps the original image format (raw JPEG/PNG/BMP) & resolution where possible.
* Re-compresses image payload with raw DEFLATE (wbits=-MAX_WBITS).
* JPEG: iteratively lowers *quality* until compressed payload ≤ original size.
* PNG: saves with `optimize=True`.
* If the new compressed stream is **smaller**, pads with `0x00`.
  If it is **larger**, tries safe trimming or progressively down-scales until
  the compressed payload fits and is decompressible.

Usage
-----
python 6_replace_stream.py \
  --map-json mapping_result.json \
  --src-hwp  "target_data/기계설비 성능점검 결과보고서(종합 1).hwp" \
  --dst-hwp  "target_data/기계설비 성능점검 결과보고서(종합 1)_수정본.hwp"

`mapping_result.json` must be a JSON object where **either** the key or the
value contains the BinData image path. The script automatically detects
whether the mapping direction is `BinData → original` (legacy) or
`original → BinData` (current).

All log messages and comments are in English for portability.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zlib
from io import BytesIO
from typing import Dict, Tuple

import olefile
from PIL import Image

# Helper functions

def parse_src_path(raw: str) -> str:
    """Best-effort fallback when `raw` is a compact, underscore-separated path.

    Example:
        'target_data_자동…_20_jpg.png' -> 'target_data/자동…/20.jpg'
    """
    try:
        raw = raw.strip()
        p0, p1, rest = raw.split('_', 2)
        base_folder = f"{p0}_{p1}"
        album, rest2 = rest.split('_', 1)
        subfolder, file_ext = rest2.split('_', 1)
        file_base, _ = file_ext.split('.', 1)
        num, ext = file_base.split('_', 1)
        return os.path.join(base_folder, album, subfolder, f"{num}.{ext}")
    except Exception:
        print(f"[WARN] parse_src_path: unexpected format, returning the raw string: {raw}")
        return raw


def safe_decompressable(data: bytes) -> bool:
    """Return True if `data` can be inflated by zlib without error."""
    try:
        zlib.decompress(data, -zlib.MAX_WBITS)
        return True
    except zlib.error:
        return False


def compress_image(img: Image.Image, fmt: str, orig_size: int) -> Tuple[bytes, bytes]:
    """Re-compress `img` with the given `fmt` trying not to exceed `orig_size`.

    Returns:
        raw_bytes: The re-encoded raw image data.
        comp_bytes: The raw-deflate compressed payload to store in the HWP stream.
    """
    fmt_upper = fmt.upper()
    if fmt_upper in {"JPEG", "JPG"}:
        last_raw = last_comp = None
        for q in range(95, 9, -5):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=q)
            data = buf.getvalue()

            comp = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
            comp_data = comp.compress(data) + comp.flush()

            print(f"[DEBUG] JPEG quality={q}, compressed={len(comp_data)} bytes")

            if len(comp_data) <= orig_size:
                print(f"[INFO] Selected JPEG quality={q} (fits original stream)")
                return data, comp_data

            last_raw, last_comp = data, comp_data

        print("[WARN] All quality levels exceeded the original size; using the last candidate")
        return last_raw, last_comp  # type: ignore[return-value]

    if fmt_upper == "PNG":
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()

        comp = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
        comp_data = comp.compress(data) + comp.flush()

        print(f"[DEBUG] PNG(optimize) compressed={len(comp_data)} bytes")
        return data, comp_data

    # Fallback for BMP or other formats
    buf = BytesIO()
    img.save(buf, format=fmt_upper)
    data = buf.getvalue()

    comp = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    comp_data = comp.compress(data) + comp.flush()

    print(f"[DEBUG] {fmt_upper} compressed={len(comp_data)} bytes")
    return data, comp_data

# Core replacement routine

def replace_streams(src: str, dst: str, mapping: Dict[str, Tuple[str, str]]) -> None:
    total = replaced = decompress_fail = trim_fail = 0

    shutil.copy2(src, dst)
    print(f"[INFO] Copied source HWP to '{dst}'")

    ole = olefile.OleFileIO(dst, write_mode=True)
    all_streams = ole.listdir(streams=True)
    print("[DEBUG] Listing OLE streams inside the document:")
    for stream in all_streams:
        print("  -", stream)

    for leaf, (orig_img_path, ext) in mapping.items():
        total += 1
        print(f"\n[STEP] Processing stream leaf='{leaf}', image='{orig_img_path}'")

        target = next((s for s in all_streams if s[0] == "BinData" and s[1].upper() == leaf.upper()), None)
        if not target:
            print(f"[WARN] Stream not found: {leaf}")
            continue

        print(f"[INFO] Stream located: {target}")

        orig_comp = ole.openstream(target).read()
        orig_size = len(orig_comp)
        print(f"[DEBUG] Original compressed size: {orig_size} bytes")

        # Determine raw image data
        if ext.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            print("[INFO] Treating as RAW image stream (no decompression needed)")
            raw = orig_comp
        else:
            try:
                raw = zlib.decompress(orig_comp, -zlib.MAX_WBITS)
                print(f"[DEBUG] Raw DEFLATE decompressed: {len(raw)} bytes")
            except zlib.error:
                try:
                    raw = zlib.decompress(orig_comp)
                    print(f"[DEBUG] zlib wrapper decompressed: {len(raw)} bytes")
                except Exception as e:
                    print(f"[ERROR] Decompression failure: {e}")
                    decompress_fail += 1
                    continue

        if not os.path.exists(orig_img_path):
            print(f"[ERROR] Image file not found: {orig_img_path}")
            continue

        try:
            mapping_img = Image.open(orig_img_path)
        except Exception as e:
            print(f"[ERROR] Failed to open image: {e}")
            continue

        try:
            orig_img = Image.open(BytesIO(raw))
        except Exception as e:
            print(f"[ERROR] Failed to decode original image: {e}")
            continue

        orig_w, orig_h = orig_img.size
        print(f"[DEBUG] Original resolution: {orig_w}×{orig_h}")

        if mapping_img.size != (orig_w, orig_h):
            print(f"[DEBUG] Resizing replacement {mapping_img.size} -> {(orig_w, orig_h)}")
            mapping_img = mapping_img.resize((orig_w, orig_h))

        fmt = orig_img.format or ext.lstrip('.').upper()
        new_raw, new_comp = compress_image(mapping_img.convert(orig_img.mode), fmt, orig_size)

        # Padding if compressed data is smaller
        if len(new_comp) < orig_size:
            pad = orig_size - len(new_comp)
            new_comp += b"\x00" * pad
            print(f"[DEBUG] Applied padding: {pad} bytes, final={len(new_comp)} bytes")

        # Trimming if compressed data is larger
        if len(new_comp) > orig_size:
            trimmed = new_comp[:orig_size]
            print(f"[WARN] Oversized stream ({len(new_comp)}->{orig_size}), attempting safe trim…")
            if safe_decompressable(trimmed):
                new_comp = trimmed
                print(f"[INFO] Trim valid, final size={len(new_comp)} bytes")
            else:
                print("[WARN] Trim invalid; attempting down-scaling…")
                success = False
                for scale in (0.9, 0.8, 0.7, 0.6, 0.5):
                    w2 = int(orig_w * scale)
                    h2 = int(orig_h * scale)
                    r2 = mapping_img.convert(orig_img.mode).resize((w2, h2))

                    _, comp2 = compress_image(r2, fmt, orig_size)

                    if len(comp2) <= orig_size and safe_decompressable(comp2):
                        if len(comp2) < orig_size:
                            comp2 += b"\x00" * (orig_size - len(comp2))
                        new_comp = comp2
                        print(f"[INFO] Down-scale to {int(scale*100)}% solved size issue ({len(new_comp)} bytes)")
                        success = True
                        break
                if not success:
                    print("[ERROR] Down-scaling also failed; skipping this stream.")
                    trim_fail += 1
                    continue

        ole.write_stream(target, new_comp)
        replaced += 1
        print(f"[INFO] Stream {target} overwritten (size {len(new_comp)} bytes)")

    ole.close()
    print(f"\n[SUMMARY] total={total}, replaced={replaced}, decompress_fail={decompress_fail}, trim_fail={trim_fail}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace BinData streams in an HWP file using an image mapping.")
    parser.add_argument("--map-json", required=True, help="Path to mapping_result.json")
    parser.add_argument("--src-hwp", required=True, help="Source .hwp file to patch")
    parser.add_argument("--dst-hwp", required=True, help="Destination .hwp to write")
    args = parser.parse_args()

    raw_map = json.load(open(args.map_json, encoding="utf-8"))
    original_map: Dict[str, Tuple[str, str]] = {}
    for k, v in raw_map.items():
        if "BinData" in k:
            bin_path, img_path = k, v
        elif "BinData" in v:
            bin_path, img_path = v, k
        else:
            bin_path, img_path = k, v

        fname = os.path.basename(bin_path)
        body = fname.split("_BinData_", 1)[1] if "_BinData_" in fname else fname
        name, ext = os.path.splitext(body)
        leaf = name

        if "_" in leaf:
            b, e = leaf.rsplit("_", 1)
            leaf = f"{b}.{e}"

        img_path_resolved = img_path if os.path.exists(img_path) else parse_src_path(img_path)
        original_map[leaf] = (img_path_resolved, ext.lower())

    replace_streams(args.src_hwp, args.dst_hwp, original_map)
