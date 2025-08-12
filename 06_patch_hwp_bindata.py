"""
06_patch_hwp_bindata.py

Patch (replace) *BinData/* streams inside a Hangul HWP (OLE compound) file
using external images determined by a mapping.

Supported mappings
------------------
1) New pipeline mapping (from 04_match.py):
   - mapping_result.json with:
       {
         "mapping": [
           {"reference": "<ref_rel>", "extracted": "<ex_rel>", ...}, ...
         ],
         "stats": {"references_total": ... , "assigned": ...},
         ...
       }
   - Requires preprocess_mapping.json (for roots) and extracted_manifest.json
     (from 01_extract_hwp.py) to recover which extracted image corresponds to
     which "BinData/BinXXXX" stream.

2) Legacy mapping (dict-like):
   - JSON mapping where either key or value contains a "BinData/..." path
     and the opposite side is an image path.

Key behaviors
-------------
- Keeps the *original* resolution. Replacement is resized to the original stream's
  decoded width/height.
- Uses raw DEFLATE (wbits=-MAX_WBITS) when the original stream is DEFLATE-compressed.
  If the original stream is RAW (e.g., plain JPEG/PNG/BMP bytes), it writes RAW.
- JPEG: iteratively reduces quality until the target (compressed/raw) size fits.
- PNG: uses optimize=True; if size still exceeds, progressively down-scales.
- Deflate mode: if new payload is smaller, pads with 0x00; if larger, tries
  safe trim (only if still decompressible), then down-scaling as a fallback.
- RAW mode: never trims; only adjusts (quality/scale). Optional trailing pad
  is allowed (decoders typically ignore trailing NULs).

Usage
-----
python 06_patch_hwp_bindata.py \
  --mapping-json mapping_result.json \
  --mapping-meta preprocess_mapping.json \
  --src-hwp "input.hwp" \
  --dst-hwp "output_patched.hwp" \
  [--extracted-manifest "<extracted_root>/extracted_manifest.json"] \
  [--jpeg-q-start 95 --jpeg-q-min 10 --jpeg-q-step 5] \
  [--downscale "0.9,0.8,0.7,0.6,0.5"] \
  [--limit 0]

Requirements
------------
- olefile (pip install olefile)
- Pillow  (pip install pillow)

Notes
-----
- This tool only patches classic HWP (OLE). HWPX(ZIP) patching is out of scope.
- We attempt to be robust but cannot guarantee 100% compatibility across all
  vendor variants. Always keep a backup of the original .hwp.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import olefile  # type: ignore
from PIL import Image  # type: ignore

# --------------------------
# Image sniffers / helpers
# --------------------------

MAGIC_SIGNATURES = {
    "jpg": (b"\xFF\xD8\xFF",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "gif": (b"GIF87a", b"GIF89a"),
    "bmp": (b"BM",),
    "tif": (b"II*\x00", b"MM\x00*"),
    "webp": (b"RIFF",),
    "ico": (b"\x00\x00\x01\x00",),
}

def sniff_image_type(data: bytes) -> Tuple[Optional[str], str]:
    """Return (ext, mime) if recognizable raster; else (None, 'unknown')."""
    head = data[:16]
    # WEBP special-case: "RIFF....WEBP"
    if head.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    for ext, sigs in MAGIC_SIGNATURES.items():
        for sig in sigs:
            if head.startswith(sig):
                if ext == "jpg":
                    return "jpg", "image/jpeg"
                if ext == "png":
                    return "png", "image/png"
                if ext == "gif":
                    return "gif", "image/gif"
                if ext == "bmp":
                    return "bmp", "image/bmp"
                if ext == "ico":
                    return "ico", "image/x-icon"
                if ext == "tif":
                    return "tif", "image/tiff"
                if ext == "webp":
                    return "webp", "image/webp"
    return None, "unknown"

def safe_decompressable(data: bytes) -> bool:
    """True if raw DEFLATE inflate succeeds."""
    try:
        zlib.decompress(data, -zlib.MAX_WBITS)
        return True
    except zlib.error:
        return False

def try_inflate(orig_comp: bytes) -> Tuple[bool, Optional[bytes]]:
    """Attempt raw-then-zlib inflate; return (is_deflate, raw_bytes or None)."""
    try:
        raw = zlib.decompress(orig_comp, -zlib.MAX_WBITS)
        return True, raw
    except zlib.error:
        try:
            raw = zlib.decompress(orig_comp)
            return True, raw
        except zlib.error:
            return False, None

# --------------------------
# Compression routines
# --------------------------

def compress_image(img: Image.Image, fmt_upper: str,
                   target_size: int,
                   jpeg_q_start: int, jpeg_q_min: int, jpeg_q_step: int) -> Tuple[bytes, bytes]:
    """
    Encode PIL image to given format; return (raw_bytes, deflate_bytes).
    For JPEG, lower quality progressively (q_start -> q_min step -q_step) to try
    to meet target_size (compared by deflate-bytes length in deflate mode,
    or by raw-bytes length in raw mode handled by caller).
    """
    fmt_upper = fmt_upper.upper()
    if fmt_upper in {"JPEG", "JPG"}:
        last_raw = last_def = None
        for q in range(int(jpeg_q_start), int(jpeg_q_min) - 1, -int(jpeg_q_step)):
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=max(1, q))
            data = buf.getvalue()
            cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
            deflated = cobj.compress(data) + cobj.flush()
            last_raw, last_def = data, deflated
            # Early return is left to caller (needs to know store mode)
        return last_raw or b"", last_def or b""

    if fmt_upper == "PNG":
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
        deflated = cobj.compress(data) + cobj.flush()
        return data, deflated

    # Fallback formats (BMP, WEBP, etc.)
    buf = BytesIO()
    img.save(buf, format=fmt_upper)
    data = buf.getvalue()
    cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    deflated = cobj.compress(data) + cobj.flush()
    return data, deflated

# --------------------------
# Mapping loaders
# --------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_legacy_mapping(obj: dict) -> Dict[str, str]:
    """
    Legacy mapping dict: either key or value contains 'BinData/...'.
    Returns {leaf: image_abs_path}
    """
    out: Dict[str, str] = {}
    for k, v in obj.items():
        k = str(k); v = str(v)
        if "bindata" in k.lower():
            leaf = extract_leaf_from_bindata(k)
            if leaf:
                out[leaf] = v
        elif "bindata" in v.lower():
            leaf = extract_leaf_from_bindata(v)
            if leaf:
                out[leaf] = k
    return out

def extract_leaf_from_bindata(s: str) -> Optional[str]:
    """
    Extract 'BinXXXX' (optionally with extension) from a path-like string.
    """
    m = re.search(r"[\\/]?BinData[\\/]([^\\/]+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    leaf = m.group(1)
    # Normalize: drop extension if like "Bin0001.jpg" -> "Bin0001"
    base = os.path.splitext(leaf)[0]
    return base

def build_leaf_map_from_manifest(extracted_manifest: Path, extracted_root: Path) -> Dict[str, str]:
    """
    From extracted_manifest.json items, map 'extracted_rel' -> 'BinXXXX' leaf.
    Only entries with source prefix "HWP:BinData/..." are considered.
    """
    if not extracted_manifest.exists():
        raise SystemExit(f"[ERR] extracted_manifest.json not found: {extracted_manifest}")
    man = load_json(extracted_manifest)
    items = man.get("items", [])
    mapping: Dict[str, str] = {}
    for it in items:
        src = str(it.get("source",""))
        out_path = Path(it.get("out_path",""))
        try:
            ex_rel = str(out_path.relative_to(extracted_root).as_posix())
        except Exception:
            # fallback: just use basename
            ex_rel = out_path.name
        if src.startswith("HWP:"):
            # Expect "HWP:BinData/Bin0001"
            m = re.search(r"HWP:BinData[\\/]([^\\/]+)", src, flags=re.IGNORECASE)
            if not m:
                continue
            leaf = os.path.splitext(m.group(1))[0]
            mapping[ex_rel] = leaf
    return mapping

def build_leaf_to_image_newformat(
    mapping_result: Path,
    preprocess_meta: Path,
    extracted_manifest: Optional[Path]
) -> Dict[str, str]:
    """
    For new format mapping_result.json (04_match.py), build {leaf -> reference_abs_path}.
    """
    mr = load_json(mapping_result)
    if "mapping" not in mr:
        return {}

    meta = load_json(preprocess_meta)
    ref_root = Path(meta.get("roots",{}).get("reference_in",""))
    ex_root  = Path(meta.get("roots",{}).get("extracted_in",""))
    if not ref_root.exists():
        raise SystemExit(f"[ERR] reference_in root not found: {ref_root}")
    if not ex_root.exists():
        raise SystemExit(f"[ERR] extracted_in root not found: {ex_root}")

    # Locate extracted_manifest.json
    if extracted_manifest is None:
        extracted_manifest = ex_root / "extracted_manifest.json"
    if not extracted_manifest.exists():
        raise SystemExit(f"[ERR] extracted_manifest.json not found at: {extracted_manifest}")

    # ex_rel -> BinXXXX
    exrel_to_leaf = build_leaf_map_from_manifest(extracted_manifest, ex_root)

    # Leaf -> reference_abs
    leaf_to_img: Dict[str, str] = {}
    for m in mr.get("mapping", []):
        ex_rel = str(m.get("extracted","")).strip()
        ref_rel = str(m.get("reference","")).strip()
        if not ex_rel or not ref_rel:
            continue
        leaf = exrel_to_leaf.get(ex_rel)
        if not leaf:
            print(f"[WARN] Cannot resolve leaf for extracted '{ex_rel}' (skip)")
            continue
        ref_abs = (ref_root / ref_rel)
        if not ref_abs.exists():
            print(f"[WARN] Reference image not found: {ref_abs} (skip leaf={leaf})")
            continue
        # If duplicates, first come first served
        leaf_to_img.setdefault(leaf, str(ref_abs))
    return leaf_to_img

def build_leaf_to_image_any(
    mapping_json: Path,
    preprocess_meta: Optional[Path],
    extracted_manifest: Optional[Path]
) -> Dict[str, str]:
    """
    Try new-format first; if fails, fall back to legacy mapping.
    """
    obj = load_json(mapping_json)
    if isinstance(obj, dict) and "mapping" in obj:
        if preprocess_meta is None:
            raise SystemExit("[ERR] --mapping-meta is required for new-format mapping_result.json")
        return build_leaf_to_image_newformat(mapping_json, preprocess_meta, extracted_manifest)

    if isinstance(obj, dict):
        # Legacy style
        out = parse_legacy_mapping(obj)
        if not out:
            print("[WARN] Legacy mapping parsed empty. Check your JSON.")
        return out

    raise SystemExit("[ERR] Unrecognized mapping JSON format.")

# --------------------------
# OLE patching
# --------------------------

def find_bindata_stream(ole: olefile.OleFileIO, leaf: str) -> Optional[Tuple[str, ...]]:
    """
    Find a stream like ('BinData','Bin0001') for the given leaf (case-insensitive).
    Returns the entry tuple or None.
    """
    leaf_low = leaf.lower()
    for path in ole.listdir(streams=True):
        # Accept deeper levels but require 'BinData' as first component
        if not path:
            continue
        if path[0].lower() != "bindata":
            continue
        last = path[-1].lower()
        # Normalize last: 'Bin0001.jpg' -> 'Bin0001'
        base = os.path.splitext(last)[0]
        if base == leaf_low:
            return tuple(path)
    return None

def choose_fmt_from_raw(raw: bytes, fallback: Optional[str] = None) -> str:
    ext, _ = sniff_image_type(raw)
    if ext == "jpg":
        return "JPEG"
    if ext == "png":
        return "PNG"
    if ext == "bmp":
        return "BMP"
    if ext == "webp":
        return "WEBP"
    if fallback:
        return fallback.upper()
    return "JPEG"

def open_image_safe(path: Path) -> Optional[Image.Image]:
    try:
        im = Image.open(path)
        # Load immediately to avoid lazy issues after file close
        im.load()
        return im
    except Exception as e:
        print(f"[ERROR] Failed to open image: {path} ({e})")
        return None

def patch_streams(
    src_hwp: Path,
    dst_hwp: Path,
    leaf_to_img: Dict[str, str],
    jpeg_q_start: int,
    jpeg_q_min: int,
    jpeg_q_step: int,
    downscale_seq: List[float],
    limit: int = 0,
) -> None:
    # Prepare destination
    shutil.copy2(src_hwp, dst_hwp)
    print(f"[INFO] Copied '{src_hwp}' -> '{dst_hwp}'")

    ole = olefile.OleFileIO(str(dst_hwp), write_mode=True)  # type: ignore
    all_streams = ole.listdir(streams=True)
    print(f"[INFO] OLE streams found: {len(all_streams)}")

    total = replaced = fail_decomp = fail_adjust = 0

    done = 0
    for leaf, img_path in leaf_to_img.items():
        if limit and done >= limit:
            break
        done += 1

        stream = find_bindata_stream(ole, leaf)
        if not stream:
            print(f"[WARN] BinData stream not found for leaf '{leaf}'")
            continue

        try:
            orig_comp = ole.openstream(stream).read()
        except Exception as e:
            print(f"[ERROR] Cannot read stream {stream}: {e}")
            continue

        total += 1
        orig_size = len(orig_comp)
        is_deflate, raw = try_inflate(orig_comp)
        if is_deflate and raw is None:
            # Shouldn't happen, but guard
            is_deflate = False
            raw = orig_comp

        if not is_deflate:
            # RAW stream (likely plain JPEG/PNG bytes)
            raw = orig_comp

        # Determine original decoded info
        try:
            orig_img = Image.open(BytesIO(raw))
            orig_img.load()
            orig_w, orig_h = orig_img.size
            orig_mode = orig_img.mode
        except Exception as e:
            print(f"[ERROR] Failed to decode original stream {stream}: {e}")
            fail_decomp += 1
            continue

        # Replacement image
        rep_abs = Path(img_path)
        rep_img = open_image_safe(rep_abs)
        if rep_img is None:
            continue

        if rep_img.size != (orig_w, orig_h):
            rep_img = rep_img.resize((orig_w, orig_h))

        # Decide format by original raw sniff (fallback by replacement ext)
        fallback_fmt = Path(img_path).suffix.lstrip(".") or "JPEG"
        fmt = choose_fmt_from_raw(raw, fallback=fallback_fmt)

        # Convert replacement to original mode (when possible)
        try:
            rep_conv = rep_img.convert(orig_mode)
        except Exception:
            rep_conv = rep_img.convert("RGB")

        # Encode once at max quality, then caller loop may re-encode at lower q in JPEG path
        new_raw, new_def = compress_image(rep_conv, fmt, orig_size, jpeg_q_start, jpeg_q_min, jpeg_q_step)

        def choose_bytes_for_mode(_raw: bytes, _def: bytes) -> bytes:
            return _def if is_deflate else _raw

        payload = choose_bytes_for_mode(new_raw, new_def)

        def fits(b: bytes) -> bool:
            return len(b) <= orig_size

        # If too large, attempt quality/scale search
        if not fits(payload):
            # JPEG path: compress_image() already produced a list of trials; we need to
            # walk quality schedule manually to find a fitting candidate.
            if fmt.upper() in {"JPEG","JPG"}:
                success = False
                for q in range(jpeg_q_start, jpeg_q_min - 1, -jpeg_q_step):
                    buf = BytesIO()
                    rep_conv.save(buf, format="JPEG", quality=max(1, q))
                    data = buf.getvalue()
                    cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
                    deflated = cobj.compress(data) + cobj.flush()
                    cand = choose_bytes_for_mode(data, deflated)
                    if len(cand) <= orig_size:
                        payload = cand
                        success = True
                        print(f"[INFO] JPEG quality={q} fits for {leaf} (size={len(payload)} <= {orig_size})")
                        break
                if not success:
                    # try downscales
                    for s in downscale_seq:
                        w2 = max(1, int(orig_w * s))
                        h2 = max(1, int(orig_h * s))
                        rep_s = rep_conv.resize((w2, h2))
                        buf = BytesIO()
                        rep_s.save(buf, format="JPEG", quality=max(1, jpeg_q_min))
                        data = buf.getvalue()
                        cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
                        deflated = cobj.compress(data) + cobj.flush()
                        cand = choose_bytes_for_mode(data, deflated)
                        if len(cand) <= orig_size:
                            payload = cand
                            success = True
                            print(f"[INFO] Down-scale {int(s*100)}% solved size for {leaf}")
                            break
                    if not success:
                        print(f"[ERROR] Cannot fit JPEG for {leaf} within original size")
                        fail_adjust += 1
                        continue
            else:
                # Non-JPEG (PNG/BMP/WEBP): try downscales
                success = False
                for s in downscale_seq:
                    w2 = max(1, int(orig_w * s))
                    h2 = max(1, int(orig_h * s))
                    rep_s = rep_conv.resize((w2, h2))
                    buf = BytesIO()
                    rep_s.save(buf, format=fmt)
                    data = buf.getvalue()
                    cobj = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
                    deflated = cobj.compress(data) + cobj.flush()
                    cand = choose_bytes_for_mode(data, deflated)
                    if len(cand) <= orig_size:
                        payload = cand
                        success = True
                        print(f"[INFO] Down-scale {int(s*100)}% solved size for {leaf}")
                        break
                if not success:
                    print(f"[ERROR] Cannot fit {fmt} for {leaf} within original size")
                    fail_adjust += 1
                    continue

        # If smaller, pad (mainly for deflate; RAW trailing NULs are usually tolerated)
        if len(payload) < orig_size:
            pad = orig_size - len(payload)
            payload += b"\x00" * pad

        # Deflate mode oversize: try safe trim (only meaningful in deflate)
        if is_deflate and len(payload) > orig_size:
            trimmed = payload[:orig_size]
            if safe_decompressable(trimmed):
                payload = trimmed
            else:
                print(f"[ERROR] Deflate payload still larger & not safely trimmable for {leaf}")
                fail_adjust += 1
                continue

        # Write stream
        try:
            ole.write_stream(stream, payload)  # type: ignore
            replaced += 1
            print(f"[OK] Patched {stream} (size={len(payload)})")
        except Exception as e:
            print(f"[ERROR] write_stream failed for {stream}: {e}")

    ole.close()
    print(f"\n[SUMMARY] total={total}, replaced={replaced}, fail_decompress={fail_decomp}, fail_adjust={fail_adjust}")

# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Patch HWP BinData streams using a mapping.")
    ap.add_argument("--mapping-json", required=True, help="mapping_result.json (new) or legacy dict mapping JSON")
    ap.add_argument("--mapping-meta", default="", help="preprocess_mapping.json (required for new-format mapping)")
    ap.add_argument("--extracted-manifest", default="", help="extracted_manifest.json (defaults to <extracted_in>/extracted_manifest.json)")

    ap.add_argument("--src-hwp", required=True, help="Source .hwp to patch (OLE)")
    ap.add_argument("--dst-hwp", required=True, help="Destination .hwp to write")

    ap.add_argument("--jpeg-q-start", type=int, default=95)
    ap.add_argument("--jpeg-q-min", type=int, default=10)
    ap.add_argument("--jpeg-q-step", type=int, default=5)
    ap.add_argument("--downscale", default="0.9,0.8,0.7,0.6,0.5",
                    help="Comma-separated scale factors for fallback down-scaling")
    ap.add_argument("--limit", type=int, default=0, help="Max number of streams to patch (0=all)")

    return ap.parse_args()

def main() -> int:
    args = parse_args()

    mapping_json = Path(args.mapping_json)
    if not mapping_json.exists():
        print(f"[ERR] mapping JSON not found: {mapping_json}")
        return 2

    preprocess_meta = Path(args.mapping_meta) if args.mapping_meta else None
    extracted_manifest = Path(args.extracted_manifest) if args.extracted_manifest else None

    leaf_to_img = build_leaf_to_image_any(mapping_json, preprocess_meta, extracted_manifest)
    if not leaf_to_img:
        print("[ERR] No leaf->image mapping produced. Abort.")
        return 3

    src_hwp = Path(args.src_hwp)
    dst_hwp = Path(args.dst_hwp)
    if not src_hwp.exists():
        print(f"[ERR] Source HWP not found: {src_hwp}")
        return 2
    if dst_hwp.exists():
        print(f"[WARN] Destination already exists; will overwrite: {dst_hwp}")

    downscale_seq = []
    for tok in args.downscale.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            s = float(tok)
            if 0.05 <= s < 1.0:
                downscale_seq.append(s)
        except Exception:
            pass
    if not downscale_seq:
        downscale_seq = [0.9, 0.8, 0.7, 0.6, 0.5]

    patch_streams(
        src_hwp=src_hwp,
        dst_hwp=dst_hwp,
        leaf_to_img=leaf_to_img,
        jpeg_q_start=args.jpeg_q_start,
        jpeg_q_min=args.jpeg_q_min,
        jpeg_q_step=args.jpeg_q_step,
        downscale_seq=downscale_seq,
        limit=args.limit,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
