"""
01_extract_hwp.py

Extract embedded raster images from Hangul Word Processor files:
- HWP (OLE Compound File, v5.*)
- HWPX (ZIP-based)

Design goals (CPU-only, zero/non-hard deps):
- No internet or external tools required.
- Optional use of "olefile" (for HWP) and "Pillow" (for size filtering); if absent,
  the script still runs with graceful degradation.
- Heuristically skips preview/thumbnail streams.
- Writes a manifest JSON for traceability.

Usage
-----
python 01_extract_hwp.py input.(hwp|hwpx) output_dir \
    [--sample N] [--min-bytes 0] [--include-preview] [--prefix extracted_] [--dry-run]

Notes
-----
- HWP (OLE) images usually live under "BinData" storages and can be zlib-deflated.
- HWPX (ZIP) embeds images as normal files (png/jpg/gif/bmp/webp/tiff etc).
- We "sniff" file signatures to select only real images and assign the right extension.

Author: (refactor proposal implementation)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Optional deps
try:
    import olefile  # type: ignore
except Exception:  # pragma: no cover
    olefile = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# --------------------------
# Utility: signature sniffers
# --------------------------

MAGIC_SIGNATURES = {
    "jpg": (b"\xFF\xD8\xFF",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "gif": (b"GIF87a", b"GIF89a"),
    "bmp": (b"BM",),
    "tif": (b"II*\x00", b"MM\x00*"),
    "webp": (b"RIFF",),
    "ico": (b"\x00\x00\x01\x00",),
    # vector/others (we keep but mark as non-preferred)
    "wmf": (b"\xD7\xCD\xC6\x9A",),
    "emf": (b"\x01\x00\x00\x00",),  # weak; EMF also has 'EMF' header inside
}

def sniff_image_type(data: bytes) -> Tuple[Optional[str], str]:
    """
    Return (preferred_ext, mime_hint) for known raster images.
    If unsure, returns (None, "unknown").
    """
    head = data[:16]
    # WEBP requires "RIFF" then "WEBP" at offset 8
    if head.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp", "image/webp"

    for ext, sigs in MAGIC_SIGNATURES.items():
        for sig in sigs:
            if head.startswith(sig):
                # tif vs others share
                if ext == "tif":
                    return "tif", "image/tiff"
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
                if ext in ("wmf", "emf"):
                    # Not a raster; we don't convert, but we can still export if requested
                    return ext, f"application/x-{ext}"
    return None, "unknown"


# --------------------------
# Data classes
# --------------------------

@dataclass
class ExtractedItem:
    source: str                 # "HWP:BinData/..." or "HWPX:zip/path"
    order: int                  # incremental index
    out_path: Path              # target file path
    ext: str                    # file extension (w/o dot)
    bytes: int                  # size saved
    mime: str                   # mime hint
    skipped_reason: Optional[str] = None


# --------------------------
# Helpers
# --------------------------

def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_preview_like(name: str) -> bool:
    """
    Heuristic to skip "preview/thumbnail" streams.
    """
    name_l = name.lower()
    patterns = ("preview", "thumbnail", "thumb", "prvimage", "prv", "_pv_", "_thumb")
    return any(k in name_l for k in patterns)

def maybe_decompress_zlib(data: bytes) -> bytes:
    """
    Try to decompress BinData payloads that are sometimes stored as raw deflate or zlib.
    """
    # Fast path: if looks like PNG/JPG/etc already, return as-is
    ext, _ = sniff_image_type(data)
    if ext:
        return data

    # Try zlib raw (-15), then normal
    for wbits in (-15, zlib.MAX_WBITS):
        try:
            out = zlib.decompress(data, wbits)
            # If decompressed looks like an image, accept it
            ext2, _ = sniff_image_type(out)
            if ext2:
                return out
            # If not, still return out as it's likely a container with image?
            # But we prefer to keep original if not an image.
        except Exception:
            continue

    return data  # give up


# --------------------------
# Core extractors
# --------------------------

def detect_format(path: Path) -> str:
    """
    Return "hwpx" (zip) or "hwp" (ole) or raise.
    """
    # First try zip
    try:
        with zipfile.ZipFile(path) as zf:
            # HWPX should contain some xmls under "Contents/"
            names = zf.namelist()
            if any(n.startswith("Contents/") for n in names):
                return "hwpx"
            # Some HWPX may have lowercase or other; fallback: any .xml at root?
            if any(n.lower().endswith(".xml") for n in names):
                return "hwpx"
    except zipfile.BadZipFile:
        pass

    # Then assume OLE (HWP v5.x)
    return "hwp"


def extract_from_hwpx(path: Path,
                      outdir: Path,
                      sample: Optional[int],
                      min_bytes: int,
                      include_preview: bool,
                      prefix: str,
                      dry_run: bool) -> List[ExtractedItem]:
    """
    Extract images from HWPX (zip). We scan all entries and keep only those
    whose content sniffs as a raster image.
    """
    items: List[ExtractedItem] = []
    order = 0
    with zipfile.ZipFile(path) as zf:
        # Gather candidate entries
        entries = []
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            # quick skip: not interested in xml, rels, fonts
            if name.lower().endswith((".xml", ".rels", ".ttf", ".otf")):
                continue
            # Usually images live under Contents/Resources/ or Contents/images/
            if not include_preview and is_preview_like(name):
                continue
            entries.append(name)

        # sampling
        if sample is not None and sample > 0 and len(entries) > sample:
            entries = sorted(entries)[:sample]

        ensure_outdir(outdir)

        for name in entries:
            try:
                data = zf.read(name)
            except Exception:
                continue

            if len(data) < max(16, min_bytes):
                # too small or meaningless
                continue

            # Some resources might be stored with arbitrary extension; sniff
            ext, mime = sniff_image_type(data)
            if not ext:
                # Not a raster image; skip
                continue

            order += 1
            out_path = outdir / f"{prefix}{order:04d}.{ext}"
            if not dry_run:
                with open(out_path, "wb") as f:
                    f.write(data)

            items.append(ExtractedItem(
                source=f"HWPX:{name}",
                order=order,
                out_path=out_path,
                ext=ext,
                bytes=len(data),
                mime=mime
            ))

    return items


def extract_from_hwp(path: Path,
                     outdir: Path,
                     sample: Optional[int],
                     min_bytes: int,
                     include_preview: bool,
                     prefix: str,
                     dry_run: bool) -> List[ExtractedItem]:
    """
    Extract images from HWP (OLE). Requires 'olefile' if available.
    We iterate 'BinData' storages and try to decompress payloads.
    """
    if olefile is None:
        raise RuntimeError(
            "olefile is required to parse .hwp (OLE) files. "
            "Install with: pip install olefile"
        )

    items: List[ExtractedItem] = []
    order = 0
    ensure_outdir(outdir)

    with olefile.OleFileIO(str(path)) as ole:
        # Collect candidates under BinData or anything that looks like image carriers
        direntries = ole.listdir(streams=True, storages=True)
        # Example entries: ['BinData', 'BinData', '...'] etc. We select streams inside 'BinData'
        # such as ['BinData', 'Bin0001']
        candidates: List[Tuple[str, ...]] = []
        for entry in direntries:
            # Skip storages; we need stream entries under BinData
            if len(entry) == 1:
                # top-level storages/streams; skip
                continue
            parent = entry[0].lower()
            name = "/".join(entry)
            if parent == "bindata":
                # Heuristic: skip previews
                if (not include_preview) and is_preview_like(name):
                    continue
                candidates.append(tuple(entry))

        # sampling
        if sample is not None and sample > 0 and len(candidates) > sample:
            candidates = sorted(candidates)[:sample]

        for entry in candidates:
            name = "/".join(entry)
            try:
                with ole.openstream(entry) as fp:
                    raw = fp.read()
            except Exception:
                continue

            if not raw or len(raw) < 8:
                continue

            # Try to decompress (many BinData payloads are deflated image bytes)
            payload = maybe_decompress_zlib(raw)

            if len(payload) < max(16, min_bytes):
                continue

            ext, mime = sniff_image_type(payload)
            if not ext:
                # Not a known raster image, skip (could be WMF/EMF/ole embedded)
                # If you want to keep EMF/WMF as-is, you can relax this condition.
                continue

            order += 1
            out_path = outdir / f"{prefix}{order:04d}.{ext}"
            if not dry_run:
                with open(out_path, "wb") as f:
                    f.write(payload)

            items.append(ExtractedItem(
                source=f"HWP:{name}",
                order=order,
                out_path=out_path,
                ext=ext,
                bytes=len(payload),
                mime=mime
            ))

    return items


# --------------------------
# Manifest & CLI
# --------------------------

def write_manifest(manifest_path: Path, items: List[ExtractedItem], src_file: Path) -> None:
    manifest = {
        "source_file": str(src_file),
        "total_extracted": len(items),
        "items": [
            {
                "source": it.source,
                "order": it.order,
                "out_path": str(it.out_path),
                "ext": it.ext,
                "bytes": it.bytes,
                "mime": it.mime,
            }
            for it in items
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract embedded images from HWP/HWPX to a flat folder."
    )
    p.add_argument("input", help="Path to input .hwp or .hwpx file")
    p.add_argument("out_dir", help="Directory to write extracted images (flat)")
    p.add_argument("-s", "--sample", type=int, default=None,
                   help="If set, extract at most N images (first N in sorted order)")
    p.add_argument("--min-bytes", type=int, default=0,
                   help="Skip payloads smaller than this many bytes (default 0)")
    p.add_argument("--include-preview", action="store_true",
                   help="Include preview/thumbnail-like streams (default: skip)")
    p.add_argument("--prefix", default="extracted_",
                   help="Output filename prefix (default: extracted_)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be extracted without writing files")

    args = p.parse_args(argv)
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        print(f"[ERR] Input not found: {in_path}", file=sys.stderr)
        return 2

    ensure_outdir(out_dir)

    fmt = detect_format(in_path)
    print(f"[INFO] Detected format: {fmt.upper()}")

    if fmt == "hwpx":
        items = extract_from_hwpx(in_path, out_dir, args.sample, args.min_bytes,
                                  args.include_preview, args.prefix, args.dry_run)
    else:
        try:
            items = extract_from_hwp(in_path, out_dir, args.sample, args.min_bytes,
                                     args.include_preview, args.prefix, args.dry_run)
        except RuntimeError as e:
            print(f"[ERR] {e}", file=sys.stderr)
            return 3

    print(f"[INFO] Extracted items: {len(items)}")
    if not args.dry_run:
        manifest_path = out_dir / "extracted_manifest.json"
        write_manifest(manifest_path, items, in_path)
        print(f"[INFO] Manifest written: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
