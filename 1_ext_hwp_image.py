# 1_ext_hwp_image.py
"""
Image extractor for Hangul word-processor files:

* HWP5  (*.hwp)  – OLE container
* HWPX (*.hwpx) – ZIP container

Features
--------
• Detects the file type by header signature.  
• Saves every embedded raster image (PNG, JPEG, BMP), or a random
  subset when --sample is given.  
• Skips preview/thumbnail streams whose names contain
  'PrvImage' or 'preview'.  
• Prints concise progress logs.

Usage
-----
python 1_ext_hwp_image_debug.py path/to/file.hwp out/dir [-s 20]
"""

from __future__ import annotations

import argparse
import random
import shutil
import zlib
import zipfile
from io import BytesIO
from pathlib import Path

import olefile
from PIL import Image

# Streams/files whose names include these keywords are previews; skip them.
PREVIEW_KEYWORDS = ("PrvImage", "preview")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_type(hwp: Path) -> str:
    """Return 'hwpx' if the file is ZIP-based, otherwise 'hwp5'."""
    print(f"[DETECT] Reading signature from {hwp}")
    with hwp.open("rb") as fp:
        sig = fp.read(4)
    kind = "hwpx" if sig == b"PK\x03\x04" else "hwp5"
    print(f"[DETECT] File format detected: {kind}")
    return kind


def _extract_hwpx(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    """Extract images from a HWPX (ZIP) file."""
    print("[HWPX] Extracting resources …")
    output: list[Path] = []

    with zipfile.ZipFile(hwp) as zf:
        all_names = [
            n for n in zf.namelist()
            if n.startswith("Contents/Resources/")
            and not any(k.lower() in n.lower() for k in PREVIEW_KEYWORDS)
        ]
        print(f"[HWPX] {len(all_names)} candidate file(s) found")

        chosen = random.sample(all_names, sample) if sample else all_names
        for idx, name in enumerate(chosen, 1):
            data = zf.read(name)
            dst = out_dir / f"{idx:03d}_{Path(name).name}"
            dst.write_bytes(data)
            output.append(dst)
            print(f"[HWPX] Saved {dst.name} ({len(data)} bytes)")

    return output


def _extract_hwp5(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    """Extract images from a legacy HWP5 (OLE) file."""
    print("[HWP5] Extracting streams …")

    from hwp_extract import HWPExtractor  # local dependency

    raw = hwp.read_bytes()
    print(f"[HWP5] File size: {len(raw)} bytes")

    extractor = HWPExtractor(data=raw)
    output: list[Path] = []
    saved = 0

    for idx, obj in enumerate(extractor.extract_files(), 1):
        if sample and saved >= sample:
            break

        name = getattr(obj, "name", f"stream_{idx}")
        if any(k.lower() in name.lower() for k in PREVIEW_KEYWORDS):
            continue  # skip preview streams

        # Try to retrieve the raw payload; skip if it fails
        try:
            raw_data = obj.data
        except Exception:
            continue

        img_data: bytes | None = None

        # 1) Quick check by magic bytes
        if (raw_data.startswith(b"\x89PNG")
                or raw_data[:2] == b"BM"
                or raw_data.startswith(b"\xff\xd8\xff")):
            img_data = raw_data
        else:
            # 2) Validate with Pillow
            try:
                Image.open(BytesIO(raw_data)).verify()
                img_data = raw_data
            except Exception:
                # 3) Try zlib-inflated payloads
                for wb in (None, -zlib.MAX_WBITS):
                    try:
                        candidate = (zlib.decompress(raw_data)
                                     if wb is None
                                     else zlib.decompress(raw_data, wb))
                        Image.open(BytesIO(candidate)).verify()
                        img_data = candidate
                        break
                    except Exception:
                        pass
                # 4) Fallback: look for PNG/JPEG inside Ole10Native
                if img_data is None:
                    try:
                        with olefile.OleFileIO(BytesIO(raw_data)) as ole:
                            sub = ole.openstream("Ole10Native").read()
                            for sig in (b"\xff\xd8\xff", b"\x89PNG"):
                                pos = sub.find(sig)
                                if pos != -1:
                                    img_data = sub[pos:]
                                    break
                    except Exception:
                        pass

        if img_data is None:
            continue  # not a valid image

        saved += 1
        dst = out_dir / f"{saved:03d}_{name.replace('/', '_')}"
        dst.write_bytes(img_data)
        output.append(dst)
        print(f"[HWP5] Saved #{saved}: {dst.name} ({len(img_data)} bytes)")

    print(f"[HWP5] Completed: {saved} image(s) saved")
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_images(hwp: Path, out_dir: Path,
                   sample: int | None = None) -> list[Path]:
    """Dispatch to the correct extractor based on file type."""
    print(f"[START] {hwp} → {out_dir} (sample={sample})")
    out_dir.mkdir(parents=True, exist_ok=True)

    if _detect_type(hwp) == "hwpx":
        return _extract_hwpx(hwp, out_dir, sample)
    return _extract_hwp5(hwp, out_dir, sample)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedded images from HWP/HWPX files")
    parser.add_argument("hwp", help="*.hwp or *.hwpx file to process")
    parser.add_argument("out_dir", help="Directory where images will be saved")
    parser.add_argument("-s", "--sample", type=int,
                        help="Save only N randomly selected images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[MAIN] Cleaning output directory: {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[MAIN] Processing file: {args.hwp}")
    saved = extract_images(Path(args.hwp), out_dir, args.sample)

    print(f"[DONE] {len(saved)} image(s) extracted to {out_dir}")


if __name__ == "__main__":
    main()
