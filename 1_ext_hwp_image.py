# 1_ext_hwp_image.py
"""
Extract images from HWP5 (*.hwp) or HWPX (*.hwpx) files.

Usage
-----
python 1_ext_hwp_image.py path/to/file.hwp out/dir [-s 200]
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

# ──────────────────────────── configurable ─────────────────────────────
PREVIEW_KEYWORDS = ("PrvImage", "preview")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
# ----------------------------------------------------------------------


def _detect_type(hwp: Path) -> str:
    """Return 'hwpx' if PK‑ZIP signature otherwise 'hwp5'."""
    with hwp.open("rb") as f:
        return "hwpx" if f.read(4) == b"PK\x03\x04" else "hwp5"


def _extract_hwpx(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    out: list[Path] = []
    with zipfile.ZipFile(hwp) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.startswith("Contents/Resources/")
            and not any(k.lower() in n.lower() for k in PREVIEW_KEYWORDS)
        ]
        if sample:
            names = random.sample(names, min(sample, len(names)))
        for i, name in enumerate(names, 1):
            data = zf.read(name)
            dst = out_dir / f"{i:03d}_{Path(name).name}"
            dst.write_bytes(data)
            out.append(dst)
            print(f"[EXTRACT] {dst.name}")
    return out


def _extract_hwp5(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    """Requires `hwp_extract.HWPExtractor` to be installed/importable."""
    from hwp_extract import HWPExtractor  # local or PyPI

    out: list[Path] = []
    doc = HWPExtractor(data=hwp.read_bytes())
    objs = list(doc.extract_files())
    if sample:
        objs = random.sample(objs, min(sample, len(objs)))

    for i, obj in enumerate(objs, 1):
        if any(k.lower() in obj.name.lower() for k in PREVIEW_KEYWORDS):
            continue
        raw = obj.data
        for wb in (None, -zlib.MAX_WBITS):
            try:
                raw = zlib.decompress(raw, wb) if wb is not None else zlib.decompress(raw)
                break
            except zlib.error:
                pass

        img_data: bytes | None = None
        try:
            Image.open(BytesIO(raw)).verify()
            img_data = raw
        except Exception:
            try:
                ole = olefile.OleFileIO(BytesIO(raw))
                sub = ole.openstream("Ole10Native").read()
                idx = sub.find(b"\xff\xd8\xff")
                img_data = sub[idx:] if idx != -1 else None
            except Exception:
                pass

        if img_data:
            dst = out_dir / f"{i:03d}_{obj.name.replace('/', '_')}"
            dst.write_bytes(img_data)
            out.append(dst)
            print(f"[EXTRACT] {dst.name}")
    return out


def extract_images(hwp: Path, out_dir: Path, sample: int | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = _detect_type(hwp)
    return (
        _extract_hwpx(hwp, out_dir, sample)
        if kind == "hwpx"
        else _extract_hwp5(hwp, out_dir, sample)
    )


def main() -> None:
    pa = argparse.ArgumentParser(description="Extract images from HWP/HWPX")
    pa.add_argument("hwp", help="*.hwp or *.hwpx file")
    pa.add_argument("out_dir", help="folder to save images")
    pa.add_argument("-s", "--sample", type=int, help="limit number of images")
    args = pa.parse_args()

    out_dir = Path(args.out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    paths = extract_images(Path(args.hwp), out_dir, args.sample)
    print(f"[DONE] extracted {len(paths)} images to {out_dir}")


if __name__ == "__main__":
    main()
