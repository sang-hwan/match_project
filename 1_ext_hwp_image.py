# 1_ext_hwp_image_debug.py
"""
Extract images from HWP5 (*.hwp) or HWPX (*.hwpx) files with controlled debug output.

Usage:
    python 1_ext_hwp_image_debug.py path/to/file.hwp out/dir [-s SAMPLE]
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
from PIL import Image, UnidentifiedImageError

# 스트림 이름에 이 키워드가 들어가면 미리보기용이므로 건너뜀
PREVIEW_KEYWORDS = ("PrvImage", "preview")


def _detect_type(hwp: Path) -> str:
    print(f"[DETECT] Reading signature from {hwp}")
    sig = hwp.open("rb").read(4)
    kind = "hwpx" if sig == b"PK\x03\x04" else "hwp5"
    print(f"[DETECT] Detected format: {kind}")
    return kind


def _extract_hwpx(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    print(f"[HWPPX] Extracting resources from HWPX")
    out: list[Path] = []
    with zipfile.ZipFile(hwp) as zf:
        all_names = [
            n for n in zf.namelist()
            if n.startswith("Contents/Resources/")
               and not any(k.lower() in n.lower() for k in PREVIEW_KEYWORDS)
        ]
        print(f"[HWPPX] Found {len(all_names)} candidates")
        chosen = (random.sample(all_names, sample) if sample else all_names)
        for idx, name in enumerate(chosen, 1):
            data = zf.read(name)
            dst = out_dir / f"{idx:03d}_{Path(name).name}"
            dst.write_bytes(data)
            out.append(dst)
            print(f"[HWPPX] Saved {dst.name} ({len(data)} bytes)")
    return out


def _extract_hwp5(hwp: Path, out_dir: Path, sample: int | None) -> list[Path]:
    print(f"[HWP5] Extracting streams from HWP5")
    from hwp_extract import HWPExtractor

    raw = hwp.read_bytes()
    print(f"[HWP5] File size: {len(raw)} bytes")
    extractor = HWPExtractor(data=raw)

    out: list[Path] = []
    saved = 0
    for idx, obj in enumerate(extractor.extract_files(), 1):
        # sample 개수만큼 저장했으면 중단
        if sample and saved >= sample:
            break

        name = getattr(obj, "name", f"stream_{idx}")
        if any(k.lower() in name.lower() for k in PREVIEW_KEYWORDS):
            continue  # 미리보기 스트림 건너뜀

        try:
            raw_data = obj.data
        except Exception:
            # 압축 해제 단계에서 에러 나면 그냥 건너뜀
            continue

        img_data: bytes | None = None

        # 1) magic-bytes 기반 빠른 포맷 감지
        if raw_data.startswith(b"\x89PNG") or raw_data[:2] == b"BM" or raw_data.startswith(b"\xff\xd8\xff"):
            img_data = raw_data
        else:
            # 2) PIL로 시도
            try:
                Image.open(BytesIO(raw_data)).verify()
                img_data = raw_data
            except Exception:
                # 3) zlib 압축 해제 + 확인
                for wb in (None, -zlib.MAX_WBITS):
                    try:
                        candidate = zlib.decompress(raw_data) if wb is None else zlib.decompress(raw_data, wb)
                        Image.open(BytesIO(candidate)).verify()
                        img_data = candidate
                        break
                    except Exception:
                        pass

                # 4) Ole10Native 내 JPEG·PNG 추출
                if img_data is None:
                    try:
                        with olefile.OleFileIO(BytesIO(raw_data)) as ole:
                            sub = ole.openstream("Ole10Native").read()
                            for sig in (b'\xff\xd8\xff', b'\x89PNG'):
                                pos = sub.find(sig)
                                if pos != -1:
                                    img_data = sub[pos:]
                                    break
                    except Exception:
                        pass

        if not img_data:
            # 유효 이미지가 아니면 건너뜀
            continue

        # 저장
        saved += 1
        dst = out_dir / f"{saved:03d}_{name.replace('/', '_')}"
        dst.write_bytes(img_data)
        out.append(dst)
        print(f"[HWP5] Saved #{saved}: {dst.name} ({len(img_data)} bytes)")

    print(f"[HWP5] Completed: saved {saved} image(s)")
    return out


def extract_images(hwp: Path, out_dir: Path, sample: int | None = None) -> list[Path]:
    print(f"[START] {hwp} → extract_images → {out_dir} (sample={sample})")
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = _detect_type(hwp)
    if kind == "hwpx":
        return _extract_hwpx(hwp, out_dir, sample)
    else:
        return _extract_hwp5(hwp, out_dir, sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract images from HWP/HWPX")
    parser.add_argument("hwp", help="*.hwp or *.hwpx file")
    parser.add_argument("out_dir", help="directory to save images")
    parser.add_argument("-s", "--sample", type=int, help="limit number of images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[MAIN] Cleaning output dir: {out_dir}")
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[MAIN] Processing file: {args.hwp}")
    saved = extract_images(Path(args.hwp), out_dir, args.sample)
    print(f"[DONE] Extracted {len(saved)} image(s) to {out_dir}")


if __name__ == "__main__":
    main()
