# 2_pre_process_image.py
"""
HWP 이미지 매핑용 전처리 스크립트 (CPU-only)

세트:
  • extracted  – HWP에서 추출된 이미지(625)
  • reference  – 삽입 전 원본 이미지(233)

출력:
  processed/extracted/{low|high}/{color|gray}/
  processed/reference/{low|high}/{color|gray}/
  → preprocess_mapping.json (출처/트랙/채널/레이블 기록)

예시(PowerShell):
  python .\2_pre_process_image.py `
    ".\images_output" `
    ".\processed\extracted" `
    ".\target_data\자동등록 사진 모음" `
    ".\processed\reference" `
    --low-size 640 --enable-high --min-pixels 10000 --colorspace lab --pad-mode replicate
  # (선택) --no-upscale / --only-extracted / --only-reference
  # (선택) --ext-label extracted --ref-label reference

주의:
  • PowerShell 백틱(`) 뒤 공백 금지
  • --orig-root / --only-original 은 폐기됨(호환용, 무시)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# ---- Defaults ----
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ---- Helpers ----
def natural_key(p: Path) -> Tuple:
    """자연 정렬 키(파일명 숫자 기준)."""
    import re
    s = p.name
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s))


def iter_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=natural_key)
    return files


def auto_trim(arr_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Otsu 기반 여백 자동 트림."""
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - thr
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return arr_rgb, False
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w < arr_rgb.shape[1] or h < arr_rgb.shape[0]:
        return arr_rgb[y : y + h, x : x + w], True
    return arr_rgb, False


def enhance(arr_rgb: np.ndarray, colorspace: str) -> Tuple[np.ndarray, np.ndarray]:
    """노이즈 제거 + 색공간 처리 → (RGB유지본, 8비트그레이)."""
    den = cv2.medianBlur(arr_rgb, 3)
    cs = colorspace.lower()

    if cs == "lab":
        lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L2 = _CLAHE.apply(L)
        lab2 = cv2.merge((L2, A, B))
        color_like = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        gray = L2
    elif cs == "hsv":
        hsv = cv2.cvtColor(den, cv2.COLOR_RGB2HSV)
        gray = hsv[..., 2]  # V
        color_like = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:  # rgb
        color_like = den
        gray = cv2.cvtColor(den, cv2.COLOR_RGB2GRAY)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return color_like, gray


def resize_letterbox(
    arr: np.ndarray,
    target: int,
    pad_mode: str,
    pad_color: Tuple[int, int, int],
    no_upscale: bool,
) -> np.ndarray:
    """긴 변을 target에 맞추고 정사각 패딩."""
    h, w = arr.shape[:2]
    if no_upscale and max(h, w) <= target:
        resized = arr
        nh, nw = h, w
    else:
        ratio = target / max(h, w)
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)

    top = (target - nh) // 2
    bottom = target - nh - top
    left = (target - nw) // 2
    right = target - nw - left

    if pad_mode == "replicate":
        border_type, value = cv2.BORDER_REPLICATE, 0
    else:
        border_type = cv2.BORDER_CONSTANT
        value = pad_color if resized.ndim == 3 else int(pad_color[0])

    return cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=value)


def save_image(out_path: Path, arr: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    img.save(str(out_path), format="PNG")


def process_one(
    src: Path,
    out_path: Path,
    size: int | None,
    min_pixels: int,
    colorspace: str,
    pad_mode: str,
    pad_color: Tuple[int, int, int],
    mapping: Dict[str, Dict],
    track: str,
    channel: str,
    no_upscale: bool,
    category: str,
) -> str:
    """단일 이미지 전처리 및 매핑 기록."""
    try:
        pil = Image.open(src)
        pil = ImageOps.exif_transpose(pil)
        arr_rgb = np.array(pil.convert("RGB"))
        h0, w0 = arr_rgb.shape[:2]
        if w0 * h0 < min_pixels:
            print(f"[SKIP] too small: {src.name} ({w0}×{h0})")
            return "skip"

        arr_rgb, trimmed = auto_trim(arr_rgb)
        if trimmed:
            print(f"[STEP] trim: {src.name}")

        color_like, gray = enhance(arr_rgb, colorspace)

        if size is not None:
            color_done = resize_letterbox(color_like, size, pad_mode, pad_color, no_upscale)
            gray_done = resize_letterbox(gray,       size, pad_mode, pad_color, no_upscale)
        else:
            color_done, gray_done = color_like, gray

        final = color_done if channel == "color" else gray_done
        save_image(out_path, final)

        mapping[out_path.name] = {
            "원본_전체_경로": str(src.resolve()),
            "원본_파일명": src.name,
            "확장자": src.suffix,
            "트랙": track,
            "채널": channel,
            "카테고리": category,  # extracted / reference
        }
        return "ok"
    except Exception as e:
        print(f"[ERROR] {src.name}: {e}")
        return "error"


def parse_pad_color(s: str) -> Tuple[int, int, int]:
    """'R,G,B' → (r,g,b), 0..255 검증."""
    try:
        parts = [int(x) for x in s.split(",")]
    except Exception as e:
        raise argparse.ArgumentTypeError("--pad-color must be 'R,G,B'") from e
    if len(parts) != 3 or any(not (0 <= v <= 255) for v in parts):
        raise argparse.ArgumentTypeError("--pad-color must be three integers in 0..255")
    return tuple(parts)  # type: ignore[return-value]


# ---- CLI ----
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess images into four variants (low/high × color/gray) for extracted/reference sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("extracted_src", help="HWP 추출 이미지 폴더")
    p.add_argument("dst_extracted", help="processed/extracted 출력 루트")
    p.add_argument("reference_src", help="삽입 전 원본(Reference) 이미지 폴더")
    p.add_argument("dst_reference", help="processed/reference 출력 루트")

    # 품질 정책
    p.add_argument("--low-size", type=int, default=640, help="'low' 트랙 캔버스 한 변")
    p.add_argument("--enable-high", action="store_true", help="'high' 트랙(원본 해상도)도 생성")
    p.add_argument("--no-upscale", action="store_true", help="'low' 트랙에서 업스케일 금지")
    p.add_argument("--min-pixels", type=int, default=0, help="가로×세로 < 값 이면 스킵")

    # 색공간/패딩
    p.add_argument("--colorspace", choices=["lab", "hsv", "rgb"], default="lab", help="LAB은 L에 CLAHE 적용")
    p.add_argument("--pad-mode", choices=["replicate", "constant"], default="replicate")
    p.add_argument("--pad-color", type=parse_pad_color, default=(0, 0, 0))

    # 부분 실행
    p.add_argument("--only-extracted", action="store_true", help="extracted만 처리")
    p.add_argument("--only-reference", action="store_true", help="reference만 처리")

    # 매핑 JSON 레이블
    p.add_argument("--ext-label", default="extracted", help="extracted 세트 레이블")
    p.add_argument("--ref-label", default="reference", help="reference 세트 레이블")

    # 폐기(호환)
    p.add_argument("--orig-root", default=None, help=argparse.SUPPRESS)            # deprecated, ignore
    p.add_argument("--only-original", action="store_true", help=argparse.SUPPRESS)  # deprecated alias

    return p


def main() -> None:
    args = build_argparser().parse_args()

    src_ext = Path(args.extracted_src)
    dst_ext = Path(args.dst_extracted)
    src_ref = Path(args.reference_src)
    dst_ref = Path(args.dst_reference)

    # 호환 플래그 병합
    only_reference = args.only_reference or getattr(args, "only_original", False)

    # 경고(비치명)
    if not src_ext.exists():
        print(f"[WARN] extracted_src not found: {src_ext}")
    if not src_ref.exists():
        print(f"[WARN] reference_src not found: {src_ref}")
    if dst_ext.resolve() == src_ext.resolve():
        print(f"[WARN] dst_extracted equals extracted_src")
    if dst_ref.resolve() == src_ref.resolve():
        print(f"[WARN] dst_reference equals reference_src")
    if getattr(args, "orig_root", None):
        print("[INFO] --orig-root is deprecated and ignored.")

    mapping: Dict[str, Dict] = {}
    counter = 1
    processed = skipped = failed = 0

    def run_batch(src_root: Path, dst_root: Path, category: str) -> None:
        nonlocal counter, processed, skipped, failed
        files = iter_images(src_root)
        if not files:
            print(f"[WARN] no images under: {src_root}")
            return

        tracks: List[Tuple[str, int | None]] = [("low", args.low_size)]
        if args.enable_high:
            tracks.append(("high", None))

        for fp in files:
            for track, size in tracks:
                for channel in ("color", "gray"):
                    suffix = "_g" if channel == "gray" else ""
                    out_name = f"{counter:06d}{suffix}.png"
                    out_dir = Path(dst_root, track, channel)
                    out_path = out_dir / out_name
                    print(f"[INFO] {category}/{track}/{channel} :: {fp.name} → {out_path.name}")
                    status = process_one(
                        src=fp,
                        out_path=out_path,
                        size=size,
                        min_pixels=args.min_pixels,
                        colorspace=args.colorspace,
                        pad_mode=args.pad_mode,
                        pad_color=args.pad_color,
                        mapping=mapping,
                        track=track,
                        channel=channel,
                        no_upscale=args.no_upscale,
                        category=category,
                    )
                    if status == "ok":
                        processed += 1
                    elif status == "skip":
                        skipped += 1
                    else:
                        failed += 1
                    counter += 1

    if not only_reference:
        print("[START] Preprocessing *extracted* …")
        run_batch(src_ext, dst_ext, args.ext_label)
    if not args.only_extracted:
        print("[START] Preprocessing *reference* …")
        run_batch(src_ref, dst_ref, args.ref_label)

    print(f"[SUMMARY] processed={processed}  skipped={skipped}  failed={failed}")

    with open("preprocess_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] preprocess_mapping.json  ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
