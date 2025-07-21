# 2_pre_process_image.py
"""
Pre-process two sets of images for mapping:
 1) HWP에서 추출된 이미지
 2) 그 이미지의 원본 (선택한 서브폴더만)

Usage
-----
python 2_pre_process_image.py \
  path/to/extracted_src \
  path/to/processed_extracted \
  path/to/original_src_subfolder \
  path/to/processed_original \
  --orig-root path/to/original_root \
  --size 512 \
  --min-pixels 0 \
  [--colorspace {lab,hsv,rgb}] \
  [--no-gray] [--no-color] \
  [--pad-mode {replicate,constant}] \
  [--pad-color R,G,B]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # ensure UTF-8 output

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def preprocess(
    src: Path,
    dst_base: Path,
    size: int,
    min_pixels: int,
    colorspace: str = "lab",        # 'lab' | 'hsv' | 'rgb'
    save_gray: bool = True,
    save_color: bool = True,
    pad_mode: str = "replicate",    # 'replicate' | 'constant'
    pad_color: tuple[int,int,int] = (0,0,0),
) -> bool:
    """
    - EXIF orientation correction
    - Color conversion to LAB/HSV/RGB & extract L/V/Gray channel
    - Linear normalization 0-255
    - Aspect-preserving resize (long side -> size)
    - Square padding to (size, size)
    - Save color and/or gray PNG(s)
    """
    # 처리 옵션 로깅
    print(f"[PREPROCESS] {src.name} -> size={size}, min_pixels={min_pixels}, "
          f"colorspace={colorspace}, save_gray={save_gray}, save_color={save_color}, "
          f"pad_mode={pad_mode}, pad_color={pad_color}")
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if w * h < min_pixels:
            print(f"[SKIP]    {src.name} ({w}×{h} < {min_pixels})")
            return False

        # Prepare RGB array
        arr_rgb = np.array(img.convert("RGB"))

        # ---------- COLOR SPACE CONVERSION ----------
        if colorspace == "lab":
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
            gray = arr_cs[..., 0]             # L channel
        elif colorspace == "hsv":
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
            gray = arr_cs[..., 2]             # V channel
        else:  # 'rgb'
            arr_cs = arr_rgb
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)

        # Linear normalization of gray channel
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # ---------- ASPECT‑PRESERVE RESIZE ----------
        ratio = size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_cs = cv2.resize(arr_cs, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ---------- SQUARE PADDING ----------
        top = (size - new_h) // 2
        bottom = size - new_h - top
        left = (size - new_w) // 2
        right = size - new_w - left
        border_type = cv2.BORDER_REPLICATE if pad_mode == "replicate" else cv2.BORDER_CONSTANT
        pad_value = pad_color if pad_mode == "constant" else None

        color_sq = cv2.copyMakeBorder(
            resized_cs, top, bottom, left, right, border_type, value=pad_value
        )
        gray_sq = cv2.copyMakeBorder(
            resized_gray, top, bottom, left, right, border_type, value=pad_value
        )

        # ---------- SAVE OUTPUTS ----------
        dst_base.parent.mkdir(parents=True, exist_ok=True)
        saved_any = False
        if save_color:
            out_color = dst_base.with_suffix(".png")
            Image.fromarray(color_sq).save(str(out_color), format="PNG")
            print(f"[OK]      COLOR {src.name} -> {out_color.name}")
            saved_any = True
        if save_gray:
            out_gray = dst_base.with_name(dst_base.stem + "_g.png")
            Image.fromarray(gray_sq).save(str(out_gray), format="PNG")
            print(f"[OK]      GRAY  {src.name} -> {out_gray.name}")
            saved_any = True

        return saved_any

    except Exception as e:
        print(f"[FAIL]    {src.name}: {e}")
        return False


def rename_original(fp: Path, orig_root: Path) -> str:
    """
    Construct flat filename with full relative path parts from orig_root,
    joined by underscores, preserving original extension before .png
    """
    parts = fp.relative_to(orig_root).parts
    flat = "_".join(parts[:-1] + (fp.stem,))
    return f"{orig_root.name}_{flat}{fp.suffix}"


def parse_pad_color(s: str) -> tuple[int,int,int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("pad-color must be R,G,B")
    return tuple(parts)


def main() -> None:
    pa = argparse.ArgumentParser(description="Enhanced image preprocessing for mapping")
    pa.add_argument("extracted_src", help="HWP-extracted images folder")
    pa.add_argument("dst_extracted", help="Processed extracted images output folder")
    pa.add_argument("original_src", help="Subfolder under original root to process")
    pa.add_argument("dst_original", help="Processed original images output folder")
    pa.add_argument("--orig-root", required=True, help="Original root folder for naming")
    pa.add_argument("--size", type=int, default=512, help="Max dimension for resize")
    pa.add_argument("--min-pixels", type=int, default=0, help="Minimum pixel area to process")
    pa.add_argument(
        "--colorspace",
        choices=["lab", "hsv", "rgb"],
        default="lab",
        help="Color space for processing (lab: L channel, hsv: V channel, rgb: gray)"
    )
    pa.add_argument("--no-gray", action="store_true", help="Do not save gray output")
    pa.add_argument("--no-color", action="store_true", help="Do not save color output")
    pa.add_argument(
        "--pad-mode",
        choices=["replicate", "constant"],
        default="replicate",
        help="Padding mode for square output"
    )
    pa.add_argument(
        "--pad-color",
        type=parse_pad_color,
        default=(0, 0, 0),
        help="Padding color for constant mode, as R,G,B"
    )
    args = pa.parse_args()

    # 전체 설정 로깅
    print(f"[CONFIG] size={args.size}, min_pixels={args.min_pixels}, "
          f"colorspace={args.colorspace}, no_gray={args.no_gray}, no_color={args.no_color}, "
          f"pad_mode={args.pad_mode}, pad_color={args.pad_color}")

    src_ext = Path(args.extracted_src)
    dst_ext = Path(args.dst_extracted)
    src_ori = Path(args.original_src)
    dst_ori = Path(args.dst_original)
    orig_root = Path(args.orig_root)
    size = args.size
    min_px = args.min_pixels
    colorspace = args.colorspace
    save_gray = not args.no_gray
    save_color = not args.no_color
    pad_mode = args.pad_mode
    pad_color = args.pad_color

    # Process extracted images
    print(f"[START] Preprocessing extracted images from {src_ext}")
    ext_files = [p for p in src_ext.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[INFO]  Found {len(ext_files)} extracted images")
    cnt_ext = 0
    for idx, fp in enumerate(ext_files, start=1):
        print(f"[{idx}/{len(ext_files)}] Extracted: {fp.name}")
        base = dst_ext / fp.with_suffix("")
        if preprocess(fp, base, size, min_px, colorspace, save_gray, save_color, pad_mode, pad_color):
            cnt_ext += 1
    print(f"[DONE]  Extracted processed: {cnt_ext}/{len(ext_files)}")

    # Process original images
    print(f"[START] Preprocessing originals in {src_ori} (root: {orig_root})")
    ori_files = [p for p in src_ori.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[INFO]  Found {len(ori_files)} original images")
    cnt_ori = 0
    for idx, fp in enumerate(ori_files, start=1):
        rel = fp.relative_to(orig_root)
        print(f"[{idx}/{len(ori_files)}] Original: {rel}")
        name = rename_original(fp, orig_root)
        base = dst_ori / Path(name).with_suffix("")
        if preprocess(fp, base, size, min_px, colorspace, save_gray, save_color, pad_mode, pad_color):
            cnt_ori += 1
    print(f"[DONE]  Originals processed: {cnt_ori}/{len(ori_files)} -> {dst_ori}")


if __name__ == "__main__":
    main()
