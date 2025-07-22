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
    colorspace: str = "lab",
    save_gray: bool = True,
    save_color: bool = True,
    pad_mode: str = "replicate",
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

        arr_rgb = np.array(img.convert("RGB"))
        # color space conversion
        if colorspace == "lab":
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
            gray = arr_cs[..., 0]
        elif colorspace == "hsv":
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
            gray = arr_cs[..., 2]
        else:
            arr_cs = arr_rgb
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # resize
        ratio = size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_cs = cv2.resize(arr_cs, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # padding
        top = (size - new_h) // 2
        bottom = size - new_h - top
        left = (size - new_w) // 2
        right = size - new_w - left
        border = cv2.BORDER_REPLICATE if pad_mode == "replicate" else cv2.BORDER_CONSTANT
        value = pad_color if pad_mode == "constant" else None
        color_sq = cv2.copyMakeBorder(resized_cs, top, bottom, left, right, border, value=value)
        gray_sq = cv2.copyMakeBorder(resized_gray, top, bottom, left, right, border, value=value)

        # save outputs
        saved = False
        # 컬러 저장
        if save_color:
            color_dir = dst_base.parent
            color_dir.mkdir(parents=True, exist_ok=True)
            out_color = color_dir / f"{dst_base.name}.png"
            Image.fromarray(color_sq).save(str(out_color), format='PNG')
            print(f"[OK]      COLOR {src.name} -> {out_color.name}")
            saved = True
        # 그레이 저장
        if save_gray:
            gray_dir = dst_base.parent.parent / 'gray'
            gray_dir.mkdir(parents=True, exist_ok=True)
            out_gray = gray_dir / f"{dst_base.name}_g.png"
            Image.fromarray(gray_sq).save(str(out_gray), format='PNG')
            print(f"[OK]      GRAY  {src.name} -> {out_gray.name}")
            saved = True

        return saved

    except Exception as e:
        print(f"[FAIL]    {src.name}: {e}")
        return False


def rename_original(fp: Path, orig_root: Path) -> str:
    """
    fp 예: target_data/자동등록 사진 모음/1. 냉동기/foo.jpg
    → 'target_data_자동등록 사진 모음_1. 냉동기_foo_jpg'
    """
    parts_dir = fp.relative_to(orig_root).parent.parts
    stem = fp.stem
    ext = fp.suffix.lstrip('.')
    flat = '_'.join(parts_dir + (stem,))
    return f"{orig_root.name}_{flat}_{ext}"


def parse_pad_color(s: str) -> tuple[int,int,int]:
    parts = [int(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('pad-color must be R,G,B')
    return tuple(parts)


def main():
    pa = argparse.ArgumentParser(description='Enhanced image preprocessing for mapping')
    pa.add_argument('extracted_src')
    pa.add_argument('dst_extracted')
    pa.add_argument('original_src')
    pa.add_argument('dst_original')
    pa.add_argument('--orig-root', required=True)
    pa.add_argument('--size', type=int, default=512)
    pa.add_argument('--min-pixels', type=int, default=0)
    pa.add_argument('--colorspace', choices=['lab','hsv','rgb'], default='lab')
    pa.add_argument('--no-gray', action='store_true')
    pa.add_argument('--no-color', action='store_true')
    pa.add_argument('--pad-mode', choices=['replicate','constant'], default='replicate')
    pa.add_argument('--pad-color', type=parse_pad_color, default=(0,0,0))
    args = pa.parse_args()

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

    print(f"[CONFIG] size={size}, min_pixels={min_px}, colorspace={colorspace}, "
          f"save_gray={save_gray}, save_color={save_color}, pad_mode={pad_mode}, pad_color={pad_color}")

    # extracted
    print(f"[START] extracted from {src_ext}")
    ext_files = [p for p in src_ext.rglob('*') if p.suffix.lower() in IMG_EXTS]
    for idx, fp in enumerate(ext_files, 1):
        rel = fp.relative_to(src_ext)
        stem = rel.stem
        extn = rel.suffix.lstrip('.')
        rel_dir = rel.parent
        # color
        base_color = dst_ext / 'color' / rel_dir / f"{stem}_{extn}"
        preprocess(fp, base_color, size, min_px, colorspace,
                   save_gray=False, save_color=True,
                   pad_mode=pad_mode, pad_color=pad_color)
        # gray
        base_gray = dst_ext / 'gray' / rel_dir / f"{stem}_{extn}"
        preprocess(fp, base_gray, size, min_px, colorspace,
                   save_gray=True, save_color=False,
                   pad_mode=pad_mode, pad_color=pad_color)
    print(f"[DONE] extracted -> {dst_ext}")

    # original
    print(f"[START] originals in {src_ori} (root {orig_root})")
    ori_files = [p for p in src_ori.rglob('*') if p.suffix.lower() in IMG_EXTS]
    for idx, fp in enumerate(ori_files,1):
        name = rename_original(fp, orig_root)
        # color
        base_c = dst_ori / 'color' / name
        preprocess(fp, base_c, size, min_px, colorspace,
                   save_gray=False, save_color=True,
                   pad_mode=pad_mode, pad_color=pad_color)
        # gray
        base_g = dst_ori / 'gray' / name
        preprocess(fp, base_g, size, min_px, colorspace,
                   save_gray=True, save_color=False,
                   pad_mode=pad_mode, pad_color=pad_color)
    print(f"[DONE] originals -> {dst_ori}")


if __name__ == '__main__':
    main()
