# 2_pre_process_image.py
"""
Enhanced image preprocessing with detailed logging and sequential numbering
  • Two resolution tracks: low (resize→square pad) & high (crop→denoise→CLAHE)
  • Automatic whitespace trim, median denoise, LAB→CLAHE
  • Save color & gray PNG outputs as 000001.png / 000001_g.png
  • Record mapping info in preprocess_mapping.json
  • Detailed print() logs for start, per-file, steps, summary
Usage
-----
python 2_pre_process_image.py \
  path/to/extracted_src \
  path/to/processed/extracted \
  path/to/original_src_subfolder \
  path/to/processed/original \
  --orig-root path/to/original_root \
  [--low-size 640] [--enable-high] \
  [--min-pixels 0] [--colorspace lab] [--pad-mode replicate] [--pad-color R,G,B]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def preprocess(
    src: Path,
    out_path: Path,
    size: int | None,
    min_pixels: int,
    colorspace: str,
    pad_mode: str,
    pad_color: tuple[int,int,int],
    mapping: dict,
    track: str,
    channel: str,
) -> str:
    """
    Process src image, save to out_path, record mapping, and return status.
    Returns: 'ok', 'skip', or 'error'.
    """
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        arr_rgb = np.array(img.convert("RGB"))
        h0, w0 = arr_rgb.shape[:2]
        if w0 * h0 < min_pixels:
            print(f"[SKIP] Too small: {src} ({w0}×{h0} px)")
            return 'skip'

        # 1) Trim whitespace
        gray0 = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trimmed = False
        if cnts:
            x, y, wc, hc = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            if wc < w0 or hc < h0:
                arr_rgb = arr_rgb[y:y+hc, x:x+wc]
                print(f"[STEP] Trim applied: {src}")
                trimmed = True

        # 2) Median denoise
        arr_rgb = cv2.medianBlur(arr_rgb, 3)
        print(f"[STEP] Median blur applied: {src}")

        # 3) Color space + CLAHE
        if colorspace == 'lab':
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(arr_cs)
            L2 = clahe.apply(L)
            arr_cs = cv2.merge((L2, A, B))
            gray = L2
            print(f"[STEP] CLAHE applied: {src}")
        elif colorspace == 'hsv':
            arr_cs = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2HSV)
            gray = arr_cs[...,2]
        else:
            arr_cs = arr_rgb
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4) Resize + pad or keep crop
        if size is not None:
            h1, w1 = arr_cs.shape[:2]
            ratio = size / max(w1, h1)
            nw, nh = int(w1 * ratio), int(h1 * ratio)
            cs_r = cv2.resize(arr_cs, (nw, nh), interpolation=cv2.INTER_AREA)
            gray_r = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            top = (size - nh) // 2; bottom = size - nh - top
            left = (size - nw) // 2; right = size - nw - left
            border = cv2.BORDER_REPLICATE if pad_mode == 'replicate' else cv2.BORDER_CONSTANT
            val = pad_color if pad_mode == 'constant' else None
            final = cv2.copyMakeBorder(cs_r if channel=='color' else gray_r, top, bottom, left, right, border, value=val)
        else:
            final = arr_cs if channel=='color' else gray

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final).save(str(out_path))
        mapping[out_path.name] = {
            "원본_전체_경로": str(src.resolve()),
            "원본_파일명": src.name,
            "확장자": src.suffix,
            "트랙": track,
            "채널": channel,
        }
        return 'ok'

    except Exception as e:
        print(f"[ERROR] Failed to process {src}: {e}")
        return 'error'


def parse_pad_color(s: str) -> tuple[int,int,int]:
    p = tuple(map(int, s.split(',')))
    if len(p) != 3:
        raise argparse.ArgumentTypeError('pad-color must be R,G,B')
    return p


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('extracted_src'); pa.add_argument('dst_extracted')
    pa.add_argument('original_src'); pa.add_argument('dst_original')
    pa.add_argument('--orig-root', required=True)
    pa.add_argument('--low-size', type=int, default=640)
    pa.add_argument('--enable-high', action='store_true')
    pa.add_argument('--min-pixels', type=int, default=0)
    pa.add_argument('--colorspace', choices=['lab','hsv','rgb'], default='lab')
    pa.add_argument('--pad-mode', choices=['replicate','constant'], default='replicate')
    pa.add_argument('--pad-color', type=parse_pad_color, default=(0,0,0))
    args = pa.parse_args()

    mapping = {}
    counter = 1
    processed = skipped = failed = 0

    src_ext = Path(args.extracted_src)
    dst_ext = Path(args.dst_extracted)
    src_ori = Path(args.original_src)
    dst_ori = Path(args.dst_original)
    orig_root = Path(args.orig_root)

    # Extracted images
    print("[START] Preprocessing extracted images...")
    for fp in src_ext.rglob('*'):
        if fp.suffix.lower() not in IMG_EXTS: continue
        for track, base in [('low', dst_ext/'low'), ('high', dst_ext/'high')] if args.enable_high else [('low', dst_ext/'low')]:
            for channel in ['color','gray']:
                suffix = '_g' if channel=='gray' else ''
                out_name = f"{counter:06d}{suffix}.png"
                out_path = base/track/channel/out_name if False else base/channel/out_name
                print(f"[INFO] Processing: {fp} → {out_path}")
                status = preprocess(fp, out_path,
                                   args.low_size if track=='low' else None,
                                   args.min_pixels, args.colorspace,
                                   args.pad_mode, args.pad_color,
                                   mapping, track, channel)
                if status == 'ok':
                    print(f"[OK] Saved: {out_path}")
                    processed += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    failed += 1
                counter += 1

    # Original images
    print("[START] Preprocessing original images...")
    for fp in src_ori.rglob('*'):
        if fp.suffix.lower() not in IMG_EXTS: continue
        for track, base in [('low', dst_ori/'low'), ('high', dst_ori/'high')] if args.enable_high else [('low', dst_ori/'low')]:
            for channel in ['color','gray']:
                suffix = '_g' if channel=='gray' else ''
                out_name = f"{counter:06d}{suffix}.png"
                out_path = base/channel/out_name
                print(f"[INFO] Processing: {fp} → {out_path}")
                status = preprocess(fp, out_path,
                                   args.low_size if track=='low' else None,
                                   args.min_pixels, args.colorspace,
                                   args.pad_mode, args.pad_color,
                                   mapping, track, channel)
                if status == 'ok':
                    print(f"[OK] Saved: {out_path}")
                    processed += 1
                elif status == 'skip':
                    skipped += 1
                else:
                    failed += 1
                counter += 1

    # Summary
    print(f"[SUMMARY] Total processed: {processed}, Skipped: {skipped}, Failed: {failed}")

    # Save mapping JSON
    with open('preprocess_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[MAPPING SAVED] preprocess_mapping.json ({len(mapping)} entries)")

if __name__ == '__main__':
    main()
