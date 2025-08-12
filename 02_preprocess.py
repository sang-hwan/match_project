"""
02_preprocess.py

CPU 친화 전처리 파이프라인:
- 오토 트림(가장 큰 컨투어 박스 기준, 실패 시 원본 유지)
- 리사이즈: max(side) = {low_size, high_size}, letterbox 패딩(선택) + pad-multiple 정렬
- 색 보정(옵션): LAB-CLAHE (L 채널만), 회색조는 CLAHE/히스토그램 보정 선택 가능
- 산출: {category}/{track}/{channel}/... 구조로 저장
- 매니페스트: preprocess_mapping.json

Usage
-----
python 02_preprocess.py EXTRACTED_IN EXTRACTED_OUT REFERENCE_IN REFERENCE_OUT \
    --low-size 640 --enable-high --high-size 1280 \
    --pad-mode edge --pad-multiple 16 \
    --colorspace lab --clahe-clip 2.0 --clahe-grid 8 \
    --gray-clahe \
    --map-json preprocess_mapping.json \
    --out-ext png

Notes
-----
- OpenCV(cv2) 필요. Windows 비ASCII 경로 대응을 위해 imdecode/imencode 기반 I/O 사용.
- pad-mode: none|constant|edge|reflect. letterbox는 pad-multiple에 맞춰 최소 패딩만 적용.
- high 트랙은 --enable-high로 활성화. high-size 기본 1280.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"[ERR] OpenCV(cv2) import 실패: {e}\npip install opencv-python")

# ---------------------------
# 유틸: 경로/이미지 I/O
# ---------------------------

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")

def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files

def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None

def imwrite_unicode(path: Path, img: np.ndarray, ext: str, jpeg_quality: int = 95, png_compress: int = 3) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if ext.lower() == ".jpg" or ext.lower() == ".jpeg":
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        elif ext.lower() == ".png":
            ok, enc = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compress)])
        elif ext.lower() == ".webp":
            ok, enc = cv2.imencode(".webp", img, [])
        else:
            # 기본 PNG
            ok, enc = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compress)])
        if not ok:
            return False
        enc.tofile(str(path))
        return True
    except Exception:
        return False

# ---------------------------
# 전처리 기본 연산
# ---------------------------

def to_bgr(img: np.ndarray) -> np.ndarray:
    """BGRA/GRAY 등 다양한 입력을 안전하게 BGR로 정규화."""
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def auto_trim(img_bgr: np.ndarray,
              method: str = "auto",
              canny1: int = 50,
              canny2: int = 150,
              min_area_ratio: float = 0.2,
              margin_ratio: float = 0.01) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    가장 큰 컨투어를 찾아 콘텐츠 영역으로 크롭. 실패 시 원본 반환.
    Returns: (cropped_img, (top, bottom, left, right) removed)
    """
    h, w = img_bgr.shape[:2]
    if method == "none":
        return img_bgr, (0,0,0,0)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, canny1, canny2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr, (0,0,0,0)

    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    area = bw * bh
    if area < min_area_ratio * (w * h):
        return img_bgr, (0,0,0,0)

    # 여백 마진
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    x0 = max(0, x - mx); y0 = max(0, y - my)
    x1 = min(w, x + bw + mx); y1 = min(h, y + bh + my)

    cropped = img_bgr[y0:y1, x0:x1].copy()
    return cropped, (y0, h - y1, x0, w - x1)

def resize_letterbox(img_bgr: np.ndarray,
                     target_max_side: int,
                     pad_multiple: int = 16,
                     pad_mode: str = "edge",
                     constant_color: Tuple[int,int,int]=(0,0,0)) -> Tuple[np.ndarray, float, Tuple[int,int,int,int]]:
    """
    max(h,w) -> target_max_side로 스케일링 후, (pad_multiple) 배수에 맞게 최소 패딩.
    Returns: (padded_img, scale, (top,bottom,left,right))
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return img_bgr, 1.0, (0,0,0,0)

    scale = target_max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if pad_multiple <= 1 or pad_mode == "none":
        return resized, scale, (0,0,0,0)

    def ceil_to_mult(x: int, m: int) -> int:
        return int(math.ceil(x / float(m)) * m)

    tgt_w = ceil_to_mult(new_w, pad_multiple)
    tgt_h = ceil_to_mult(new_h, pad_multiple)
    pad_w = max(0, tgt_w - new_w)
    pad_h = max(0, tgt_h - new_h)

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "none": cv2.BORDER_CONSTANT,  # won't be used
    }.get(pad_mode, cv2.BORDER_REPLICATE)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=constant_color)
    return padded, scale, (top, bottom, left, right)

def enhance_color_lab(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def to_gray(img_bgr: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g = clahe.apply(g)
    return g

# ---------------------------
# 매핑 스키마
# ---------------------------

@dataclass
class VariantRecord:
    category: str          # "extracted" | "reference"
    src_abspath: str
    src_relpath: str
    track: str             # "low" | "high"
    channel: str           # "color" | "gray"
    out_abspath: str
    out_relpath: str
    orig_wh: Tuple[int,int]
    trimmed_pad: Tuple[int,int,int,int]  # (top,bottom,left,right) removed
    scale: float
    pad: Tuple[int,int,int,int]          # (top,bottom,left,right) added

# ---------------------------
# 메인 처리
# ---------------------------

def process_one_image(category: str,
                      src_path: Path,
                      src_root: Path,
                      out_root: Path,
                      args) -> List[VariantRecord]:
    recs: List[VariantRecord] = []

    img0 = imread_unicode(src_path)
    if img0 is None:
        print(f"[WARN] 읽기 실패: {src_path}")
        return recs

    img0 = to_bgr(img0)
    H0, W0 = img0.shape[:2]
    rel = src_path.relative_to(src_root)

    # 1) Auto-trim
    img_trim, removed = auto_trim(
        img0, method=args.trim,
        canny1=args.canny1, canny2=args.canny2,
        min_area_ratio=args.min_area_ratio,
        margin_ratio=args.margin_ratio
    )

    # 공통 함수: 한 트랙(low/high) 처리
    def process_track(track_name: str, target_side: int) -> None:
        img_resized, scale, pad = resize_letterbox(
            img_trim, target_max_side=target_side,
            pad_multiple=args.pad_multiple, pad_mode=args.pad_mode
        )

        # color
        color = img_resized
        if args.colorspace == "lab":
            color = enhance_color_lab(color, clip=args.clahe_clip, grid=args.clahe_grid)

        # gray
        gray = to_gray(img_resized, use_clahe=args.gray_clahe)

        # 저장 경로: {out_root}/{track}/{channel}/<rel_without_ext>.{out-ext}
        stem = rel.with_suffix("")  # keep subdirs, remove ext
        out_color = out_root / track_name / "color" / (str(stem) + f".{args.out_ext}")
        out_gray  = out_root / track_name / "gray"  / (str(stem) + f".{args.out_ext}")

        ok1 = imwrite_unicode(out_color, color, f".{args.out_ext}", args.jpeg_q, args.png_c)
        ok2 = imwrite_unicode(out_gray,  gray,  f".{args.out_ext}", args.jpeg_q, args.png_c)

        if ok1:
            recs.append(VariantRecord(
                category=category,
                src_abspath=str(src_path.resolve()),
                src_relpath=str(rel).replace("\\", "/"),
                track=track_name,
                channel="color",
                out_abspath=str(out_color.resolve()),
                out_relpath=str(out_color.relative_to(out_root)).replace("\\", "/"),
                orig_wh=(W0, H0),
                trimmed_pad=removed,
                scale=scale,
                pad=pad
            ))
        else:
            print(f"[WARN] 저장 실패(color): {out_color}")

        if ok2:
            recs.append(VariantRecord(
                category=category,
                src_abspath=str(src_path.resolve()),
                src_relpath=str(rel).replace("\\", "/"),
                track=track_name,
                channel="gray",
                out_abspath=str(out_gray.resolve()),
                out_relpath=str(out_gray.relative_to(out_root)).replace("\\", "/"),
                orig_wh=(W0, H0),
                trimmed_pad=removed,
                scale=scale,
                pad=pad
            ))
        else:
            print(f"[WARN] 저장 실패(gray): {out_gray}")

    # low
    process_track("low", args.low_size)

    # high
    if args.enable_high:
        process_track("high", args.high_size)

    return recs

def process_category(category: str,
                     in_root: Path,
                     out_root: Path,
                     args) -> List[VariantRecord]:
    files = list_images(in_root)
    print(f"[INFO] {category}: 입력 {len(files)}개")
    all_recs: List[VariantRecord] = []
    for i, p in enumerate(files, 1):
        if args.max_files and i > args.max_files:
            break
        recs = process_one_image(category, p, in_root, out_root, args)
        all_recs.extend(recs)
        if i % 50 == 0:
            print(f"[INFO] {category}: 진행 {i}/{len(files)}")
    print(f"[INFO] {category}: 산출 변형 {len(all_recs)}개")
    return all_recs

# ---------------------------
# 매핑 저장
# ---------------------------

def build_mapping(args, recs_ex: List[VariantRecord], recs_ref: List[VariantRecord]) -> Dict:
    by_src = {"extracted": {}, "reference": {}}
    for r in recs_ex + recs_ref:
        cat = r.category
        d = by_src[cat].setdefault(r.src_relpath, {
            "src_abspath": r.src_abspath,
            "variants": {"low": {}, "high": {}},
            "orig_wh": r.orig_wh,
        })
        d["orig_wh"] = r.orig_wh  # 업데이트(일관성)
        d["variants"][r.track][r.channel] = r.out_abspath

    mapping = {
        "version": 1,
        "params": {
            "low_size": args.low_size,
            "enable_high": args.enable_high,
            "high_size": args.high_size,
            "pad_mode": args.pad_mode,
            "pad_multiple": args.pad_multiple,
            "colorspace": args.colorspace,
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
            "gray_clahe": args.gray_clahe,
            "trim": args.trim,
            "canny1": args.canny1,
            "canny2": args.canny2,
            "min_area_ratio": args.min_area_ratio,
            "margin_ratio": args.margin_ratio,
            "out_ext": args.out_ext,
        },
        "roots": {
            "extracted_in": str(Path(args.extracted_in).resolve()),
            "extracted_out": str(Path(args.extracted_out).resolve()),
            "reference_in": str(Path(args.reference_in).resolve()),
            "reference_out": str(Path(args.reference_out).resolve()),
        },
        "summary": {
            "extracted_files": len(set([r.src_relpath for r in recs_ex])),
            "reference_files": len(set([r.src_relpath for r in recs_ref])),
            "variants_total": len(recs_ex) + len(recs_ref),
        },
        "index": [asdict(r) for r in (recs_ex + recs_ref)],
        "by_src": by_src,
    }
    return mapping

def save_mapping(path: Path, mapping: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess extracted/reference images (trim, resize+letterbox, color/gray) and write mapping JSON.")
    ap.add_argument("extracted_in", help="추출 이미지 입력 루트")
    ap.add_argument("extracted_out", help="추출 이미지 출력 루트")
    ap.add_argument("reference_in", help="원본 이미지 입력 루트")
    ap.add_argument("reference_out", help="원본 이미지 출력 루트")

    ap.add_argument("--low-size", type=int, default=640, help="low 트랙 max(side)")
    ap.add_argument("--enable-high", action="store_true", help="high 트랙 생성 여부")
    ap.add_argument("--high-size", type=int, default=1280, help="high 트랙 max(side)")

    ap.add_argument("--pad-mode", choices=["none", "constant", "edge", "reflect"], default="edge", help="letterbox 패딩 방식")
    ap.add_argument("--pad-multiple", type=int, default=16, help="출력 크기를 해당 배수로 패딩(최소)")

    ap.add_argument("--colorspace", choices=["none", "lab"], default="lab", help="color 보정 방식 (LAB-CLAHE or 없음)")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="LAB-CLAHE clipLimit")
    ap.add_argument("--clahe-grid", type=int, default=8, help="LAB-CLAHE tileGridSize(정사각)")

    ap.add_argument("--gray-clahe", action="store_true", help="회색조에도 CLAHE 적용(기본 On 권장)")
    ap.add_argument("--trim", choices=["auto", "none"], default="auto", help="오토 트림 사용 여부")
    ap.add_argument("--canny1", type=int, default=50, help="Canny threshold1")
    ap.add_argument("--canny2", type=int, default=150, help="Canny threshold2")
    ap.add_argument("--min-area-ratio", type=float, default=0.2, help="컨투어 박스 최소 비율")
    ap.add_argument("--margin-ratio", type=float, default=0.01, help="트림 박스 여유 비율")

    ap.add_argument("--out-ext", choices=["png", "jpg", "jpeg", "webp"], default="png", help="출력 이미지 확장자")
    ap.add_argument("--jpeg-q", type=int, default=95, help="JPG 저장 품질")
    ap.add_argument("--png-c", type=int, default=3, help="PNG 압축 레벨(0~9)")

    ap.add_argument("--map-json", default="preprocess_mapping.json", help="매핑 JSON 저장 경로")
    ap.add_argument("--max-files", type=int, default=0, help="디버그용: 카테고리당 최대 처리 개수(0=무제한)")

    return ap.parse_args()

def main() -> int:
    args = parse_args()

    ex_in = Path(args.extracted_in)
    ex_out = Path(args.extracted_out)
    ref_in = Path(args.reference_in)
    ref_out = Path(args.reference_out)

    if not ex_in.exists():
        print(f"[ERR] 미존재 경로: {ex_in}"); return 2
    if not ref_in.exists():
        print(f"[ERR] 미존재 경로: {ref_in}"); return 2

    print("[INFO] 전처리 시작")
    recs_ex = process_category("extracted", ex_in, ex_out, args)
    recs_ref = process_category("reference", ref_in, ref_out, args)

    mapping = build_mapping(args, recs_ex, recs_ref)
    save_mapping(Path(args.map_json), mapping)

    print(f"[INFO] 매핑 저장: {args.map_json}")
    print(f"[INFO] 완료: extracted 변형 {len(recs_ex)}, reference 변형 {len(recs_ref)}, 총 {len(recs_ex)+len(recs_ref)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
