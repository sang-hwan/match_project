#!/usr/bin/env python3
"""
Global candidate generation using pHash + Color Histogram with identity mapping via preprocess_mapping.json
  • Ensures gray & color images align by original source identity
  • Two-stage filtering: pHash distance ≤ threshold AND histogram distance ≤ threshold AND identical source
  • Returns top-K most similar candidates for each base image
  • Outputs mapping as JSON: {"basename.png": [{"name": candidate.png, "type": "pHash+Hist"}, ...]}
  • Detailed print() logs for parameter summary, input counts, per-image progress, and result summary
Usage
-----
python 3_mk_phash_candidates.py \
  path/to/extracted/low/gray \
  path/to/original/low/gray \
  path/to/extracted/low/color \
  path/to/original/low/color \
  output/candidates.json \
  --mapping preprocess_mapping.json \
  [--phash-threshold 18] \
  [--hist-threshold 0.3] \
  [--topk 20]
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_PHASH_THRESH = 18
DEFAULT_HIST_THRESH = 0.3
DEFAULT_TOPK = 20


def compute_phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as img:
        return imagehash.phash(img)


def compute_color_hist(path: Path) -> np.ndarray:
    with Image.open(path) as pil_img:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    chans = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0,256])
        cv2.normalize(h, h)
        chans.append(h.flatten())
    return np.concatenate(chans)


def main():
    pa = argparse.ArgumentParser(description="pHash+Hist 후보군 생성 with original identity mapping via preprocess_mapping.json")
    pa.add_argument("extracted_gray", help="processed/extracted/low/gray 디렉토리")
    pa.add_argument("original_gray", help="processed/original/low/gray 디렉토리")
    pa.add_argument("extracted_color", help="processed/extracted/low/color 디렉토리")
    pa.add_argument("original_color", help="processed/original/low/color 디렉토리")
    pa.add_argument("out_json", help="출력 후보군 JSON 경로")
    pa.add_argument("--mapping", required=True, help="preprocess_mapping.json 경로")
    pa.add_argument("--phash-threshold", type=int, default=DEFAULT_PHASH_THRESH,
                    help=f"pHash 해밍 거리 임계값 (기본: {DEFAULT_PHASH_THRESH})")
    pa.add_argument("--hist-threshold", type=float, default=DEFAULT_HIST_THRESH,
                    help=f"Histogram 거리 임계값 (기본: {DEFAULT_HIST_THRESH})")
    pa.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                    help=f"최종 후보 상위 K개 (기본: {DEFAULT_TOPK})")
    args = pa.parse_args()

    # Load identity mapping
    with open(args.mapping, encoding='utf-8') as mf:
        id_map = json.load(mf)

    phash_thresh = args.phash_threshold
    hist_thresh = args.hist_threshold
    topk = args.topk

    print(f"[START] pHash ≤ {phash_thresh} & Hist ≤ {hist_thresh} & same identity → Top-{topk} 후보 생성 시작")

    # Directory setup
    gray_ext_dir = Path(args.extracted_gray)
    gray_ori_dir = Path(args.original_gray)
    color_ext_dir = Path(args.extracted_color)
    color_ori_dir = Path(args.original_color)
    out_path = Path(args.out_json)

    # Collect files
    ext_gray = [p for p in gray_ext_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    ori_gray = [p for p in gray_ori_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    ext_color = [p for p in color_ext_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    ori_color = [p for p in color_ori_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    print(f"[INFO] Gray — Extracted: {len(ext_gray)}, Original: {len(ori_gray)}")
    print(f"[INFO] Color — Extracted: {len(ext_color)}, Original: {len(ori_color)}")

    # Compute pHash for gray images
    print("[INFO] Gray pHash 계산 중...")
    ext_hashes = {p: compute_phash(p) for p in ext_gray}
    ori_hashes = {p: compute_phash(p) for p in ori_gray}
    print("[DONE] Gray pHash 완료")

    # Determine basis vs compare set
    if len(ori_hashes) <= len(ext_hashes):
        basis_hashes, compare_hashes = ori_hashes, ext_hashes
        basis_gray_root, compare_gray_root = gray_ori_dir, gray_ext_dir
        basis_color_root, compare_color_root = color_ori_dir, color_ext_dir
        basis_label = 'original'
    else:
        basis_hashes, compare_hashes = ext_hashes, ori_hashes
        basis_gray_root, compare_gray_root = gray_ext_dir, gray_ori_dir
        basis_color_root, compare_color_root = color_ext_dir, color_ori_dir
        basis_label = 'extracted'
    print(f"[INFO] 기준 세트: {basis_label} ({len(basis_hashes)}) vs 비교 세트 ({len(compare_hashes)})")

    # Cache compare histograms
    print("[INFO] 컬러 히스토그램 캐시 생성중...")
    compare_hists = {}
    for img_path in compare_color_root.rglob('*'):
        if img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        key = img_path.name
        try:
            compare_hists[key] = compute_color_hist(img_path)
        except Exception as e:
            print(f"[WARN] Hist 실패: {img_path.name}: {e}")
    print(f"[DONE] Hist 캐시 완료: {len(compare_hists)} items")

    # Generate candidates
    mapping = {}
    total = len(basis_hashes)
    for idx, (bpath, bhash) in enumerate(basis_hashes.items(), start=1):
        bkey = bpath.name
        orig_id = id_map.get(bkey, {}).get('원본_전체_경로')
        rel_gray = bpath.relative_to(basis_gray_root).as_posix()
        print(f"[{idx:03d}/{total}] 처리: {rel_gray}")

        if not orig_id:
            print(f"[WARN] 매핑 정보 없음: {bkey}")
            mapping[rel_gray] = []
            continue

        # Find corresponding color preprocessed filename(s)
        color_candidates = [fname for fname, meta in id_map.items()
                            if meta.get('채널')=='color' and meta.get('원본_전체_경로')==orig_id]
        if not color_candidates:
            print(f"[WARN] 기준 컬러 없음 (identity): {bkey}")
            mapping[rel_gray] = []
            continue

        # Use first matched color file
        cfile = color_candidates[0]
        basis_color_path = basis_color_root / cfile
        if not basis_color_path.exists():
            print(f"[WARN] 파일 미발견: {basis_color_path}")
            mapping[rel_gray] = []
            continue

        # Compute basis histogram
        try:
            basis_hist = compute_color_hist(basis_color_path)
        except Exception as e:
            print(f"[ERROR] Hist 오류: {basis_color_path.name}: {e}")
            mapping[rel_gray] = []
            continue

        # pHash distances
        pdists = [(p.relative_to(compare_gray_root).as_posix(), bhash - h)
                  for p,h in compare_hashes.items()]
        pdists.sort(key=lambda x: x[1])
        phash_pass = sum(1 for _,d in pdists if d<=phash_thresh)

        # Filter by identity & histogram
        candidates = []
        hist_pass = 0
        for rel, dist in pdists:
            if dist>phash_thresh:
                break
            ck = Path(rel).name
            cid = id_map.get(ck,{}).get('원본_전체_경로')
            if cid!=orig_id:
                continue
            hvec = compare_hists.get(ck)
            if hvec is None:
                continue
            hd = cv2.compareHist(basis_hist, hvec, cv2.HISTCMP_BHATTACHARYYA)
            if hd<=hist_thresh:
                hist_pass+=1
                candidates.append({"name": rel, "type": "pHash+Hist"})
                if len(candidates)>=topk:
                    break

        print(f"▶ pHash 통과: {phash_pass} → Identity+Hist: {hist_pass} → 채택: {len(candidates)}")
        mapping[rel_gray] = candidates

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 후보군 JSON 저장완료: {out_path} ({len(mapping)} items)")

if __name__ == '__main__':
    main()
