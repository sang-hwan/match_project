# 3_mk_phash_candidates.py
"""
pHash(Gray) + Color Histogram 후보 추림 스크립트

이 스크립트는 전처리된 그레이스케일(gray) 이미지와 컬러(color) 이미지 세트를 기반으로,
두 단계의 병렬 필터링(pHash와 컬러 히스토그램)을 적용하여 매핑 후보를 추출합니다.
결과는 {"이미지명": [{"name": 후보파일명, "type": "pHash"|"Hist"}, ...]} 구조의 JSON으로 저장됩니다.
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import imagehash
import cv2
import numpy as np

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULT_THRESH = 6
DEFAULT_HIST_THRESH = 0.5
DEFAULT_TOPN = 30
DEFAULT_TOPN_PHASH = DEFAULT_TOPN // 2
DEFAULT_TOPN_HIST = DEFAULT_TOPN - DEFAULT_TOPN_PHASH


def compute_phash(path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash for the given image file."""
    with Image.open(path) as img:
        return imagehash.phash(img)


def compute_color_hist(path: Path) -> np.ndarray:
    """Compute normalized HSV color histogram (3×32 bins) for the given image file."""
    # Use PIL to support unicode paths, then convert to OpenCV format
    with Image.open(path) as pil_img:
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    chans = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        cv2.normalize(h, h)
        chans.append(h.flatten())
    return np.concatenate(chans)


def main():
    pa = argparse.ArgumentParser(
        description="pHash(Gray) + ColorHist 후보 추림: preprocessed gray/color 이미지에서 병렬 필터링"
    )
    pa.add_argument("extracted_gray_dir", help="processed/extracted/gray 디렉토리 경로")
    pa.add_argument("original_gray_dir", help="processed/original/gray 디렉토리 경로")
    pa.add_argument("extracted_color_dir", help="processed/extracted/color 디렉토리 경로")
    pa.add_argument("original_color_dir", help="processed/original/color 디렉토리 경로")
    pa.add_argument("out_json", help="출력 JSON 파일 경로")
    pa.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESH,
        help=f"pHash 해밍 거리 임계값 (기본값: {DEFAULT_THRESH})"
    )
    pa.add_argument(
        "--hist-threshold", type=float, default=DEFAULT_HIST_THRESH,
        help=f"Color histogram 거리 임계값 (기본값: {DEFAULT_HIST_THRESH})"
    )
    pa.add_argument(
        "--topn-phash", type=int, default=DEFAULT_TOPN_PHASH,
        help=f"pHash 후보 최대 개수 (기본값: {DEFAULT_TOPN_PHASH})"
    )
    pa.add_argument(
        "--topn-hist", type=int, default=DEFAULT_TOPN_HIST,
        help=f"ColorHist 후보 최대 개수 (기본값: {DEFAULT_TOPN_HIST})"
    )
    args = pa.parse_args()

    threshold = args.threshold
    hist_threshold = args.hist_threshold
    topn_phash = args.topn_phash
    topn_hist = args.topn_hist

    print("[START] pHash(Gray)+ColorHist 후보 추림 시작")
    gray_ext_dir = Path(args.extracted_gray_dir)
    gray_ori_dir = Path(args.original_gray_dir)
    color_ext_dir = Path(args.extracted_color_dir)
    color_ori_dir = Path(args.original_color_dir)

    print(f"[INFO] Gray - 추출: {gray_ext_dir}, 원본: {gray_ori_dir}")
    print(f"[INFO] Color - 추출: {color_ext_dir}, 원본: {color_ori_dir}")

    # 파일 수집
    ext_gray_files = [p for p in gray_ext_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    ori_gray_files = [p for p in gray_ori_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    print(f"[INFO] Gray 파일 - 추출: {len(ext_gray_files)}, 원본: {len(ori_gray_files)}")

    # pHash 계산
    print("[STEP] Gray pHash 계산 중...")
    ext_gray_hashes = {p: compute_phash(p) for p in ext_gray_files}
    ori_gray_hashes = {p: compute_phash(p) for p in ori_gray_files}
    print(f"[DONE] Gray pHash 완료 (추출: {len(ext_gray_hashes)}, 원본: {len(ori_gray_hashes)})")

    # 기준 세트 결정
    if len(ori_gray_hashes) <= len(ext_gray_hashes):
        basis_hashes = ori_gray_hashes
        compare_hashes = ext_gray_hashes
        basis_gray_root, compare_gray_root = gray_ori_dir, gray_ext_dir
        basis_color_root, compare_color_root = color_ori_dir, color_ext_dir
        basis_label, compare_label = "original", "extracted"
    else:
        basis_hashes = ext_gray_hashes
        compare_hashes = ori_gray_hashes
        basis_gray_root, compare_gray_root = gray_ext_dir, gray_ori_dir
        basis_color_root, compare_color_root = color_ext_dir, color_ori_dir
        basis_label, compare_label = "extracted", "original"
    print(f"[INFO] 기준 세트: {basis_label}({len(basis_hashes)}) vs 비교 세트: {compare_label}({len(compare_hashes)})")

    # 비교 세트 ColorHist 사전 생성
    print("[STEP] 비교 세트 ColorHist 사전 생성 중...")
    compare_color_hists: dict[str, np.ndarray] = {}
    for img_path in compare_color_root.rglob('*'):
        if img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        rel_color = img_path.relative_to(compare_color_root).as_posix()
        compare_color_hists[rel_color] = compute_color_hist(img_path)
    print(f"[DONE] ColorHist 사전 생성 완료 ({len(compare_color_hists)})")

    # 매핑 생성
    mapping: dict[str, list[dict[str, str]]] = {}
    print("[START] 후보 매핑 생성")
    total = len(basis_hashes)
    for idx, (basis_path, basis_hash) in enumerate(basis_hashes.items(), start=1):
        # Key: _g 제거하여 일관된 이름 사용
        raw_rel_gray = basis_path.relative_to(basis_gray_root).as_posix()
        clean_rel = raw_rel_gray.replace('_g.png', '.png')
        print(f"[{idx}/{total}] 처리 중: {clean_rel}")

        # pHash 후보 거리 계산 및 정렬
        phash_dists = [
            (p.relative_to(compare_gray_root).as_posix(), basis_hash - h)
            for p, h in compare_hashes.items()
        ]
        phash_dists.sort(key=lambda x: x[1])

        # Basis 컬러 히스토그램 계산
        basis_color_path = basis_color_root / Path(clean_rel)
        if not basis_color_path.exists():
            print(f"[WARN] 컬러 이미지 미발견: {basis_color_path}")
            continue
        basis_hist = compute_color_hist(basis_color_path)

        # Hist 후보 거리 계산 및 정렬
        hist_dists = [
            (rel, cv2.compareHist(basis_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            for rel, hist in compare_color_hists.items()
        ]
        hist_dists.sort(key=lambda x: x[1])

        print(f"  ▶ pHash Top-{topn_phash} distances: {phash_dists[:topn_phash]}")
        print(f"  ▶ Hist Top-{topn_hist} distances: {hist_dists[:topn_hist]}")

        # 후보 선택
        candidates: list[dict[str, str]] = []
        # pHash 후보
        for name, dist in phash_dists:
            if dist > threshold or len([c for c in candidates if c['type']=='pHash']) >= topn_phash:
                break
            clean_name = name.replace('_g.png', '.png')
            candidates.append({"name": clean_name, "type": "pHash"})
        print(f"  ▶ pHash 후보 수: {len([c for c in candidates if c['type']=='pHash'])}")
        # Hist 후보
        for name, dist in hist_dists:
            if dist > hist_threshold or len([c for c in candidates if c['type']=='Hist']) >= topn_hist:
                break
            candidates.append({"name": name, "type": "Hist"})
        print(f"  ▶ Hist 후보 수: {len([c for c in candidates if c['type']=='Hist'])}")

        mapping[clean_rel] = candidates

    print(f"[DONE] 후보 매핑 완성 (총 항목: {len(mapping)})")

    # JSON 저장
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 매핑 JSON 저장: {out_path}")


if __name__ == '__main__':
    main()
