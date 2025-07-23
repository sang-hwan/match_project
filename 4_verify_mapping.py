#!/usr/bin/env python3
# 4_verify_mapping.py
"""
Mixed ORB+SIFT + RANSAC 검증 스크립트

이 스크립트는 pHash/Hist 타입에 따라 gray/raw 컬러 이미지를
각기 ORB 또는 SIFT 매칭 + RANSAC 검증을 통해 1:1 최종 매핑을 수행합니다.
입력 JSON 구조:
  { "orig1.png": [ {"name": "candA.png", "type": "pHash"}, ... ] }
출력 JSON 구조:
  { "orig1.png": "best_cand.png", ... }
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Matcher 설정
FLANN_INDEX_KDTREE = 1
# ORB(binary)용 BFMatcher
BF_ORB = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# SIFT용 FLANN KDTree Matcher
FLANN_SIFT = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
)
# Ratio test threshold
RATIO = 0.7


def compute_orb_descriptors(img_path: Path):
    """GRAYSCALE 이미지에 ORB 키포인트+디스크립터 계산"""
    with Image.open(img_path) as pil_img:
        arr = np.array(pil_img.convert('L'))
    orb = cv2.ORB_create(500)
    return orb.detectAndCompute(arr, None)


def compute_sift_descriptors(img_path: Path):
    """이미지에 SIFT 키포인트+디스크립터 계산"""
    with Image.open(img_path) as pil_img:
        arr = np.array(pil_img.convert('L'))
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(arr, None)


def compute_ransac_ratio(kps1, desc1, kps2, desc2, matcher) -> float:
    """매처+RANSAC 인라이어 비율 계산"""
    # Descriptor가 충분치 않으면 검증 불가 처리
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0.0
    # knnMatch에서 unpack 오류 방지
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good.append(m)
    # 충분한 매칭 없으면 스킵
    if len(good) < 4:
        return 0.0
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    return float(np.sum(mask) / len(mask))


def main():
    pa = argparse.ArgumentParser(description="pHash/Hist 타입별 ORB/SIFT+RANSAC 최종 매핑")
    pa.add_argument("candidates_json", help="pHash+Hist 후보 JSON 파일 경로")
    pa.add_argument("gray_ex_dir", help="processed/extracted/gray 디렉토리 경로 (preprocessed gray)")
    pa.add_argument("gray_orig_dir", help="processed/original/gray 디렉토리 경로 (preprocessed gray)")
    pa.add_argument("raw_ex_dir", help="HWP 추출 raw 컬러 이미지 폴더 경로 (images_output)")
    pa.add_argument("raw_orig_dir", help="원본 raw 컬러 이미지 루트 폴더 경로 (target_data/자동등록 사진 모음)")
    pa.add_argument("out_map", help="출력 매핑 JSON 파일 경로")
    args = pa.parse_args()

    print("[START] 후보 정보 로딩...")
    candidates = json.loads(Path(args.candidates_json).read_text(encoding='utf-8'))
    print(f"[INFO] 총 대상 이미지 수: {len(candidates)}")

    gray_ex_dir = Path(args.gray_ex_dir)
    gray_orig_dir = Path(args.gray_orig_dir)
    raw_ex_dir  = Path(args.raw_ex_dir)
    raw_orig_dir= Path(args.raw_orig_dir)
    mapping = {}

    print("[START] 타입별 매핑 검증 진행")
    for idx, (orig_rel, cand_list) in enumerate(candidates.items(), start=1):
        print(f"[{idx}/{len(candidates)}] 처리: {orig_rel} (후보 {len(cand_list)}개)")
        best_score, best_cand = -1.0, None
        for obj in cand_list:
            name = obj['name']
            typ  = obj.get('type', '')
            if typ == 'pHash':
                orig_path = gray_orig_dir / orig_rel.replace('.png', '_g.png')
                ex_path   = gray_ex_dir   / name.replace('.png', '_g.png')
                kps_o, desc_o = compute_orb_descriptors(orig_path)
                kps_e, desc_e = compute_orb_descriptors(ex_path)
                score = compute_ransac_ratio(kps_o, desc_o, kps_e, desc_e, BF_ORB)
            else:
                # raw 컬러 이미지 경로 탐색
                candidates_raw = list(raw_orig_dir.rglob(orig_rel))
                if not candidates_raw:
                    print(f"[WARN] raw 원본 미발견: {orig_rel}")
                    continue
                orig_path = candidates_raw[0]
                ex_path   = raw_ex_dir / name
                kps_o, desc_o = compute_sift_descriptors(orig_path)
                kps_e, desc_e = compute_sift_descriptors(ex_path)
                score = compute_ransac_ratio(kps_o, desc_o, kps_e, desc_e, FLANN_SIFT)
            print(f"    {typ}: {name}, score={score:.3f}")
            if score > best_score:
                best_score, best_cand = score, name
        if best_cand:
            mapping[orig_rel] = best_cand
            print(f"[MAP] {orig_rel} → {best_cand} (score={best_score:.3f})")
        else:
            print(f"[WARN] 매핑 실패: {orig_rel}")

    print(f"[DONE] 매핑 완료: {len(mapping)}/{len(candidates)} items")

    out = Path(args.out_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[SAVE] 결과 저장: {out}")

if __name__ == '__main__':
    main()
