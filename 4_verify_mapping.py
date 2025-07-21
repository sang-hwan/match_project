# 4_verify_mapping.py
"""
ORB + RANSAC 검증
- pHash로 추린 후보 정보를 기준으로 ORB+RANSAC 검증을 통해 최종 매핑을 수행합니다.
- {원본 이미지 경로+파일명+확장자명: 추출한 파일명+확장자명} 형태의 JSON을 저장합니다.

Usage
-----
python 4_verify_mapping.py \
  candidates.json \
  path/to/processed_extracted \
  path/to/processed_original \
  out_map.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

# FLANN matcher 설정
FLANN = cv2.FlannBasedMatcher(
    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
    dict(checks=50)
)
RATIO = 0.75  # NNDR 기준


def compute_orb_descriptors(img_path: Path):
    """이미지를 PIL로 읽어 ORB 키포인트와 디스크립터 계산"""
    with Image.open(img_path) as pil_img:
        gray_img = pil_img.convert('L')
        arr = np.array(gray_img)
    orb = cv2.ORB_create(1000)
    kps, desc = orb.detectAndCompute(arr, None)
    return kps, desc


def compute_ransac_ratio(kps1, desc1, kps2, desc2) -> float:
    """ORB 매칭 + RANSAC을 통해 인라이어 비율 계산"""
    if desc1 is None or desc2 is None:
        return 0.0
    matches = FLANN.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO * n.distance:
            good.append(m)
    if len(good) < 4:
        return 0.0
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None:
        return 0.0
    return float(np.sum(mask) / len(mask))


def main():
    pa = argparse.ArgumentParser(description="ORB+RANSAC 검증을 통한 최종 매핑")
    pa.add_argument("candidates_json", help="pHash 후보 매핑 정보 JSON 파일")
    pa.add_argument("extracted_dir", help="전처리된 추출 이미지 폴더 경로")
    pa.add_argument("original_dir", help="전처리된 원본 이미지 폴더 경로")
    pa.add_argument("out_map", help="출력 매핑 JSON 파일 경로")
    args = pa.parse_args()

    print("[START] 로딩 pHash 후보 매핑 정보...")
    candidates = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    print(f"[INFO] 총 원본 이미지 수: {len(candidates)}")

    extracted_dir = Path(args.extracted_dir)
    original_dir = Path(args.original_dir)
    mapping: dict[str, str] = {}

    print("[START] ORB+RANSAC 매핑 시작")
    for idx, (orig_rel, cand_list) in enumerate(candidates.items(), start=1):
        print(f"[{idx}/{len(candidates)}] 매핑 대상: {orig_rel} (후보 {len(cand_list)}개)")
        orig_path = original_dir / orig_rel
        kps_o, desc_o = compute_orb_descriptors(orig_path)
        best_score, best_cand = -1.0, None
        for cand_rel in cand_list:
            ext_path = extracted_dir / cand_rel
            kps_e, desc_e = compute_orb_descriptors(ext_path)
            score = compute_ransac_ratio(kps_o, desc_o, kps_e, desc_e)
            print(f"    후보: {cand_rel}, RANSAC 인라이어 비율={score:.3f}")
            if score > best_score:
                best_score, best_cand = score, cand_rel
        if best_cand is not None:
            mapping[orig_rel] = best_cand
            print(f"[MAP] {orig_rel} → {best_cand} (score={best_score:.3f})")
        else:
            print(f"[WARN] 일치하는 후보 없음: {orig_rel}")

    print(f"[DONE] 매핑 완료: {len(mapping)}/{len(candidates)} items")

    # --- 여기가 추가된 부분: 끝에 한 번만 붙은 ".png" 제거 ---
    clean_mapping: dict[str, str] = {}
    for orig_rel, cand_rel in mapping.items():
        # 끝에 .png가 붙어 있으면 한 번만 제거
        new_orig = orig_rel[:-4] if orig_rel.lower().endswith(".png") else orig_rel
        new_cand = cand_rel[:-4]  if cand_rel.lower().endswith(".png")  else cand_rel
        clean_mapping[new_orig] = new_cand

    # 결과 저장
    out_path = Path(args.out_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(clean_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[SAVE] 결과 저장됨: {out_path}")

if __name__ == '__main__':
    main()
