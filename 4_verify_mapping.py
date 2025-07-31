# 4_verify_mapping.py
#
# ──────────────────────────────────────────────────────────────
# ❶ candidates.json          : pHash/히스토그램으로 추린 후보 집합
# ❷ preprocess_mapping.json  : 전처리 ↔ 원본 메타 매핑
# ❸ 이미지 루트 디렉터리      : 실제 *.png / *.bmp … 파일이 존재하는 경로
#
# 목표
#   • 각 타깃(original) 에 대해 최종 1:1 대응 candidate 를 찾는다.
#   • ORB → (매칭 부족 시) SIFT → RANSAC 으로 정합성 판단
#   • 결과를 mapping_result.json 에 기록
#
# 실행 예시
#   $ python 4_verify_mapping.py \
#       --candidates   ./candidates.json \
#       --mapping      ./preprocess_mapping.json \
#       --img-root     ./all_images_root \
#       --output       ./mapping_result.json
#
# 주의
#   • CPU-only 환경 기준; OpenCV-contrib installing 필수(SIFT 사용 시)
#   • print() 만 사용해 세부 진행 로그를 모두 표출
# ──────────────────────────────────────────────────────────────
"""
Python ≥ 3.8, OpenCV-contrib ≥ 4.7 필요
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

# ───── 고정/조정 가능 파라미터 ─────
NFEATURES_ORB = 700
NFEATURES_SIFT = 300
LOWE_RATIO_ORB = 0.75          # pHash 거리에 따라 동적으로 조정할 수도 있음
LOWE_RATIO_SIFT = 0.70
MIN_MATCHES = 12
RANSAC_REPROJ_THRESH = 3.0      # px
RANSAC_MIN_INLIERS = 10
RANSAC_MIN_INLIER_RATIO = 0.4   # inliers / matches
USE_SIFT_FALLBACK = True


# ──────────────────────────────────────────────────────────────
def load_json(path: str) -> Dict:
    """UTF-8 JSON 로드 + 존재 여부 체크"""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found → {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def read_image(img_path: str) -> Any:
    """BGR 로 이미지를 읽어 반환; 실패 시 None"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Cannot read image: {img_path}")
    return img


def orb_match(
    img1, img2, ratio: float = LOWE_RATIO_ORB, nfeatures: int = NFEATURES_ORB
) -> Tuple[int, int, List[cv2.DMatch]]:
    """ORB 특징 매칭: (match_total, inlier_total, inlier_matches)"""
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, 0, []

    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return len(knn), len(good), good


def sift_match(
    img1, img2, ratio: float = LOWE_RATIO_SIFT, nfeatures: int = NFEATURES_SIFT
) -> Tuple[int, int, List[cv2.DMatch]]:
    """SIFT 특징 매칭: ORB 보완용 (opencv-contrib 필요)"""
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    except AttributeError:
        return 0, 0, []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, 0, []

    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return len(knn), len(good), good


def ransac_homography(
    kp1, kp2, matches: List[cv2.DMatch], reproj_thresh: float = RANSAC_REPROJ_THRESH
) -> Tuple[int, float]:
    """RANSAC으로 Homography & inlier 계산"""
    if len(matches) < MIN_MATCHES:
        return 0, 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(
        pts1, pts2, cv2.RANSAC, reproj_thresh, maxIters=1000
    )
    if mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(matches)
    return inliers, inlier_ratio


def evaluate_pair(
    work: Tuple[str, Dict, str, Dict[str, str]]
) -> Tuple[str, Dict[str, Any]]:
    """
    한 타깃 이미지와 후보 리스트 → 최적 후보 계산
    Return:
        target_path, {
            'best': {...},             # inlier 기준 최상 후보
            'all': [ {...}, ... ]      # (선택) 전체 후보 세부 결과
        }
    """
    target_path, info, img_root, preprocess_to_origin = work
    target_abs = str(Path(img_root) / target_path)
    img_target = read_image(target_abs)
    if img_target is None:
        return target_path, {"error": "target not found"}

    best = None
    all_results = []

    # ORB 객체 미리 생성 후 반복 사용
    orb = cv2.ORB_create(nfeatures=NFEATURES_ORB, fastThreshold=10)
    kp_t, des_t = orb.detectAndCompute(img_target, None)

    # Pre-extract SIFT for target (가능한 경우)
    sift = None
    kp_t_sift = des_t_sift = None
    if USE_SIFT_FALLBACK and hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=NFEATURES_SIFT)
        kp_t_sift, des_t_sift = sift.detectAndCompute(img_target, None)

    # 후보 루프
    for cand in info["candidates"]:
        cand_path = cand["name"]
        cand_abs = str(Path(img_root) / cand_path)
        img_cand = read_image(cand_abs)
        if img_cand is None:
            continue

        kp_c, des_c = orb.detectAndCompute(img_cand, None)
        bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf_orb.knnMatch(des_t, des_c, k=2) if des_t is not None and des_c is not None else []
        good = [m for m, n in knn if m.distance < LOWE_RATIO_ORB * n.distance]

        method_used = "ORB"
        # ORB 매칭 부족 시 SIFT 시도
        if len(good) < MIN_MATCHES and sift is not None:
            kp_c_sift, des_c_sift = sift.detectAndCompute(img_cand, None)
            bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            knn_sift = bf_sift.knnMatch(des_t_sift, des_c_sift, k=2) \
                if des_t_sift is not None and des_c_sift is not None else []
            good = [m for m, n in knn_sift if m.distance < LOWE_RATIO_SIFT * n.distance]
            kp_c = kp_c_sift
            kp_t_ = kp_t_sift
            method_used = "SIFT"
        else:
            kp_t_ = kp_t

        inliers, inlier_ratio = ransac_homography(kp_t_, kp_c, good)

        result = {
            "candidate": cand_path,
            "method": method_used,
            "matches": len(good),
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "phash": cand.get("phash"),
            "hist": cand.get("hist"),
        }
        all_results.append(result)

        # 최적 후보 갱신 (inliers → inlier_ratio → matches 순)
        if (
            inliers >= RANSAC_MIN_INLIERS
            and inlier_ratio >= RANSAC_MIN_INLIER_RATIO
        ):
            if (
                best is None
                or inliers > best["inliers"]
                or (
                    inliers == best["inliers"]
                    and inlier_ratio > best["inlier_ratio"]
                )
            ):
                best = result

    return target_path, {"best": best, "all": all_results}


def main():
    parser = argparse.ArgumentParser(
        description="Find 1:1 mapping via ORB → (SIFT) → RANSAC"
    )
    parser.add_argument("--candidates", "-c", required=True)
    parser.add_argument("--mapping", "-m", required=True)
    parser.add_argument(
        "--img-root",
        "-i",
        required=True,
        help="Root directory where all images reside",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="mapping_result.json",
        help="Output JSON path (default: mapping_result.json)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=max(cpu_count() - 1, 1),
        help="Parallel worker processes (default: CPU-1)",
    )
    args = parser.parse_args()

    print("▶ Loading JSON files …")
    candidates = load_json(args.candidates)
    preprocess_mapping = load_json(args.mapping)

    # (선택) preprocess → origin 역매핑 dict (검증 단계에서 사용 가능)
    preprocess_to_origin = {
        k: v["원본_전체_경로"] for k, v in preprocess_mapping.items()
    }

    tasks = [
        (target, info, args.img_root, preprocess_to_origin)
        for target, info in candidates.items()
    ]

    print(
        f"▶ Starting mapping: {len(tasks):,} targets, "
        f"{args.workers} workers …"
    )
    with Pool(processes=args.workers) as pool:
        results = pool.map(evaluate_pair, tasks)

    # 결과 집계
    mapping_result = {}
    unmatched = []
    for target, res in results:
        mapping_result[target] = res
        if res["best"] is None:
            unmatched.append(target)

    # 저장
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(mapping_result, fp, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    print(f"Total targets processed  : {len(tasks):,}")
    print(f"Matched (best != None)   : {len(tasks) - len(unmatched):,}")
    print(f"Unmatched                : {len(unmatched):,}")
    print(f"Output JSON              : {args.output}")
    print("====================")

    if unmatched:
        print("-- Unmatched targets (≤20) --")
        for t in unmatched[:20]:
            print("  ", t)


if __name__ == "__main__":
    main()
