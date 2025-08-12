# 4_verify_mapping.py
"""
최종 1:1 매핑 산출기 (자동 폴더 생성 반영)

기능 요약
- candidates.json의 각 타깃(original_path)과 후보(extracted) 전부를 ORB+RANSAC으로 스코어링
- 임계값(orb-score, min-inliers, min-inlier-ratio)로 필터링
- 점수 내림차순 탐욕(greedy)으로 1:1 매핑 선정(동점은 inliers -> kpA -> kpB)
- pair_scores.csv와 mapping_result.json 출력
- 저장 경로의 부모 폴더를 자동 생성

핵심 변경점
- candidates.json의 키가 'low|<원본경로>' 형태여도 info["original_path"]를 우선 사용
- 전처리본 선택은 preprocess_mapping.json LUT 기반(카테고리/트랙/채널 일치)
- --orb-threshold 또는 --thresholds JSON(orb_score)을 사용 가능 (둘 다 있으면 --orb-threshold 우선)
- --color-first로 color 우선 사용(기본은 gray 우선)
- 저장 직전에 출력 경로의 부모 폴더 자동 생성
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from joblib import Memory
from tqdm import tqdm

cv2.setNumThreads(0)
DESC_CACHE = Memory(".cache/descriptors", verbose=0)


# ============== 유틸 ==============
def norm_path(p: str) -> str:
    return os.path.normpath(p).lower()


def safe_imread(path: str, flag=cv2.IMREAD_GRAYSCALE):
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, flag)
    if img is not None:
        return img
    # 윈도우 한글 경로 대응
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flag)
    except Exception:
        return None


# ============== ORB + RANSAC ==============
@DESC_CACHE.cache
def extract_orb(path: str, n: int = 1500):
    img = safe_imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.empty((0, 2), dtype=np.float32), None
    orb = cv2.ORB_create(nfeatures=n)
    kps, des = orb.detectAndCompute(img, None)
    pts = np.float32([kp.pt for kp in kps]) if kps else np.empty((0, 2), dtype=np.float32)
    return pts, des


def orb_score(a: str, b: str, nfeatures: int = 1500, ratio: float = 0.75, ransac_thresh: float = 5.0) -> Tuple[int, int, int]:
    ptsA, desA = extract_orb(a, n=nfeatures)
    ptsB, desB = extract_orb(b, n=nfeatures)
    if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
        return 0, len(ptsA), len(ptsB)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desA, desB, k=2)
    good = [m for m, n in matches if n is not None and m.distance < ratio * n.distance]
    if len(good) < 4:
        return 0, len(ptsA), len(ptsB)

    src = np.float32([ptsA[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ptsB[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, len(ptsA), len(ptsB)


# ============== LUT (preprocess_mapping.json) ==============
def build_lookup(pre_map: Dict, images_root: Path) -> Dict[str, Dict[Tuple[str, str, str], str]]:
    """
    identity(원본_전체_경로 정규화) -> {(category(extracted|reference), track(low|high), channel(gray|color)): relative processed path}
    """
    lut: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for proc_name, meta in pre_map.items():
        origin = meta.get("원본_전체_경로")
        cat = str(meta.get("카테고리")).lower()
        trk = str(meta.get("트랙")).lower()
        ch = str(meta.get("채널")).lower()
        if origin is None or cat not in ("extracted", "reference"):
            raise ValueError("preprocess_mapping.json: '원본_전체_경로' 또는 '카테고리'가 누락/오류입니다.")
        rel = f"{cat}/{trk}/{ch}/{proc_name}"
        lut.setdefault(norm_path(origin), {})[(cat, trk, ch)] = rel
    return lut


def category_of_identity(identity_path: str, track: str, lut: Dict[str, Dict[Tuple[str, str, str], str]]) -> Optional[str]:
    key = norm_path(identity_path)
    options = lut.get(key, {})
    cats = {c for (c, t, _), _rel in options.items() if t == track}
    if "extracted" in cats:
        return "extracted"
    if "reference" in cats:
        return "reference"
    return None


def pick_preprocessed(identity_path: str, track: str, prefer_gray: bool,
                      lut: Dict[str, Dict[Tuple[str, str, str], str]], images_root: Path) -> Optional[str]:
    """
    동일 identity 내 (카테고리→트랙→채널) 우선순위로 전처리본 경로 반환.
    없는 경우 None.
    """
    key = norm_path(identity_path)
    options = lut.get(key, {})
    if not options:
        return None

    cat = category_of_identity(identity_path, track, lut)
    if cat is None:
        return None

    order = ["gray", "color"] if prefer_gray else ["color", "gray"]
    for ch in order:
        tup = (cat, track, ch)
        if tup in options:
            return str(images_root / options[tup])

    # 동일 트랙 내 임의 채널 대체
    for (c2, t2, ch2), rel in options.items():
        if c2 == cat and t2 == track:
            return str(images_root / rel)

    return None


# ============== 페어 생성 ==============
def iter_candidate_pairs(candidates: Dict, lut: Dict[str, Dict[Tuple[str, str, str], str]]) -> Iterable[Tuple[str, str, str]]:
    """
    candidates.json -> (ex_identity, or_identity, track) 생성
    - 원본 경로는 info["original_path"]를 우선 사용, 없으면 키에서 '|' 분리
    - 카테고리는 LUT로 판별하여 extracted/reference를 구분
    """
    for key, info in candidates.items():
        track = str(info.get("track", "low")).lower()
        origin_path = info.get("original_path") or (key.split("|", 1)[-1] if "|" in key else key)
        items = info.get("candidates", [])
        if not isinstance(items, list):
            continue

        for c in items:
            name = c.get("name")
            if not name:
                continue

            ca = category_of_identity(origin_path, track, lut)
            cb = category_of_identity(name, track, lut)
            if ca is None or cb is None or ca == cb:
                continue

            if ca == "extracted" and cb == "reference":
                ex, ori = origin_path, name
            else:
                ex, ori = name, origin_path
            yield (ex, ori, track)


# ============== 스코어링/필터/할당 ==============
def resolve_threshold(args) -> float:
    if args.orb_threshold is not None:
        return float(args.orb_threshold)
    if args.thresholds and Path(args.thresholds).is_file():
        try:
            js = json.loads(Path(args.thresholds).read_text("utf-8"))
            val = js.get("orb_score", 0.0)
            return float(val) if val is not None else 0.0
        except Exception:
            return 0.0
    return 0.0


def score_one(pair: Tuple[str, str, str], prefer_gray: bool, lut, images_root: Path,
              nfeatures: int, ratio: float, ransac: float):
    ex_id, or_id, track = pair
    ex_used = pick_preprocessed(ex_id, track, prefer_gray, lut, images_root)
    or_used = pick_preprocessed(or_id, track, prefer_gray, lut, images_root)

    # 전처리본이 없으면 identity 원본 경로로 대체 시도
    if ex_used is None:
        ex_used = ex_id
    if or_used is None:
        or_used = or_id

    inl, kA, kB = orb_score(ex_used, or_used, nfeatures=nfeatures, ratio=ratio, ransac_thresh=ransac)
    score = inl / (kA + 1e-6)
    return dict(
        ex_id=Path(ex_id).name, or_id=Path(or_id).name,
        ex_path=ex_id, or_path=or_id,
        ex_used=str(ex_used), or_used=str(or_used),
        track=track, inliers=inl, kpA=kA, kpB=kB, score=float(score),
    )


def greedy_assign(rows: List[Dict]) -> List[Dict]:
    """
    점수 순 정렬 후 1:1 탐욕 할당.
    동점 타이브레이커: inliers -> kpA -> kpB
    """
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["score"], r["inliers"], r["kpA"], r["kpB"]),
        reverse=True,
    )
    used_ex, used_or = set(), set()
    chosen: List[Dict] = []
    for r in rows_sorted:
        if r["ex_path"] in used_ex or r["or_path"] in used_or:
            continue
        used_ex.add(r["ex_path"])
        used_or.add(r["or_path"])
        chosen.append(r)
    return chosen


# ============== CLI & MAIN ==============
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create final 1:1 mapping using ORB+RANSAC scoring on candidate pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-c", "--candidates", required=True, help="candidates.json")
    ap.add_argument("-m", "--mapping", default="preprocess_mapping.json")
    ap.add_argument("-i", "--images-root", default="processed")
    ap.add_argument("-o", "--output", default="mapping_result.json")
    ap.add_argument("--scores-csv", default="pair_scores.csv")
    ap.add_argument("--thresholds", default=None, help="thresholds.json (from 4_B)")
    ap.add_argument("--orb-threshold", type=float, default=None, help="override orb score threshold")
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument("--min-inlier-ratio", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--color-first", action="store_true", help="prefer color over gray")
    ap.add_argument("--orb-nfeatures", type=int, default=1500)
    ap.add_argument("--lowe-ratio", type=float, default=0.75)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    return ap.parse_args()


def main():
    args = parse_args()
    images_root = Path(args.images_root)

    candidates = json.loads(Path(args.candidates).read_text("utf-8"))
    pre_map = json.loads(Path(args.mapping).read_text("utf-8"))
    lut = build_lookup(pre_map, images_root)

    prefer_gray = not args.color_first
    orb_thr = resolve_threshold(args)
    print(f"[PARAM] ORB threshold: {orb_thr:.4f}")

    # 후보 페어 생성
    pairs = list(iter_candidate_pairs(candidates, lut))
    print(f"[INFO] candidate pairs to score: {len(pairs):,}")

    if not pairs:
        print("Scoring: 0it [00:00, ?it/s]")
        print("[WARN] no scores computed; exiting.")
        return

    # 스코어링 (멀티스레드)
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                score_one,
                p, prefer_gray, lut, images_root,
                args.orb_nfeatures, args.lowe_ratio, args.ransac_thresh
            )
            for p in pairs
        ]
        for f in tqdm(as_completed(futures), total=len(futures), ncols=80, desc="Scoring"):
            r = f.result()
            rows.append(r)

    # CSV 저장(전체 페어) ─ 부모 폴더 자동 생성
    Path(args.scores_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.scores_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {args.scores_csv}")

    # 필터링
    def keep(r):
        return (
            r["score"] >= orb_thr and
            r["inliers"] >= args.min_inliers and
            (r["inliers"] / (r["kpA"] + 1e-6)) >= args.min_inlier_ratio
        )

    filtered = [r for r in rows if keep(r)]
    print(f"[INFO] passed threshold filter: {len(filtered):,}")

    if not filtered:
        # 빈 결과라도 폴더를 생성하고 저장
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"pairs": [], "map": {}, "meta": {"orb_threshold": orb_thr}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[WARN] no pairs passed; wrote empty result to {args.output}")
        return

    # 1:1 탐욕 할당
    chosen = greedy_assign(filtered)
    print(f"[INFO] assigned 1:1 pairs: {len(chosen):,}")

    # 출력 구조: (1) 상세 리스트, (2) 간단 맵
    out_pairs = [
        {
            "original_path": r["or_path"],
            "extracted_path": r["ex_path"],
            "score": r["score"],
            "inliers": r["inliers"],
            "kpA": r["kpA"],
            "kpB": r["kpB"],
            "track": r["track"],
            "ex_used": r["ex_used"],
            "or_used": r["or_used"],
        }
        for r in chosen
    ]
    out_map = {r["or_path"]: r["ex_path"] for r in chosen}

    out_json = {
        "pairs": out_pairs,
        "map": out_map,
        "meta": {
            "orb_threshold": orb_thr,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
            "color_first": bool(args.color_first),
        },
        "stats": {
            "scored_pairs": len(rows),
            "passed_filter": len(filtered),
            "assigned_pairs": len(chosen),
        },
    }

    # JSON 저장 ─ 부모 폴더 자동 생성
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {args.output}")
    print("[DONE] mapping complete.")


if __name__ == "__main__":
    main()
