# 4_verify_mapping.py
"""
Verify 1:1 mapping ORIGINAL → EXTRACTED using ORB+RANSAC scores.

Inputs : candidates.json, preprocess_mapping.json, (opt) thresholds.json
Outputs: mapping_result.json, pair_scores.csv, unmatched.json, logs

Example:
python 4_verify_mapping.py ^
  -c candidates.json ^
  -m preprocess_mapping.json ^
  -i processed ^
  -o mapping_result.json ^
  --scores-csv pair_scores.csv ^
  --thresholds thresholds.json ^
  --workers 8
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from joblib import Memory
from tqdm import tqdm

cv2.setNumThreads(0)
CACHE = Memory(".cache/verify_orb", verbose=0)

# ─────────────── Path / IO ───────────────
def norm_identity(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower()

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_thresholds(thr_path: Optional[Path]) -> Optional[float]:
    if thr_path and thr_path.is_file():
        try:
            val = float(load_json(thr_path).get("orb_score", 0.0))
            return val if val > 0 else None
        except Exception:
            return None
    return None

# ─────────────── Mapping lookup ───────────────
def norm_channel(ch: str) -> str:
    t = str(ch).strip().lower()
    if t in {"gray", "grey", "그레이"}: return "gray"
    if t in {"color", "colour", "컬러"}: return "color"
    return t

def build_lookup(mapping: dict) -> Dict[str, Dict[Tuple[str, str, str], str]]:
    """
    identity(original full-path, normalized) ->
        {(category, track, channel): relative_preproc_path}
    """
    lut: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for proc_fname, meta in mapping.items():
        origin = norm_identity(meta["원본_전체_경로"])
        track = str(meta["트랙"]).lower()
        channel = norm_channel(meta["채널"])
        category = "extracted" if "bindata" in origin else "original"
        rel = f"{category}/{track}/{channel}/{proc_fname}"
        lut.setdefault(origin, {})[(category, track, channel)] = rel
    return lut

def pick_preproc(origin: str, preferred_cat: str, track: str, prefer_gray: bool,
                 lut: Dict[str, Dict[Tuple[str, str, str], str]]) -> Optional[str]:
    key = norm_identity(origin)
    options = lut.get(key, {})
    if not options:
        return None
    order = ["gray", "color"] if prefer_gray else ["color", "gray"]
    # same category first
    for ch in order:
        tup = (preferred_cat, track, ch)
        if tup in options:
            return options[tup]
    # opposite category fallback (same track)
    for (cat, trk, ch), rel in options.items():
        if trk == track and ch in order:
            return rel
    return None

# ─────────────── Robust image loading & features ───────────────
def _safe_imread_gray(p: Path) -> Optional[np.ndarray]:
    if not p.is_file():
        return None
    try:
        buf = np.fromfile(str(p), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

@CACHE.cache
def _extract_orb(path_str: str, nfeat: int):
    img = _safe_imread_gray(Path(path_str))
    if img is None:
        return np.zeros((0, 2), np.float32), None
    orb = cv2.ORB_create(nfeatures=nfeat, fastThreshold=10)
    kps, des = orb.detectAndCompute(img, None)
    pts = np.float32([kp.pt for kp in (kps or [])])
    return pts, des

def _match_orb(desA, desB, ratio: float) -> List[cv2.DMatch]:
    if desA is None or desB is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    m = bf.knnMatch(desA, desB, k=2)
    return [a for a, b in m if b is not None and a.distance < ratio * b.distance]

def _ransac_inliers(ptsA, ptsB, matches: List[cv2.DMatch], ransac_thresh: float, min_matches: int) -> Tuple[int, float]:
    if len(matches) < min_matches:
        return 0, 0.0
    src = np.float32([ptsA[m.queryIdx] for m in matches])
    dst = np.float32([ptsB[m.trainIdx] for m in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return 0, 0.0
    inl = int(mask.sum())
    return inl, inl / max(len(matches), 1)

def pair_score(a_path: str, b_path: str, nfeatures: int, lowe_ratio: float,
               ransac_thresh: float, min_matches: int) -> Tuple[int, int, int, float, float]:
    """
    Returns (inliers, kpA, kpB, inlier_ratio, score); score = inliers / kpA
    """
    ptsA, desA = _extract_orb(a_path, nfeat=nfeatures)
    ptsB, desB = _extract_orb(b_path, nfeat=nfeatures)
    kpA, kpB = len(ptsA), len(ptsB)
    if kpA == 0 or kpB == 0:
        return 0, kpA, kpB, 0.0, 0.0
    good = _match_orb(desA, desB, lowe_ratio)
    inl, ratio = _ransac_inliers(ptsA, ptsB, good, ransac_thresh, min_matches)
    score = inl / (kpA + 1e-6)
    return inl, kpA, kpB, ratio, float(score)

# ─────────────── Candidate expansion ───────────────
def iter_candidate_pairs(candidates: dict, lut: Dict[str, Dict[Tuple[str, str, str], str]],
                         img_root: Path, gray_first: bool = True) -> Iterable[Tuple[str, str, str, str, str]]:
    """
    Yield (orig_path, ex_path, track, orig_img_abs, ex_img_abs)
    All pairs normalized to ORIGINAL → EXTRACTED.
    """
    for key_origin, info in candidates.items():
        track = str(info.get("track", "low")).lower()
        key_cat = "extracted" if "bindata" in norm_identity(key_origin) else "original"

        key_rel = pick_preproc(key_origin, key_cat, track, gray_first, lut) or \
                  pick_preproc(key_origin, key_cat, track, not gray_first, lut)
        if not key_rel:
            continue
        key_img = str(img_root / key_rel)

        cand_list = info.get("candidates") or info.get("extracted_candidates") or []
        for cand in cand_list:
            cand_origin = cand.get("name")
            if not cand_origin:
                continue
            cand_cat = "extracted" if "bindata" in norm_identity(cand_origin) else "original"
            if cand_cat == key_cat:
                continue

            cand_rel = pick_preproc(cand_origin, cand_cat, track, gray_first, lut) or \
                       pick_preproc(cand_origin, cand_cat, track, not gray_first, lut)
            if not cand_rel:
                continue
            cand_img = str(img_root / cand_rel)

            if key_cat == "original" and cand_cat == "extracted":
                yield key_origin, cand_origin, track, key_img, cand_img
            else:
                yield cand_origin, key_origin, track, cand_img, key_img

# ─────────────── Greedy 1:1 assignment ───────────────
def greedy_assign(sorted_rows: List[dict]) -> Dict[str, str]:
    matched_o: set[str] = set()
    matched_e: set[str] = set()
    mapping: Dict[str, str] = {}
    for r in sorted_rows:
        o, e = r["orig"], r["extracted"]
        if o in matched_o or e in matched_e:
            continue
        matched_o.add(o)
        matched_e.add(e)
        mapping[o] = e
    return mapping

# ─────────────── Worker (top-level for multiprocessing) ───────────────
def _score_task(t: Tuple[str, str, str, str, str, int, float, float, int]):
    o, e, trk, op, ep, nfeat, lowe, rthr, min_matches = t
    inl, kA, kB, ratio, score = pair_score(op, ep, nfeat, lowe, rthr, min_matches)
    return dict(
        orig=o, extracted=e, track=trk,
        inliers=inl, kpA=kA, kpB=kB, inlier_ratio=ratio, score=score,
        orig_img=op, extracted_img=ep
    )

# ─────────────── CLI ───────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify 1:1 mapping by ORB+RANSAC over candidate pairs")
    p.add_argument("-c", "--candidates", required=True, help="candidates.json")
    p.add_argument("-m", "--mapping", required=True, help="preprocess_mapping.json")
    p.add_argument("-i", "--img-root", required=True, help="root processed dir (contains extracted/original)")
    p.add_argument("-o", "--output", default="mapping_result.json", help="output mapping JSON (ORIGINAL→EXTRACTED)")
    p.add_argument("--scores-csv", default="pair_scores.csv", help="CSV of all scored pairs")
    p.add_argument("--unmatched-json", default="unmatched.json", help="list of originals without assignment")
    p.add_argument("--thresholds", default=None, help="thresholds.json (from 4_B_extract_threshold.py)")
    p.add_argument("--orb-threshold", type=float, default=None, help="override ORB score threshold (inliers/kpA)")
    p.add_argument("--min-inliers", type=int, default=6)
    p.add_argument("--min-inlier-ratio", type=float, default=0.15)
    p.add_argument("--min-matches", type=int, default=8, help="minimum good matches to run RANSAC")
    p.add_argument("--orb-nfeatures", type=int, default=800)
    p.add_argument("--lowe-ratio", type=float, default=0.75)
    p.add_argument("--ransac-thresh", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=max(cpu_count() - 1, 1))
    p.add_argument("--color-first", action="store_true", help="prefer color over gray (default: gray-first)")
    return p.parse_args()

# ─────────────── Main ───────────────
def main():
    args = parse_args()

    candidates = load_json(Path(args.candidates))
    mapping_mp = load_json(Path(args.mapping))
    lut = build_lookup(mapping_mp)
    img_root = Path(args.img_root).resolve()

    thr_json = read_thresholds(Path(args.thresholds)) if args.thresholds else None
    orb_thr = args.orb_threshold if args.orb_threshold is not None else thr_json
    print(f"[PARAM] ORB threshold: {orb_thr:.4f}" if orb_thr is not None else "[PARAM] ORB threshold: (none)")

    pairs = list(iter_candidate_pairs(candidates=candidates, lut=lut, img_root=img_root, gray_first=not args.color_first))
    print(f"[INFO] candidate pairs to score: {len(pairs):,}")

    tasks = [
        (o, e, trk, op, ep, args.orb_nfeatures, args.lowe_ratio, args.ransac_thresh, args.min_matches)
        for (o, e, trk, op, ep) in pairs
    ]

    rows: List[dict] = []
    with Pool(processes=args.workers) as pool:
        for r in tqdm(pool.imap_unordered(_score_task, tasks), total=len(tasks), ncols=80, desc="Scoring"):
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no scores computed; exiting.")
        save_json({}, Path(args.output))
        save_json([], Path(args.unmatched_json))
        return

    df.sort_values(["score", "inliers"], ascending=[False, False], inplace=True)
    df.to_csv(args.scores_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVE] pair_scores.csv  rows={len(df):,}  → {args.scores_csv}")

    # Filters
    mask = (df["inliers"] >= args.min_inliers) & (df["inlier_ratio"] >= args.min_inlier_ratio)
    if orb_thr is not None:
        mask &= (df["score"] >= float(orb_thr))
    eligible = df[mask].copy()
    print(f"[INFO] eligible pairs after filters: {len(eligible):,}")

    # Greedy 1:1 assignment (max score first)
    eligible_sorted = eligible.sort_values(["score", "inliers"], ascending=[False, False]).to_dict("records")
    mapping = greedy_assign(eligible_sorted)

    all_originals = {o for (o, _, _, _, _) in pairs}
    unmatched = sorted(list(all_originals - set(mapping.keys())))

    save_json(mapping, Path(args.output))
    save_json(unmatched, Path(args.unmatched_json))

    print("\n[SUMMARY]")
    print(f"total ORIGINALs referenced : {len(all_originals)}")
    print(f"matched                    : {len(mapping)}")
    print(f"unmatched                  : {len(unmatched)}")
    if unmatched[:10]:
        print("first unmatched examples:")
        for u in unmatched[:10]:
            print("  -", u)

if __name__ == "__main__":
    main()
