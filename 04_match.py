"""
04_match.py

Score candidate pairs with ORB + RANSAC and produce a 1:1 mapping.

Pipeline
--------
1) Load candidates.json (from 03_candidates.py)
   - Key: "low|<reference_src_relpath>" -> [{extracted, phash_dist, hist_dist}, ...]
2) Read preprocess_mapping.json to resolve actual image paths (low/gray).
3) Precompute ORB features for all needed images (reference+extracted).
4) Score each candidate pair:
   - BFMatcher KNN (k=2) + Lowe ratio
   - findHomography(..., RANSAC) -> inliers, inlier_ratio
   - Score = inliers + alpha * inlier_ratio
5) Two-phase selection:
   - Phase A (precision): accept pairs meeting (min_inliers_A, min_ratio_A)
   - Phase B (recall sweep, optional): for yet-unmatched refs, if top-1 meets (min_inliers_B, min_ratio_B)
     AND satisfies Top-2 margin rule -> accept
6) 1:1 assignment (greedy by score). If --assign hungarian and SciPy is available,
   we use Hungarian within each phase (A first, then B on remaining).

Outputs
-------
- pair_scores.csv : all scored pairs with metrics and pass label
- mapping_result.json : final 1:1 mapping

Usage
-----
python 04_match.py \
  --candidates candidates.json \
  --mapping preprocess_mapping.json \
  --scores pair_scores.csv \
  --output mapping_result.json \
  --orb-nfeatures 1000 --ratio 0.75 --ransac-th 5.0 \
  --min-inliers 8 --min-inlier-ratio 0.15 \
  --enable-recall-sweep \
  --min-inliers-b 6 --min-inlier-ratio-b 0.12 \
  --top2-margin 4 --top2-multiplier 1.25 \
  --score-alpha 5.0 \
  --assign greedy \
  --workers 8

Notes
-----
- All operations are CPU-only and use OpenCV.
- We use the low/gray variants from preprocess outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERR] OpenCV(cv2) import 실패: {e}\n  pip install opencv-python")

# --------------------------
# I/O helpers
# --------------------------

def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None

# --------------------------
# Mapping / candidates I/O
# --------------------------

@dataclass
class VariantPaths:
    color_low: Optional[Path]
    gray_low: Optional[Path]
    color_high: Optional[Path]
    gray_high: Optional[Path]

def load_mapping(mapping_json: Path) -> Tuple[Dict[str, VariantPaths], Dict[str, VariantPaths], Dict]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        mp = json.load(f)

    def _collect(cat: str) -> Dict[str, VariantPaths]:
        res = {}
        for src_rel, info in mp["by_src"][cat].items():
            v = info["variants"]
            def _get(track: str, ch: str) -> Optional[Path]:
                p = v.get(track, {}).get(ch)
                return Path(p) if p else None
            res[src_rel] = VariantPaths(
                color_low=_get("low", "color"),
                gray_low=_get("low", "gray"),
                color_high=_get("high", "color"),
                gray_high=_get("high", "gray"),
            )
        return res
    return _collect("extracted"), _collect("reference"), mp

@dataclass
class Cand:
    ex_rel: str
    phash_dist: int
    hist_dist: float

def load_candidates(candidates_json: Path) -> Dict[str, List[Cand]]:
    with open(candidates_json, "r", encoding="utf-8") as f:
        cj = json.load(f)
    cands: Dict[str, List[Cand]] = {}
    raw = cj.get("candidates", {})
    for key, arr in raw.items():
        lst = []
        for d in arr:
            lst.append(Cand(
                ex_rel = d["extracted"],
                phash_dist = int(d.get("phash_dist", -1)),
                hist_dist  = float(d.get("hist_dist", 1.0)),
            ))
        cands[key] = lst
    return cands

# --------------------------
# ORB feature extraction
# --------------------------

@dataclass
class Feat:
    kps: Optional[np.ndarray]    # shape (N, 2) float32
    desc: Optional[np.ndarray]   # shape (N, 32) uint8

def compute_orb(path: Path, nfeatures: int) -> Feat:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Feat(None, None)
    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    kps, des = orb.detectAndCompute(img, None)
    if kps is None or des is None or len(kps) == 0:
        return Feat(None, None)
    pts = np.array([k.pt for k in kps], dtype=np.float32)  # (N,2)
    return Feat(pts, des)

def precompute_features(ref_paths: Dict[str, VariantPaths],
                        ex_paths: Dict[str, VariantPaths],
                        needed_ref: List[str],
                        needed_ex: List[str],
                        nfeatures: int,
                        workers: int = 0) -> Tuple[Dict[str, Feat], Dict[str, Feat]]:
    """
    Simple sequential precompute (stable on all platforms).
    For CPU-only, ORB is fast enough; threads give modest gains.
    """
    ref_feats: Dict[str, Feat] = {}
    ex_feats: Dict[str, Feat] = {}

    # Reference
    for rel in needed_ref:
        p = ref_paths.get(rel).gray_low if rel in ref_paths else None
        if not p or not p.exists():
            ref_feats[rel] = Feat(None, None)
            continue
        ref_feats[rel] = compute_orb(p, nfeatures)

    # Extracted
    for rel in needed_ex:
        p = ex_paths.get(rel).gray_low if rel in ex_paths else None
        if not p or not p.exists():
            ex_feats[rel] = Feat(None, None)
            continue
        ex_feats[rel] = compute_orb(p, nfeatures)

    return ref_feats, ex_feats

# --------------------------
# Pair scoring
# --------------------------

@dataclass
class PairScore:
    ref_rel: str
    ex_rel: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    phash_dist: int
    hist_dist: float
    pass_label: str  # "", "A", "B"
    score: float

def ratio_test_knn(des1: np.ndarray, des2: np.ndarray, ratio: float) -> List[Tuple[int,int]]:
    """
    Return list of index pairs after Lowe's ratio test.
    """
    if des1 is None or des2 is None:
        return []
    if len(des1) == 0 or len(des2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    pairs = []
    for m in knn:
        if len(m) < 2:
            continue
        a, b = m[0], m[1]
        if a.distance < ratio * b.distance:
            pairs.append((a.queryIdx, a.trainIdx))
    return pairs

def ransac_inliers(pts1: np.ndarray, pts2: np.ndarray, pairs: List[Tuple[int,int]], ransac_th: float) -> Tuple[int, int, float]:
    """
    Compute homography with RANSAC; return (inliers, good_matches, inlier_ratio).
    pts1: reference keypoints (Nx2), pts2: extracted keypoints (Mx2)
    """
    gm = len(pairs)
    if gm < 4:
        return 0, gm, 0.0
    src = np.float32([pts1[i] for (i, j) in pairs])  # ref
    dst = np.float32([pts2[j] for (i, j) in pairs])  # ex
    try:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_th, maxIters=2000, confidence=0.995)
    except Exception:
        H, mask = None, None
    if mask is None:
        return 0, gm, 0.0
    inliers = int(mask.ravel().astype(np.uint8).sum())
    ratio = (inliers / gm) if gm > 0 else 0.0
    return inliers, gm, float(ratio)

def score_pair(ref_feat: Feat, ex_feat: Feat,
               ratio: float, ransac_th: float,
               phash_dist: int, hist_dist: float,
               alpha: float,
               ref_rel: str, ex_rel: str) -> PairScore:
    if ref_feat.kps is None or ref_feat.desc is None or ex_feat.kps is None or ex_feat.desc is None:
        return PairScore(ref_rel, ex_rel, 0, 0, 0.0, phash_dist, hist_dist, "", 0.0)
    pairs = ratio_test_knn(ref_feat.desc, ex_feat.desc, ratio)
    inl, gm, ir = ransac_inliers(ref_feat.kps, ex_feat.kps, pairs, ransac_th)
    s = float(inl) + float(alpha) * float(ir)
    return PairScore(ref_rel, ex_rel, gm, inl, ir, phash_dist, hist_dist, "", s)

# --------------------------
# Assignment utilities
# --------------------------

def greedy_assign(candidates: List[PairScore]) -> List[PairScore]:
    """
    Greedy 1:1 by descending score.
    """
    chosen: List[PairScore] = []
    used_ref = set()
    used_ex  = set()
    for p in sorted(candidates, key=lambda x: x.score, reverse=True):
        if p.ref_rel in used_ref or p.ex_rel in used_ex:
            continue
        chosen.append(p)
        used_ref.add(p.ref_rel)
        used_ex.add(p.ex_rel)
    return chosen

def hungarian_assign(candidates: List[PairScore]) -> List[PairScore]:
    """
    Optional Hungarian assignment on the given candidate set.
    We build a bipartite cost matrix with normalized scores.
    Fallback to greedy if SciPy is unavailable.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception:
        return greedy_assign(candidates)

    refs = sorted({p.ref_rel for p in candidates})
    exes = sorted({p.ex_rel for p in candidates})
    idx_ref = {r:i for i,r in enumerate(refs)}
    idx_ex  = {e:i for i,e in enumerate(exes)}

    # Normalize scores to [0,1]; higher = better -> cost = 1 - score_norm
    if not candidates:
        return []
    max_inl = max(p.inliers for p in candidates) or 1
    max_s   = max(p.score for p in candidates) or 1.0

    # We prefer inliers; mix with score for stability
    def norm_score(p: PairScore) -> float:
        a = p.inliers / max_inl
        b = p.score / max_s
        return 0.7*a + 0.3*b

    import numpy as np  # local
    INF = 1e6
    cost = np.full((len(refs), len(exes)), INF, dtype=np.float32)
    for p in candidates:
        cost[idx_ref[p.ref_rel], idx_ex[p.ex_rel]] = 1.0 - float(norm_score(p))

    row_ind, col_ind = linear_sum_assignment(cost)
    chosen: List[PairScore] = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= INF:
            continue
        ref = refs[r]; ex = exes[c]
        # find the PairScore
        best = None
        for p in candidates:
            if p.ref_rel == ref and p.ex_rel == ex:
                best = p; break
        if best is not None:
            chosen.append(best)
    return chosen

def select_assignment(candidates: List[PairScore], method: str) -> List[PairScore]:
    method = method.lower()
    if method == "hungarian":
        return hungarian_assign(candidates)
    return greedy_assign(candidates)

# --------------------------
# Phase selection logic
# --------------------------

def phase_A_filter(scores_by_ref: Dict[str, List[PairScore]],
                   min_inliers: int, min_ratio: float) -> List[PairScore]:
    pool: List[PairScore] = []
    for ref, lst in scores_by_ref.items():
        for p in lst:
            if p.inliers >= min_inliers and p.inlier_ratio >= min_ratio:
                q = PairScore(**{**p.__dict__})
                q.pass_label = "A"
                pool.append(q)
    return pool

def phase_B_candidates(scores_by_ref: Dict[str, List[PairScore]],
                       remaining_refs: List[str],
                       min_inliers_b: int, min_ratio_b: float,
                       top2_margin: int, top2_multiplier: float) -> List[PairScore]:
    """
    For each remaining reference, consider top-1 by score if it clears relaxed thresholds
    and also satisfies a Top-2 margin rule: (top1.inliers - top2.inliers >= margin) OR
    (top1.score / max(1e-6, top2.score) >= multiplier). If no top-2, accept top-1 if passes thresholds.
    """
    out: List[PairScore] = []
    for ref in remaining_refs:
        lst = sorted(scores_by_ref.get(ref, []), key=lambda x: x.score, reverse=True)
        if not lst:
            continue
        top1 = lst[0]
        if top1.inliers < min_inliers_b or top1.inlier_ratio < min_ratio_b:
            continue
        if len(lst) == 1:
            q = PairScore(**{**top1.__dict__}); q.pass_label = "B"; out.append(q); continue
        top2 = lst[1]
        cond = (top1.inliers - top2.inliers >= top2_margin) or \
               (top1.score / max(1e-6, top2.score) >= top2_multiplier)
        if cond:
            q = PairScore(**{**top1.__dict__}); q.pass_label = "B"; out.append(q)
    return out

# --------------------------
# CSV / JSON outputs
# --------------------------

def write_scores_csv(path: Path, scores: List[PairScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_rel","ex_rel","good_matches","inliers","inlier_ratio",
                    "phash_dist","hist_dist","pass","score"])
        for p in scores:
            w.writerow([p.ref_rel, p.ex_rel, p.good_matches, p.inliers,
                        f"{p.inlier_ratio:.6f}", p.phash_dist, f"{p.hist_dist:.6f}",
                        p.pass_label, f"{p.score:.6f}"])

def write_mapping_json(path: Path,
                       final_pairs: List[PairScore],
                       used_ref: set, used_ex: set,
                       all_refs: List[str], all_exs: List[str],
                       thresholds: Dict, assign_method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unmatched_refs = [r for r in all_refs if r not in used_ref]
    # We don't strictly need unassigned_extracted, but keep for diagnostics
    unassigned_ex = [e for e in all_exs if e not in used_ex]
    obj = {
        "version": 1,
        "track": "low",
        "basis": "reference",
        "thresholds": thresholds,
        "assign_method": assign_method,
        "stats": {
            "references_total": len(all_refs),
            "extracted_total": len(all_exs),
            "assigned": len(final_pairs),
            "unmatched_references": len(unmatched_refs),
        },
        "mapping": [
            {
                "reference": p.ref_rel,
                "extracted": p.ex_rel,
                "inliers": p.inliers,
                "inlier_ratio": p.inlier_ratio,
                "good_matches": p.good_matches,
                "phash_dist": p.phash_dist,
                "hist_dist": p.hist_dist,
                "score": p.score,
                "pass": p.pass_label,
            } for p in final_pairs
        ],
        "unmatched_references": unmatched_refs,
        "unassigned_extracted": unassigned_ex,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Score candidate pairs with ORB+RANSAC and produce final 1:1 mapping.")
    ap.add_argument("--candidates", required=True, help="candidates.json from 03_candidates.py")
    ap.add_argument("--mapping", default="preprocess_mapping.json", help="preprocess mapping json")
    ap.add_argument("--scores", default="pair_scores.csv", help="CSV path for all scored pairs")
    ap.add_argument("--output", default="mapping_result.json", help="Final mapping JSON output")

    # ORB/RANSAC params
    ap.add_argument("--orb-nfeatures", type=int, default=1000, help="ORB nfeatures")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    ap.add_argument("--ransac-th", type=float, default=5.0, help="findHomography RANSAC reprojection threshold")

    # Phase A thresholds (precision)
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument("--min-inlier-ratio", type=float, default=0.15)

    # Phase B thresholds (recall sweep)
    ap.add_argument("--enable-recall-sweep", action="store_true")
    ap.add_argument("--min-inliers-b", type=int, default=6)
    ap.add_argument("--min-inlier-ratio-b", type=float, default=0.12)
    ap.add_argument("--top2-margin", type=int, default=4)
    ap.add_argument("--top2-multiplier", type=float, default=1.25)

    # Scoring / assignment
    ap.add_argument("--score-alpha", type=float, default=5.0, help="score = inliers + alpha * inlier_ratio")
    ap.add_argument("--assign", choices=["greedy","hungarian"], default="greedy", help="1:1 assignment per phase")

    # Misc
    ap.add_argument("--workers", type=int, default=0, help="(reserved) feature parallelism; currently sequential")

    return ap.parse_args()

# --------------------------
# Main
# --------------------------

def main() -> int:
    args = parse_args()

    cand_path = Path(args.candidates)
    map_path  = Path(args.mapping)
    if not cand_path.exists():
        print(f"[ERR] candidates.json not found: {cand_path}")
        return 2
    if not map_path.exists():
        print(f"[ERR] preprocess_mapping.json not found: {map_path}")
        return 2

    # Load mapping and candidates
    ex_paths, ref_paths, mp = load_mapping(map_path)
    cands = load_candidates(cand_path)

    # Build needed sets
    ref_keys = sorted(cands.keys())  # "low|<rel>"
    ref_rels: List[str] = []
    for k in ref_keys:
        if not k.startswith("low|"):
            continue
        ref_rels.append(k.split("|",1)[1])

    # dedup extracted rels used in any candidate
    ex_rels_set = set()
    for k in ref_keys:
        for c in cands.get(k, []):
            ex_rels_set.add(c.ex_rel)
    ex_rels: List[str] = sorted(ex_rels_set)

    # Precompute ORB features (low/gray)
    print(f"[INFO] Precompute ORB features: refs={len(ref_rels)} ex={len(ex_rels)}")
    ref_feats, ex_feats = precompute_features(
        ref_paths, ex_paths, ref_rels, ex_rels, nfeatures=args.orb_nfeatures, workers=args.workers
    )

    # Score all pairs
    print("[INFO] Scoring candidate pairs...")
    scores_by_ref: Dict[str, List[PairScore]] = {}
    all_scores: List[PairScore] = []
    for k in ref_keys:
        if not k.startswith("low|"):
            continue
        ref_rel = k.split("|",1)[1]
        lst = []
        for c in cands.get(k, []):
            ref_feat = ref_feats.get(ref_rel, Feat(None,None))
            ex_feat  = ex_feats.get(c.ex_rel, Feat(None,None))
            ps = score_pair(ref_feat, ex_feat,
                            ratio=args.ratio, ransac_th=args.ransac_th,
                            phash_dist=c.phash_dist, hist_dist=c.hist_dist,
                            alpha=args.score_alpha,
                            ref_rel=ref_rel, ex_rel=c.ex_rel)
            lst.append(ps)
            all_scores.append(ps)
        # sort by score desc for later
        lst.sort(key=lambda x: x.score, reverse=True)
        scores_by_ref[ref_rel] = lst

    # Phase A
    pool_A = phase_A_filter(scores_by_ref, args.min_inliers, args.min_inlier_ratio)
    chosen_A = select_assignment(pool_A, args.assign)
    used_ref = set(p.ref_rel for p in chosen_A)
    used_ex  = set(p.ex_rel for p in chosen_A)
    print(f"[INFO] Phase A chosen: {len(chosen_A)}")

    # Phase B (optional)
    chosen_B: List[PairScore] = []
    if args.enable_recall_sweep:
        remain_refs = [r for r in ref_rels if r not in used_ref]
        pool_B = phase_B_candidates(scores_by_ref, remain_refs,
                                    args.min_inliers_b, args.min_inlier_ratio_b,
                                    args.top2_margin, args.top2_multiplier)
        # Filter out pairs conflicting with already used extracted
        pool_B = [p for p in pool_B if p.ex_rel not in used_ex]
        chosen_B = select_assignment(pool_B, args.assign)
        # Update used sets
        used_ref.update(p.ref_rel for p in chosen_B)
        used_ex.update(p.ex_rel for p in chosen_B)
        print(f"[INFO] Phase B chosen: {len(chosen_B)}")

    final_pairs = chosen_A + chosen_B

    # Mark pass labels inside all_scores for CSV
    accepted_pairs = {(p.ref_rel, p.ex_rel): p.pass_label for p in final_pairs}
    for ps in all_scores:
        lab = accepted_pairs.get((ps.ref_rel, ps.ex_rel), "")
        ps.pass_label = lab

    # Write outputs
    write_scores_csv(Path(args.scores), all_scores)
    thresholds = {
        "ratio": args.ratio,
        "ransac_th": args.ransac_th,
        "min_inliers_A": args.min_inliers,
        "min_inlier_ratio_A": args.min_inlier_ratio,
        "enable_recall_sweep": bool(args.enable_recall_sweep),
        "min_inliers_B": args.min_inliers_b,
        "min_inlier_ratio_B": args.min_inlier_ratio_b,
        "top2_margin": args.top2_margin,
        "top2_multiplier": args.top2_multiplier,
        "score_alpha": args.score_alpha,
    }
    write_mapping_json(Path(args.output), final_pairs, used_ref, used_ex,
                       ref_rels, ex_rels, thresholds, args.assign)

    print(f"[INFO] DONE: assigned={len(final_pairs)} / references={len(ref_rels)}")
    print(f"[INFO] pair_scores.csv -> {args.scores}")
    print(f"[INFO] mapping_result.json -> {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
