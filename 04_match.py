"""
04_match.py

Score candidate pairs with feature matching + geometric verification,
optionally photometric verification (SSIM/NCC), then assign a 1:1 mapping.

Inputs
------
- candidates.json (from 03_candidates.py)
- preprocess_mapping.json (from 02_preprocess.py)

Outputs
-------
- pair_scores.csv : per (ref, ex) scored rows (legacy-compatible columns preserved)
- mapping_result.json :
    {
      "version": 2,
      "basis": "reference",
      "track": "low",
      "parameters": {...},
      "stats": {...},
      "mapping": { "low|<ref_rel>": "<ex_rel>", ... },   # legacy-friendly
      "assignments": [ {detail...} ],
      "unmatched_reference": [...],
      "scores_csv": "<path to pair_scores.csv>"
    }

Key features
------------
- Detector ensemble: ORB (default) + optional SIFT/AKAZE/BRISK
- Ratio test + optional mutual check (symmetric)
- Homography via RANSAC; tries USAC if available in OpenCV build
- Photometric verification (optional): warp ref->ex and compute SSIM / NCC
- Conservative acceptance policy:
    - Phase A (precision-first):  inliers >= thA & inlier_ratio >= thA
    - Phase B (optional recall):  inliers >= thB & inlier_ratio >= thB
    - Top-2 gap rule & (optional) SSIM/NCC thresholds
- Greedy 1:1 assignment by score (Hungarian fallback is not used due to SciPy dependency)

Notes
-----
- CPU-only friendly. Caches descriptors in-memory during a single run.
- Uses low/gray variant for geometry & photometric checks.
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

# =========================
# ---------- I/O ----------
# =========================

def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None

# =========================
# ---- Mapping helpers ----
# =========================

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
        by_src = {}
        for src_rel, info in mp["by_src"][cat].items():
            v = info["variants"]
            def _get(track: str, ch: str) -> Optional[Path]:
                p = v.get(track, {}).get(ch)
                return Path(p) if p else None
            by_src[src_rel] = VariantPaths(
                color_low=_get("low", "color"),
                gray_low=_get("low", "gray"),
                color_high=_get("high", "color"),
                gray_high=_get("high", "gray"),
            )
        return by_src

    return _collect("extracted"), _collect("reference"), mp

# =========================
# ---- SSIM / NCC ---------
# =========================

def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img

def compute_ssim(x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Simplified global SSIM over the overlapping region (not windowed).
    Good enough for a strong photometric gate.
    """
    x = _to_float01(x)
    y = _to_float01(y)
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() < 64:
            return 0.0
        x = x[m]
        y = y[m]
    if x.size < 64 or y.size < 64:
        return 0.0
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    ux = float(np.mean(x))
    uy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - ux) * (y - uy)))
    num = (2 * ux * uy + C1) * (2 * cov + C2)
    den = (ux * ux + uy * uy + C1) * (vx + vy + C2)
    if den <= 0:
        return 0.0
    s = num / den
    return float(max(0.0, min(1.0, s)))

def compute_ncc(x: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    x = _to_float01(x)
    y = _to_float01(y)
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() < 64:
            return 0.0
        x = x[m]
        y = y[m]
    if x.size < 64 or y.size < 64:
        return 0.0
    x -= x.mean()
    y -= y.mean()
    sx = float(np.linalg.norm(x))
    sy = float(np.linalg.norm(y))
    if sx <= 1e-9 or sy <= 1e-9:
        return 0.0
    return float(np.dot(x, y) / (sx * sy))

# =========================
# ---- Feature matching ----
# =========================

@dataclass
class DetPack:
    name: str
    detector: object
    norm: int

def make_detectors(use_orb: bool, use_sift: bool, use_akaze: bool, use_brisk: bool,
                   orb_nfeatures: int) -> List[DetPack]:
    packs: List[DetPack] = []

    if use_orb:
        try:
            det = cv2.ORB_create(nfeatures=int(orb_nfeatures))
            packs.append(DetPack("ORB", det, cv2.NORM_HAMMING))
        except Exception:
            print("[WARN] ORB 생성 실패 -> 건너뜀")

    if use_sift:
        # SIFT availability differs by OpenCV build
        det = None
        if hasattr(cv2, "SIFT_create"):
            try:
                det = cv2.SIFT_create()
            except Exception:
                det = None
        if det is None:
            try:
                # contrib path
                det = cv2.xfeatures2d.SIFT_create()  # type: ignore
            except Exception:
                det = None
        if det is not None:
            packs.append(DetPack("SIFT", det, cv2.NORM_L2))
        else:
            print("[WARN] SIFT 미지원(OpenCV 빌드 확인) -> 비활성")

    if use_akaze:
        try:
            det = cv2.AKAZE_create()
            packs.append(DetPack("AKAZE", det, cv2.NORM_HAMMING))
        except Exception:
            print("[WARN] AKAZE 생성 실패 -> 건너뜀")

    if use_brisk:
        try:
            det = cv2.BRISK_create()
            packs.append(DetPack("BRISK", det, cv2.NORM_HAMMING))
        except Exception:
            print("[WARN] BRISK 생성 실패 -> 건너뜀")

    if not packs:
        raise SystemExit("[ERR] 사용 가능한 디스크립터가 없습니다. ORB/SIFT/AKAZE/BRISK 중 하나 이상 활성화하세요.")
    return packs

def try_find_homography(pts1: np.ndarray, pts2: np.ndarray, ransac_reproj_th: float):
    """
    Tries USAC if available, otherwise RANSAC.
    Returns (H, mask) or (None, None)
    """
    method = None
    # Prefer USAC if present
    # Some builds expose USAC methods under cv2.USAC_* constants.
    if hasattr(cv2, "USAC_ACCURATE"):
        try:
            H, mask = cv2.findHomography(pts1, pts2, cv2.USAC_ACCURATE, ransac_reproj_th)
            if H is not None and mask is not None:
                return H, mask
        except Exception:
            pass
    # Fallback: RANSAC
    try:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_reproj_th)
        return H, mask
    except Exception:
        return None, None

def kp_and_desc(det: DetPack, img_gray: np.ndarray):
    try:
        kps, des = det.detector.detectAndCompute(img_gray, None)
        return kps or [], des
    except Exception:
        return [], None

def symmetric_knn_filter(matches_fwd, matches_bwd):
    """
    Build a lookup of symmetric (mutual) pairs based on best match indices.
    """
    def first_dm(entry):
        # entry: list/tuple of DMatch 또는 DMatch 단일일 가능성
        x = entry[0]
        if isinstance(x, (list, tuple)):  # tuple[DMatch,...] 방어
            x = x[0]
        return x  # DMatch

    best_fwd, best_bwd = {}, {}
    for m in matches_fwd:
        if len(m) >= 1:
            dm = first_dm(m)
            best_fwd[(dm.queryIdx, dm.trainIdx)] = True
    for m in matches_bwd:
        if len(m) >= 1:
            dm = first_dm(m)
            best_bwd[(dm.queryIdx, dm.trainIdx)] = True

    keep = {}
    for (qi, ti) in best_fwd.keys():
        if (ti, qi) in best_bwd:
            keep[(qi, ti)] = True
    return keep

def evaluate_pair(det_packs: List[DetPack],
                  ref_img: np.ndarray, ex_img: np.ndarray,
                  ratio_th: float, mutual_check: bool,
                  ransac_th: float,
                  do_photometric: bool,
                  ssim_th: float, ncc_th: float) -> Tuple[int, float, float, float, str]:
    """
    Returns:
        inliers, inlier_ratio, ssim, ncc, best_detector_name
    """
    best = dict(score=-1.0, inliers=0, inlier_ratio=0.0, H=None, det="NA")
    for pack in det_packs:
        kps1, des1 = kp_and_desc(pack, ref_img)
        kps2, des2 = kp_and_desc(pack, ex_img)
        if des1 is None or des2 is None or len(kps1) < 4 or len(kps2) < 4:
            continue

        bf = cv2.BFMatcher(pack.norm, crossCheck=False)
        try:
            m12 = bf.knnMatch(des1, des2, k=2)
            if mutual_check:
                m21 = bf.knnMatch(des2, des1, k=2)
        except Exception:
            continue

        good = []
        for m in m12:
            if len(m) < 2: 
                continue
            if m[0].distance < ratio_th * m[1].distance:
                good.append(m[0])

        if mutual_check:
            keep = symmetric_knn_filter(m12, m21)
            good = [g for g in good if ((g.queryIdx, g.trainIdx) in keep)]

        if len(good) < 4:
            continue

        pts1 = np.float32([kps1[g.queryIdx].pt for g in good]).reshape(-1,1,2)
        pts2 = np.float32([kps2[g.trainIdx].pt for g in good]).reshape(-1,1,2)
        H, mask = try_find_homography(pts1, pts2, ransac_th)
        if H is None or mask is None:
            continue
        inliers = int(mask.ravel().sum())
        if inliers <= 0:
            continue
        inlier_ratio = float(inliers) / float(len(good))
        score = inliers + 5.0 * inlier_ratio  # alpha=5.0 (legacy-compatible)

        if score > best["score"]:
            best.update(score=score, inliers=inliers, inlier_ratio=inlier_ratio, H=H, det=pack.name)

    if best["score"] < 0:
        return 0, 0.0, 0.0, 0.0, "NA"

    ssim = 0.0
    ncc = 0.0
    if do_photometric and best["H"] is not None:
        try:
            h, w = ex_img.shape[:2]
            warped = cv2.warpPerspective(ref_img, best["H"], (w, h), flags=cv2.INTER_LINEAR)
            # valid mask: where warp filled (not zeros everywhere)
            mask = (warped > 0).astype(np.uint8)
            if mask.sum() < 64:
                mask = None
            ssim = compute_ssim(warped, ex_img, mask)
            ncc = compute_ncc(warped, ex_img, mask)
        except Exception:
            ssim = 0.0
            ncc = 0.0

    return int(best["inliers"]), float(best["inlier_ratio"]), float(ssim), float(ncc), str(best["det"])

# =========================
# ---------- CLI ----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="Match candidate pairs with feature+RANSAC(+SSIM/NCC) and produce mapping_result.json & pair_scores.csv")
    ap.add_argument("--candidates", required=True, help="candidates.json (from 03_candidates.py)")
    ap.add_argument("--mapping", default="preprocess_mapping.json", help="preprocess_mapping.json (from 02_preprocess.py)")
    ap.add_argument("--scores", default="map_dist/pair_scores.csv", help="Output CSV for all scored pairs")
    ap.add_argument("--output", default="map_dist/mapping_result.json", help="Output JSON for final mapping")

    # Detectors
    ap.add_argument("--use-orb", action="store_true", default=True)
    ap.add_argument("--use-sift", action="store_true", default=False)
    ap.add_argument("--use-akaze", action="store_true", default=False)
    ap.add_argument("--use-brisk", action="store_true", default=False)
    ap.add_argument("--orb-nfeatures", type=int, default=1000)

    # Matching
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold")
    ap.add_argument("--mutual-check", action="store_true", help="Enable symmetric mutual check")

    # Geometry (RANSAC/USAC)
    ap.add_argument("--ransac-th", type=float, default=5.0, help="RANSAC reprojection threshold (pixels)")

    # Acceptance thresholds
    ap.add_argument("--min-inliers", type=int, default=12, help="Phase-A minimum inliers")
    ap.add_argument("--min-inlier-ratio", type=float, default=0.22, help="Phase-A minimum inlier ratio")
    ap.add_argument("--enable-recall-sweep", action="store_true", help="Enable Phase-B relaxed acceptance")
    ap.add_argument("--min-inliers-b", type=int, default=8, help="Phase-B minimum inliers")
    ap.add_argument("--min-inlier-ratio-b", type=float, default=0.15, help="Phase-B minimum inlier ratio")

    ap.add_argument("--alpha", type=float, default=5.0, help="Score weight: score = inliers + alpha * inlier_ratio")
    ap.add_argument("--top2-margin", type=float, default=4.0, help="Accept only if (top1 - top2) >= margin")
    ap.add_argument("--top2-multiplier", type=float, default=1.25, help="Accept only if top1 >= top2 * multiplier")

    # Photometric verification
    ap.add_argument("--verify-photometric", action="store_true", help="Enable SSIM/NCC photometric gate")
    ap.add_argument("--ssim-th", type=float, default=0.80)
    ap.add_argument("--ncc-th", type=float, default=0.65)

    # Assignment
    ap.add_argument("--assign", choices=["greedy", "hungarian"], default="greedy")

    return ap.parse_args()

# =========================
# --------- MAIN ----------
# =========================

def main() -> int:
    args = parse_args()

    # Load candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        C = json.load(f)
    cand_map = C["candidates"]  # { "low|ref_rel": [ {extracted, phash_dist, hist_dist}, ... ] }

    # Load mapping (for image paths)
    extracted_paths, reference_paths, mp_raw = load_mapping(Path(args.mapping))

    # Prepare detectors
    det_packs = make_detectors(args.use_orb, args.use_sift, args.use_akaze, args.use_brisk, args.orb_nfeatures)

    # Descriptor/image caches (in-memory)
    gray_cache_ref: Dict[str, np.ndarray] = {}
    gray_cache_ex: Dict[str, np.ndarray] = {}

    def get_gray_ref(ref_rel: str) -> Optional[np.ndarray]:
        if ref_rel in gray_cache_ref:
            return gray_cache_ref[ref_rel]
        vp = reference_paths.get(ref_rel)
        if not vp or not vp.gray_low:
            return None
        img = imread_unicode(vp.gray_low, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            gray_cache_ref[ref_rel] = img
        return img

    def get_gray_ex(ex_rel: str) -> Optional[np.ndarray]:
        if ex_rel in gray_cache_ex:
            return gray_cache_ex[ex_rel]
        vp = extracted_paths.get(ex_rel)
        if not vp or not vp.gray_low:
            return None
        img = imread_unicode(vp.gray_low, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            gray_cache_ex[ex_rel] = img
        return img

    # Prepare outputs
    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = scores_path.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    # Legacy-compatible header + extended fields
    csv_w.writerow([
        "ref", "ex", "score", "inliers", "inlier_ratio",
        "phash_dist", "hist_dist",
        "detector", "ssim", "ncc",
        "phase"  # "A"/"B"/""
    ])

    # Evaluate all candidate pairs per reference
    per_ref_best: Dict[str, Dict] = {}   # key="low|ref_rel" -> best cand/info
    total_pairs = 0

    for key_ref, cand_list in cand_map.items():
        # key_ref == "low|<ref_rel>"
        if not key_ref.startswith("low|"):
            # compatibility: still proceed
            ref_rel = key_ref.split("|", 1)[-1]
        else:
            ref_rel = key_ref[4:]

        ref_img = get_gray_ref(ref_rel)
        if ref_img is None:
            # write empty
            per_ref_best[key_ref] = None
            continue

        # Compute scores for each candidate ex
        scored = []
        for c in cand_list:
            ex_rel = c["extracted"]
            ex_img = get_gray_ex(ex_rel)
            if ex_img is None:
                continue

            inl, inl_ratio, ssim, ncc, detname = evaluate_pair(
                det_packs, ref_img, ex_img,
                ratio_th=args.ratio, mutual_check=args.mutual_check,
                ransac_th=args.ransac_th,
                do_photometric=args.verify_photometric,
                ssim_th=args.ssim_th, ncc_th=args.ncc_th
            )
            score = inl + args.alpha * inl_ratio
            scored.append({
                "ex_rel": ex_rel,
                "score": float(score),
                "inliers": int(inl),
                "inlier_ratio": float(inl_ratio),
                "phash_dist": int(c.get("phash_dist", 0)),
                "hist_dist": float(c.get("hist_dist", 1.0)),
                "detector": detname,
                "ssim": float(ssim),
                "ncc": float(ncc),
            })

        # Write CSV rows (also need top2 info -> compute after sorting)
        scored.sort(key=lambda d: d["score"], reverse=True)
        total_pairs += len(scored)

        # Top-2 gap metrics
        top2_gap = 0.0
        top2_mult = 9999.0
        if len(scored) >= 2:
            top2_gap = scored[0]["score"] - scored[1]["score"]
            top2_mult = (scored[0]["score"] / max(1e-6, scored[1]["score"])) if scored[1]["score"] > 0 else 9999.0
        elif len(scored) == 1:
            top2_gap = scored[0]["score"]
            top2_mult = 9999.0

        # Determine acceptance (A/B) for top1 only (conservative)
        accepted_phase = ""
        chosen = None
        if scored:
            s0 = scored[0]
            geom_A = (s0["inliers"] >= args.min_inliers and s0["inlier_ratio"] >= args.min_inlier_ratio)
            geom_B = (s0["inliers"] >= args.min_inliers_b and s0["inlier_ratio"] >= args.min_inlier_ratio_b) if args.enable_recall_sweep else False
            gap_ok = (top2_gap >= args.top2_margin) and (top2_mult >= args.top2_multiplier)
            photo_ok = True
            if args.verify_photometric:
                photo_ok = (s0["ssim"] >= args.ssim_th) or (s0["ncc"] >= args.ncc_th)

            if geom_A and gap_ok and photo_ok:
                accepted_phase = "A"
                chosen = s0
            elif geom_B and gap_ok and photo_ok:
                accepted_phase = "B"
                chosen = s0

        # Write rows with phase tag
        for idx, r in enumerate(scored):
            ph = accepted_phase if (idx == 0 and accepted_phase) else ""
            csv_w.writerow([
                f"{key_ref}", r["ex_rel"], f"{r['score']:.6f}",
                r["inliers"], f"{r['inlier_ratio']:.6f}",
                r["phash_dist"], f"{r['hist_dist']:.6f}",
                r["detector"], f"{r['ssim']:.6f}", f"{r['ncc']:.6f}",
                ph
            ])

        per_ref_best[key_ref] = {
            "top": scored[0] if scored else None,
            "accepted_phase": accepted_phase,
            "top2_gap": float(top2_gap),
            "top2_mult": float(top2_mult)
        }

    csv_f.close()

    # 1:1 greedy assignment (by score desc over accepted refs)
    # If two refs want the same ex, higher score wins.
    accepted = []
    for key_ref, info in per_ref_best.items():
        if not info or not info["accepted_phase"] or not info["top"]:
            continue
        top = info["top"]
        accepted.append({
            "ref": key_ref,
            "ex": top["ex_rel"],
            "score": top["score"],
            "inliers": top["inliers"],
            "inlier_ratio": top["inlier_ratio"],
            "detector": top["detector"],
            "ssim": top["ssim"],
            "ncc": top["ncc"],
            "phase": info["accepted_phase"],
            "top2_gap": info["top2_gap"],
            "top2_mult": info["top2_mult"],
        })

    accepted.sort(key=lambda d: d["score"], reverse=True)

    assigned: Dict[str, str] = {}  # ref -> ex
    used_ex: Dict[str, bool] = {}
    for a in accepted:
        ref = a["ref"]
        ex = a["ex"]
        if ref in assigned:
            continue
        if used_ex.get(ex, False):
            continue
        assigned[ref] = ex
        used_ex[ex] = True

    # Build mapping_result.json
    all_refs = list(cand_map.keys())
    unmatched_ref = [r for r in all_refs if r not in assigned]

    out_obj = {
        "version": 2,
        "basis": "reference",
        "track": "low",
        "parameters": {
            "detectors": [p.name for p in det_packs],
            "ratio": args.ratio,
            "mutual_check": bool(args.mutual_check),
            "ransac_th": args.ransac_th,
            "min_inliers_A": args.min_inliers,
            "min_inlier_ratio_A": args.min_inlier_ratio,
            "enable_recall_sweep": bool(args.enable_recall_sweep),
            "min_inliers_B": args.min_inliers_b,
            "min_inlier_ratio_B": args.min_inlier_ratio_b,
            "alpha": args.alpha,
            "top2_margin": args.top2_margin,
            "top2_multiplier": args.top2_multiplier,
            "verify_photometric": bool(args.verify_photometric),
            "ssim_th": args.ssim_th,
            "ncc_th": args.ncc_th,
            "assign": args.assign,
        },
        "stats": {
            "references": len(all_refs),
            "assigned": len(assigned),
            "coverage": (float(len(assigned)) / float(max(1, len(all_refs)))),
            "total_scored_pairs": int(total_pairs),
        },
        # legacy-friendly mapping (dict)
        "mapping": {ref: ex for ref, ex in assigned.items()},
        # extended detail list
        "assignments": [
            {
                "reference": ref,
                "extracted": ex,
                "score": float(next(a["score"] for a in accepted if a["ref"] == ref and a["ex"] == ex)),
                "phase": next(a["phase"] for a in accepted if a["ref"] == ref and a["ex"] == ex),
            }
            for ref, ex in assigned.items()
        ],
        "unmatched_reference": unmatched_ref,
        "scores_csv": str(scores_path.as_posix())
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[INFO] pair_scores 저장: {scores_path}")
    print(f"[INFO] mapping_result 저장: {out_path}")
    print(f"[INFO] coverage: {out_obj['stats']['coverage']:.3f} ({out_obj['stats']['assigned']}/{out_obj['stats']['references']})")
    if args.assign == "hungarian":
        print("[WARN] 'hungarian' 지정됨: SciPy 미사용으로 greedy로 대체했습니다.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
