"""
04_match.py  (rev. feedback-loop ready)

Inputs
------
- candidates.json (from 03)
- preprocess_mapping.json (from 02) for variant file paths
- (optional) feedback.json (from 05) for per-ref overrides

Pipeline
--------
For each reference (basis=reference):
  For each candidate extracted:
    1) detect features (ORB/SIFT/…)
    2) match with ratio test (+ optional mutual check)
    3) estimate H with USAC/RANSAC -> inliers, inlier_ratio, reproj_rmse
    4) photometric verify (optional):
         - warp ex->ref with H, then SSIM/NCC
         - optional ECC refinement then re-SSIM/NCC
         - optional G-SSIM (gradient-SSIM)
    5) final candidate score = inliers + 5 * inlier_ratio
  Pick the best/second-best -> apply top-2 margin/multiplier rule
  Accept by Phase A thresholds; if --enable-recall-sweep, fallback to Phase B

Assignment
----------
--assign greedy : 1:1 greedy by descending score (skip taken extracted)
--assign reuse  : 1:N allowed (no global uniqueness constraint)

Outputs
-------
1) pair_scores.csv : all (ref,ex) with geometry/photometric/phase/score
2) mapping_result.json :
   {
     "version": 2,
     "assign_mode": "greedy|reuse",
     "scores_csv": "<path>",
     "stats": {...},
     "mapping": {"<ref_rel>": "<ex_rel>", ...},
     "assignments": [
       {"ref": "<ref_rel>", "ex": "<ex_rel>", "score": <float>, "phase": "A|B",
        "inliers": <int>, "inlier_ratio": <float>, "ssim": <float>, "ncc": <float>,
        "top2_margin": <float>, "top2_multiplier": <float>, "suspect": <bool>}
     ],
     "unassigned": ["<ref_rel>", ...],
     "suspects": ["<ref_rel>", ...]
   }

CLI Example
-----------
python 04_match.py \
  --candidates artifacts/candidates.json \
  --mapping artifacts/preprocess_mapping.json \
  --scores artifacts/pair_scores.csv \
  --output artifacts/mapping_result.json \
  --use-orb --use-sift --ratio 0.75 --mutual-check \
  --ransac-th 5.0 \
  --min-inliers 12 --min-inlier-ratio 0.22 \
  --enable-recall-sweep --min-inliers-b 8 --min-inlier-ratio-b 0.15 \
  --verify-photometric --ssim-th 0.80 --ncc-th 0.65 \
  --enable-ecc --use-gssim \
  --assign greedy \
  --feedback-in artifacts/feedback.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERR] OpenCV(cv2) import failed: {e}\nInstall: pip install opencv-python")

# =========================
# ---------- I/O ----------
# =========================

def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, flags)
    except Exception:
        return None

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# =========================
# --- Load mapping/cands ---
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
        out = {}
        for src_rel, info in mp["by_src"][cat].items():
            v = info["variants"]
            def _get(track: str, ch: str) -> Optional[Path]:
                p = v.get(track, {}).get(ch)
                return Path(p) if p else None
            out[src_rel] = VariantPaths(
                color_low=_get("low", "color"),
                gray_low=_get("low", "gray"),
                color_high=_get("high", "color"),
                gray_high=_get("high", "gray"),
            )
        return out

    return _collect("extracted"), _collect("reference"), mp

def load_candidates(cands_json: Path) -> Dict[str, List[dict]]:
    obj = json.load(open(cands_json, "r", encoding="utf-8"))
    if obj.get("basis") != "reference":
        raise SystemExit("[ERR] candidates.json basis must be 'reference'")
    cands = obj.get("candidates")
    if not isinstance(cands, dict):
        raise SystemExit("[ERR] invalid candidates.json")
    return cands

# =========================
# ------- Feedback --------
# =========================

@dataclass
class RefOverride:
    top2_margin: Optional[float] = None
    top2_multiplier: Optional[float] = None
    ssim_th: Optional[float] = None
    ncc_th: Optional[float] = None
    min_inliers: Optional[int] = None
    min_inlier_ratio: Optional[float] = None

@dataclass
class FeedbackCfg:
    per_ref: Dict[str, RefOverride]
    global_overrides: RefOverride
    reuse_enabled: Optional[bool] = None   # hint only; actual mode is --assign

def _norm_ref_key(s: str) -> str:
    s = (s or "").strip().replace("\\", "/")
    if "|" in s:
        pref, rest = s.split("|", 1)
        if pref in ("low", "high"):
            return rest
    return s

def load_feedback(path: Optional[Path]) -> FeedbackCfg:
    per_ref: Dict[str, RefOverride] = {}
    glob = RefOverride()
    reuse_enabled = None
    if not path or not path.exists():
        return FeedbackCfg(per_ref, glob, reuse_enabled)

    try:
        fb = json.load(open(path, "r", encoding="utf-8"))
        actions = fb.get("actions", {})

        # tighten_verification
        tv = actions.get("tighten_verification", {}) or {}
        def _upd(obj: RefOverride, src: dict):
            if "top2_margin" in src:
                try: obj.top2_margin = float(src["top2_margin"])
                except Exception: pass
            if "top2_multiplier" in src:
                try: obj.top2_multiplier = float(src["top2_multiplier"])
                except Exception: pass
            if "ssim_th" in src:
                try: obj.ssim_th = float(src["ssim_th"])
                except Exception: pass
            if "ncc_th" in src:
                try: obj.ncc_th = float(src["ncc_th"])
                except Exception: pass
            if "min_inliers" in src:
                try: obj.min_inliers = int(src["min_inliers"])
                except Exception: pass
            if "min_inlier_ratio" in src:
                try: obj.min_inlier_ratio = float(src["min_inlier_ratio"])
                except Exception: pass

        _upd(glob, tv)
        for item in tv.get("refs", []) or []:
            # allow both string (ref) and dict {ref: "...", ...}
            if isinstance(item, str):
                per_ref[_norm_ref_key(item)] = RefOverride(
                    top2_margin=glob.top2_margin, top2_multiplier=glob.top2_multiplier,
                    ssim_th=glob.ssim_th, ncc_th=glob.ncc_th,
                    min_inliers=glob.min_inliers, min_inlier_ratio=glob.min_inlier_ratio
                )
            elif isinstance(item, dict):
                ref = _norm_ref_key(item.get("ref", ""))
                if not ref: continue
                r = RefOverride()
                _upd(r, item)
                per_ref[ref] = r

        # allow_reuse hint
        ar = actions.get("allow_reuse", {}) or {}
        if "enabled" in ar:
            reuse_enabled = bool(ar.get("enabled"))

    except Exception as e:
        print(f"[WARN] feedback parse failed: {e}")

    return FeedbackCfg(per_ref, glob, reuse_enabled)

# =========================
# ---- Feature/Matching ---
# =========================

def make_detector(use_sift: bool, use_orb: bool):
    dets = []
    if use_sift:
        sift = None
        try:
            sift = cv2.SIFT_create()
        except Exception:
            pass
        if sift is not None:
            dets.append(("SIFT", sift, cv2.NORM_L2))
    if use_orb:
        try:
            orb = cv2.ORB_create(nfeatures=4000, fastThreshold=7)
            dets.append(("ORB", orb, cv2.NORM_HAMMING))
        except Exception:
            pass
    if not dets:
        raise SystemExit("[ERR] No detectors available. Use --use-orb or install xfeatures2d for SIFT.")
    return dets

def knn_ratio_matches(desc1, desc2, norm_type, ratio=0.75, mutual_check=False):
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    m12 = matcher.knnMatch(desc1, desc2, k=2)
    good12 = []
    for m, n in m12:
        if m.distance < ratio * n.distance:
            good12.append(m)

    if not mutual_check:
        return good12

    # mutual NN check
    m21 = matcher.knnMatch(desc2, desc1, k=2)
    nn21 = {}
    for m, n in m21:
        if m.distance < ratio * n.distance:
            nn21[m.queryIdx] = m.trainIdx  # desc2 idx -> desc1 idx

    mutual = []
    for m in good12:
        # m.queryIdx in desc1, m.trainIdx in desc2
        if nn21.get(m.trainIdx, -1) == m.queryIdx:
            mutual.append(m)
    return mutual

def estimate_homography(kp1, kp2, matches, use_usac: bool, ransac_th: float):
    if len(matches) < 4:
        return None, [], float("inf")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    method = 0
    if use_usac and hasattr(cv2, "USAC_ACCURATE"):
        method = cv2.USAC_ACCURATE
        H, mask = cv2.findHomography(pts1, pts2, method, ransac_th, maxIters=10000, confidence=0.999)
    else:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_th, maxIters=5000, confidence=0.999)
    if H is None or mask is None:
        return None, [], float("inf")
    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, ok in zip(matches, mask) if ok]
    # reprojection RMSE on inliers
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    pts1_proj = cv2.perspectiveTransform(pts1_in, H)
    err = np.sqrt(np.mean(np.sum((pts1_proj - pts2_in) ** 2, axis=2)))
    return H, inlier_matches, float(err)

# =========================
# ---- Photometric utils ---
# =========================

def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    SSIM for single-channel uint8 arrays of identical shape.
    """
    if x.shape != y.shape:
        return 0.0
    if x.ndim != 2:
        return 0.0
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # constants for L=255
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))

def compute_gssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Gradient-SSIM: apply Sobel magnitude then SSIM.
    """
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    gm_x = cv2.magnitude(gx, gy)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    gm_y = cv2.magnitude(gx, gy)
    gm_x = np.uint8(np.clip(gm_x, 0, 255))
    gm_y = np.uint8(np.clip(gm_y, 0, 255))
    return compute_ssim(gm_x, gm_y)

def compute_ncc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Zero-mean normalized cross correlation on single-channel.
    """
    if x.shape != y.shape or x.ndim != 2:
        return 0.0
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = x - x.mean()
    y = y - y.mean()
    num = float((x * y).sum())
    den = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))

def warp_ex_to_ref(ref_gray: np.ndarray, ex_gray: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    try:
        h, w = ref_gray.shape[:2]
        warp = cv2.warpPerspective(ex_gray, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warp
    except Exception:
        return None

def ecc_refine(ref_gray: np.ndarray, warp_init: np.ndarray) -> Optional[np.ndarray]:
    """
    ECC (affine) refinement on gray images. Returns refined warp result.
    """
    try:
        ref_f = ref_gray.astype(np.float32) / 255.0
        mov_f = warp_init.astype(np.float32) / 255.0
        # Start from identity affine
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        try:
            cc, warp = cv2.findTransformECC(ref_f, mov_f, warp,
                                            motionType=cv2.MOTION_AFFINE,
                                            criteria=criteria, inputMask=None, gaussFiltSize=3)
        except cv2.error:
            return warp_init
        h, w = ref_gray.shape[:2]
        refined = cv2.warpAffine(mov_f, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        refined = np.uint8(np.clip(refined * 255.0, 0, 255))
        return refined
    except Exception:
        return warp_init

# =========================
# ------- Scoring ---------
# =========================

@dataclass
class CandScore:
    ref: str
    ex: str
    detector: str
    matches: int
    inliers: int
    inlier_ratio: float
    reproj_rmse: float
    score: float
    ssim: float
    ncc: float
    gssim: float
    phase: str
    passed_geom: bool
    passed_photo: bool
    top2_margin: float
    top2_multiplier: float
    ecc_used: bool
    reason: str

def evaluate_candidate(ref_rel: str,
                       ex_rel: str,
                       ref_paths: VariantPaths,
                       ex_paths: VariantPaths,
                       detectors,
                       ratio: float,
                       mutual_check: bool,
                       use_usac: bool,
                       ransac_th: float,
                       verify_photo: bool,
                       ssim_th: float,
                       ncc_th: float,
                       use_ecc: bool,
                       use_gssim: bool) -> Optional[CandScore]:
    # load low/gray
    p_ref = ref_paths.gray_low or ref_paths.color_low
    p_ex  = ex_paths.gray_low or ex_paths.color_low
    if not p_ref or not p_ref.exists() or not p_ex or not p_ex.exists():
        return None

    ref_gray = imread_unicode(p_ref, cv2.IMREAD_GRAYSCALE)
    ex_gray  = imread_unicode(p_ex,  cv2.IMREAD_GRAYSCALE)
    if ref_gray is None or ex_gray is None:
        return None

    best = None
    best_H = None
    best_det_name = "NA"
    for det_name, det, norm in detectors:
        try:
            kp1, desc1 = det.detectAndCompute(ref_gray, None)
            kp2, desc2 = det.detectAndCompute(ex_gray, None)
        except Exception:
            continue

        good = knn_ratio_matches(desc1, desc2, norm, ratio=ratio, mutual_check=mutual_check)
        H, inliers, rmse = estimate_homography(kp1, kp2, good, use_usac, ransac_th)
        if H is None:
            continue
        inlier_ratio = (len(inliers) / max(1, len(good)))
        score = float(len(inliers) + 5.0 * inlier_ratio)
        if (best is None) or (score > best[0]):
            best = (score, len(good), len(inliers), inlier_ratio, rmse, det_name, H)
            best_det_name = det_name
            best_H = H

    if best is None or best_H is None:
        return None

    score_val, n_matches, n_inl, inl_ratio, rmse, det_used, H = best

    # photometric verification
    ssim = ncc = gssim = 0.0
    ecc_used = False
    passed_photo = True
    if verify_photo:
        w = warp_ex_to_ref(ref_gray, ex_gray, H)
        if w is None:
            w = ex_gray
        ssim = compute_ssim(ref_gray, w)
        ncc  = compute_ncc(ref_gray, w)
        if use_gssim:
            gssim = compute_gssim(ref_gray, w)
        # ECC refinement for borderline
        if use_ecc and (ssim < max(0.9, ssim_th) or ncc < max(0.9, ncc_th)):
            w2 = ecc_refine(ref_gray, w)
            if w2 is not None:
                ecc_used = True
                ssim = max(ssim, compute_ssim(ref_gray, w2))
                ncc  = max(ncc,  compute_ncc(ref_gray, w2))
                if use_gssim:
                    gssim = max(gssim, compute_gssim(ref_gray, w2))
        passed_photo = (ssim >= ssim_th) or (ncc >= ncc_th)

    return CandScore(
        ref=ref_rel, ex=ex_rel, detector=best_det_name,
        matches=int(n_matches), inliers=int(n_inl), inlier_ratio=float(inl_ratio),
        reproj_rmse=float(rmse), score=float(score_val),
        ssim=float(ssim), ncc=float(ncc), gssim=float(gssim),
        phase="?", passed_geom=True, passed_photo=bool(passed_photo),
        top2_margin=0.0, top2_multiplier=0.0, ecc_used=bool(ecc_used),
        reason=""
    )

# =========================
# --------- CLI ----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="Score candidates and produce final mapping (greedy 1:1 or reuse 1:N).")

    ap.add_argument("--candidates", required=True, help="candidates.json from 03")
    ap.add_argument("--mapping", required=True, help="preprocess_mapping.json from 02")
    ap.add_argument("--scores", required=True, help="Output pair_scores.csv")
    ap.add_argument("--output", required=True, help="Output mapping_result.json")

    # detectors / matching
    ap.add_argument("--use-orb", action="store_true")
    ap.add_argument("--use-sift", action="store_true")
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    ap.add_argument("--mutual-check", action="store_true", help="Mutual nearest filter on top of ratio test")
    ap.add_argument("--use-usac", action="store_true", help="Use USAC if available; else RANSAC")
    ap.add_argument("--ransac-th", type=float, default=5.0)

    # acceptance thresholds
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--min-inlier-ratio", type=float, default=0.22)
    ap.add_argument("--enable-recall-sweep", action="store_true")
    ap.add_argument("--min-inliers-b", type=int, default=8)
    ap.add_argument("--min-inlier-ratio-b", type=float, default=0.15)

    # top-2 uniqueness (per ref)
    ap.add_argument("--top2-margin", type=float, default=4.0)
    ap.add_argument("--top2-multiplier", type=float, default=1.25)

    # photometric
    ap.add_argument("--verify-photometric", action="store_true")
    ap.add_argument("--ssim-th", type=float, default=0.80)
    ap.add_argument("--ncc-th", type=float, default=0.65)
    ap.add_argument("--enable-ecc", action="store_true")
    ap.add_argument("--use-gssim", action="store_true")

    # assignment mode
    ap.add_argument("--assign", choices=["greedy", "reuse"], default="greedy")

    # feedback overrides
    ap.add_argument("--feedback-in", default=None)

    # misc
    ap.add_argument("--max-cands-per-ref", type=int, default=64)

    return ap.parse_args()

# =========================
# --------- MAIN ---------
# =========================

def main() -> int:
    args = parse_args()
    if not (args.use_orb or args.use_sift):
        print("[WARN] No detector flag given; defaulting to --use-orb")
        args.use_orb = True

    cands = load_candidates(Path(args.candidates))
    ex_paths, ref_paths, _mp_raw = load_mapping(Path(args.mapping))
    fb = load_feedback(Path(args.feedback_in) if args.feedback_in else None)

    # detectors
    detectors = make_detector(use_sift=args.use_sift, use_orb=args.use_orb)

    # containers
    all_scores: List[CandScore] = []
    per_ref_scores: Dict[str, List[CandScore]] = {}

    # iterate refs
    keys = list(cands.keys())  # keys like "low|<ref_rel>"
    for i, key in enumerate(keys, 1):
        if not key.startswith("low|"):
            # accept both forms; map to ref_rel
            ref_rel = _norm_ref_key(key)
        else:
            ref_rel = key.split("|", 1)[1]
        ref_vp = ref_paths.get(ref_rel)
        if ref_vp is None:
            per_ref_scores[key] = []
            continue

        cand_list = cands.get(key, [])[: max(1, int(args.max_cands_per_ref))]
        tmp_scores: List[CandScore] = []
        for c in cand_list:
            ex_rel = c.get("extracted")
            if not ex_rel:
                continue
            ex_vp = ex_paths.get(ex_rel)
            if ex_vp is None:
                continue
            sc = evaluate_candidate(
                ref_rel, ex_rel, ref_vp, ex_vp,
                detectors=detectors,
                ratio=args.ratio, mutual_check=args.mutual_check,
                use_usac=args.use_usac, ransac_th=args.ransac_th,
                verify_photo=args.verify_photometric,
                ssim_th=args.ssim_th, ncc_th=args.ncc_th,
                use_ecc=args.enable_ecc, use_gssim=args.use_gssim
            )
            if sc is not None:
                tmp_scores.append(sc)

        # sort by score desc
        tmp_scores.sort(key=lambda s: s.score, reverse=True)
        per_ref_scores[key] = tmp_scores
        all_scores.extend(tmp_scores)

        if i % 25 == 0 or i == len(keys):
            print(f"[INFO] Scoring progress: {i}/{len(keys)}")

    # determine acceptance per ref (Phase A/B + top-2 rule + overrides)
    acceptables: Dict[str, List[CandScore]] = {}
    suspects: Set[str] = set()
    for key, lst in per_ref_scores.items():
        ref_rel = key.split("|", 1)[1] if "|" in key else _norm_ref_key(key)
        over = fb.per_ref.get(ref_rel)
        min_inl_A  = over.min_inliers      if over and over.min_inliers      is not None else args.min_inliers
        min_rat_A  = over.min_inlier_ratio if over and over.min_inlier_ratio is not None else args.min_inlier_ratio
        ssim_A     = over.ssim_th          if over and over.ssim_th          is not None else args.ssim_th
        ncc_A      = over.ncc_th           if over and over.ncc_th           is not None else args.ncc_th
        t2marg     = over.top2_margin      if over and over.top2_margin      is not None else args.top2_margin
        t2mult     = over.top2_multiplier  if over and over.top2_multiplier  is not None else args.top2_multiplier

        if not lst:
            acceptables[key] = []
            continue

        # mark phase and filter by thresholds
        phaseA: List[CandScore] = []
        phaseB: List[CandScore] = []
        for sc in lst:
            passed_geom = (sc.inliers >= min_inl_A) and (sc.inlier_ratio >= min_rat_A)
            passed_photo = (not args.verify_photometric) or ((sc.ssim >= ssim_A) or (sc.ncc >= ncc_A))
            sc.passed_geom = bool(passed_geom)
            sc.passed_photo = bool(passed_photo)

            if passed_geom and passed_photo:
                sc.phase = "A"
                phaseA.append(sc)
            elif args.enable_recall_sweep:
                # Phase B: slightly relaxed
                passed_geom_b = (sc.inliers >= args.min_inliers_b) and (sc.inlier_ratio >= args.min_inlier_ratio_b)
                passed_photo_b = (not args.verify_photometric) or ((sc.ssim >= min(0.9, ssim_A)) or (sc.ncc >= min(0.9, ncc_A)))
                if passed_geom_b and passed_photo_b:
                    sc.phase = "B"
                    phaseB.append(sc)

        candidates_ok = phaseA if phaseA else phaseB

        # top-2 rule (on candidates_ok list; if only 1 exists, keep it)
        if candidates_ok:
            candidates_ok.sort(key=lambda s: s.score, reverse=True)
            top1 = candidates_ok[0]
            top2 = candidates_ok[1] if len(candidates_ok) >= 2 else None
            margin = float(top1.score - (top2.score if top2 else 0.0))
            mult = float(top1.score / max(1e-6, (top2.score if top2 else 1e-6)))
            top1.top2_margin = margin
            top1.top2_multiplier = mult
            # suspect if separation is weak (will still be considered in assignment but flagged)
            suspect = (margin < max(0.0, t2marg)) or (mult < max(1.0, t2mult))
            if suspect:
                suspects.add(ref_rel)
            # retain ordered list for assignment stage
            acceptables[key] = candidates_ok
        else:
            acceptables[key] = []

    # assignment
    mapping: Dict[str, str] = {}
    assignments: List[dict] = []
    taken_ex: Set[str] = set()

    if args.assign == "greedy":
        # build preference list: (best_score, key, idx_in_list)
        prefs: List[Tuple[float, str, int]] = []
        for key, lst in acceptables.items():
            if lst:
                prefs.append((lst[0].score, key, 0))
        prefs.sort(reverse=True, key=lambda x: x[0])

        for _score, key, _ in prefs:
            lst = acceptables[key]
            ref_rel = key.split("|", 1)[1] if "|" in key else _norm_ref_key(key)
            chosen = None
            for sc in lst:  # try next-best if top taken
                if sc.ex not in taken_ex:
                    chosen = sc
                    break
            if chosen is None:
                continue
            mapping[ref_rel] = chosen.ex
            taken_ex.add(chosen.ex)
            # suspect mark recorded from earlier computation
            sus = ref_rel in suspects
            assignments.append({
                "ref": ref_rel, "ex": chosen.ex, "score": chosen.score, "phase": chosen.phase,
                "inliers": chosen.inliers, "inlier_ratio": chosen.inlier_ratio,
                "reproj_rmse": chosen.reproj_rmse,
                "ssim": chosen.ssim, "ncc": chosen.ncc, "gssim": chosen.gssim,
                "top2_margin": chosen.top2_margin, "top2_multiplier": chosen.top2_multiplier,
                "detector": chosen.detector,
                "suspect": bool(sus)
            })
    else:  # reuse (1:N)
        for key, lst in acceptables.items():
            if not lst:
                continue
            ref_rel = key.split("|", 1)[1] if "|" in key else _norm_ref_key(key)
            chosen = lst[0]
            mapping[ref_rel] = chosen.ex
            sus = ref_rel in suspects
            assignments.append({
                "ref": ref_rel, "ex": chosen.ex, "score": chosen.score, "phase": chosen.phase,
                "inliers": chosen.inliers, "inlier_ratio": chosen.inlier_ratio,
                "reproj_rmse": chosen.reproj_rmse,
                "ssim": chosen.ssim, "ncc": chosen.ncc, "gssim": chosen.gssim,
                "top2_margin": chosen.top2_margin, "top2_multiplier": chosen.top2_multiplier,
                "detector": chosen.detector,
                "suspect": bool(sus)
            })

    assigned_refs = set(mapping.keys())
    all_refs = {key.split("|", 1)[1] if "|" in key else _norm_ref_key(key) for key in cands.keys()}
    unassigned = sorted(list(all_refs - assigned_refs))
    suspects_list = sorted(list(suspects & assigned_refs))

    # stats
    phaseA_cnt = sum(1 for a in assignments if a.get("phase") == "A")
    phaseB_cnt = sum(1 for a in assignments if a.get("phase") == "B")
    stats = {
        "references": len(all_refs),
        "assigned": len(assignments),
        "unassigned": len(unassigned),
        "phaseA": phaseA_cnt,
        "phaseB": phaseB_cnt,
        "assign_mode": args.assign
    }

    # write pair_scores.csv
    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ref", "ex", "detector",
            "matches", "inliers", "inlier_ratio", "reproj_rmse",
            "score", "phase",
            "ssim", "ncc", "gssim",
            "passed_geom", "passed_photo",
            "top2_margin", "top2_multiplier", "ecc_used", "reason"
        ])
        for key, lst in per_ref_scores.items():
            ref_rel = key.split("|", 1)[1] if "|" in key else _norm_ref_key(key)
            # we want to record all evaluated candidates (not only accepted)
            for sc in lst:
                w.writerow([
                    ref_rel, sc.ex, sc.detector,
                    sc.matches, sc.inliers, f"{sc.inlier_ratio:.6f}", f"{sc.reproj_rmse:.6f}",
                    f"{sc.score:.6f}", sc.phase,
                    f"{sc.ssim:.6f}", f"{sc.ncc:.6f}", f"{sc.gssim:.6f}",
                    int(sc.passed_geom), int(sc.passed_photo),
                    f"{sc.top2_margin:.6f}", f"{sc.top2_multiplier:.6f}",
                    int(sc.ecc_used), sc.reason or ""
                ])

    # write mapping_result.json
    out_obj = {
        "version": 2,
        "assign_mode": args.assign,
        "scores_csv": str(scores_path.resolve()),
        "stats": stats,
        "mapping": mapping,                     # ref_rel -> ex_rel
        "assignments": assignments,             # rich per-pair info
        "unassigned": unassigned,
        "suspects": suspects_list
    }
    save_json(Path(args.output), out_obj)
    print(f"[INFO] pair_scores saved: {scores_path}")
    print(f"[INFO] mapping_result saved: {args.output}")
    print(f"[INFO] assigned={stats['assigned']} / refs={stats['references']} (phaseA={phaseA_cnt}, phaseB={phaseB_cnt})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
