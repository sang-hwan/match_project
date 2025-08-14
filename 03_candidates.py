"""
03_candidates.py  (rev. feedback-loop ready)

Build candidate pairs (reference -> extracted) using:
- Stage-1: pHash(gray, low) within Hamming radius (BK-tree; progressive expansion)
- Stage-2: HSV histogram(color, low) with **correct** Bhattacharyya distance
- Optional cheap **prefilters** (orig_wh aspect ratio gap, edge density gap on low/gray)

New in this revision
--------------------
1) Feedback-driven overrides  (for 02→03→04→05 loop)
   - --feedback-in feedback.json
     * actions.expand_candidates.refs: [ "low|<ref_rel>" or "<ref_rel>" ]
     * actions.expand_candidates.phash_radius_delta: int (e.g., +4)
     * actions.expand_candidates.hist_threshold_new: float (e.g., 0.80)
     * actions.expand_candidates.min_cand_per_basis: int (e.g., 12)
     * (optional) actions.expand_candidates.per_ref: [
           {"ref": "low|<ref_rel>", "phash_radius_delta": 6, "hist_threshold": 0.85, "min_cand_per_basis": 15}, ...
       ]

   -> Listed references get **per-ref overrides** for pHash search radius/hist threshold/min candidates.

2) Partial run + merge
   - --only-refs-from-feedback : process only refs from feedback.expand_candidates.refs (plus redo_preprocess.refs)
   - --only-refs refs.txt      : process only refs listed in a text file (one src_relpath per line; accepts "low|...")

   If partial processing is requested and an old --out exists, load it and **merge**:
   - For processed refs, replace their candidate lists.
   - Keep others as-is. 04_match.py remains fully functional.

Output
------
candidates.json (compatible with 04_match.py)
{
  "version": 2,
  "basis": "reference",
  "track": "low",
  "features": {
    "phash_channel": "gray",
    "hist_channel": "color",
    "bins_h": 16, "bins_s": 8, "bins_v": 8
  },
  "thresholds": { "phash": 28, "hist": 0.70 },
  "min_cand_per_basis": 5,
  "prefilter": {
    "aspect_log_abs_max": 0.50,
    "edge_density_abs_diff_max": 0.30
  },
  "overrides": { "count": <int>, "from_feedback": true|false },
  "stats": {...},
  "candidates": {
    "low|<reference_src_relpath>": [
      {"extracted": "<extracted_src_relpath>", "phash_dist": <int>, "hist_dist": <float>},
      ...
    ],
    ...
  }
}

Usage
-----
python 03_candidates.py EXTRACTED_OUT REFERENCE_OUT \
  --mapping preprocess_mapping.json \
  --out candidates.json \
  --cache .cache/features \
  --workers 8 \
  --phash-threshold 28 --phash-step 2 --phash-max 36 \
  --hist-threshold 0.70 --min-cand-per-basis 5 \
  --prefilter-aspect 0.50 --prefilter-edge 0.30 \
  [--feedback-in artifacts/feedback.json --only-refs-from-feedback] \
  [--only-refs refs.txt] \
  [--verify]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

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
        img = cv2.imdecode(buf, flags)
        return img
    except Exception:
        return None

# =========================
# ----- Mapping loader ----
# =========================

@dataclass
class VariantPaths:
    """Absolute paths for a single source image (one src_relpath)."""
    color_low: Optional[Path]
    gray_low: Optional[Path]
    color_high: Optional[Path]
    gray_high: Optional[Path]

def load_mapping(mapping_json: Path) -> Tuple[Dict[str, VariantPaths], Dict[str, VariantPaths], Dict]:
    """
    Returns:
      (extracted_by_src, reference_by_src, raw_mapping_dict)
      where each value is VariantPaths with absolute Paths or None.
    """
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
# ----- Feature utils -----
# =========================

def hamming64(x: int, y: int) -> int:
    """Hamming distance over 64-bit integers."""
    z = (x ^ y) & ((1 << 64) - 1)
    return int(bin(z).count("1"))

def phash64(img_gray: np.ndarray) -> int:
    """
    64-bit perceptual hash (DCT-based).
    Steps:
      - resize -> 32x32
      - DCT -> take top-left 8x8 (excluding DC)
      - threshold by median
    """
    g = img_gray
    if g.ndim != 2:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(g)
    block = dct[:8, :8].copy()
    vals = block.flatten()[1:]  # exclude DC
    med = np.median(vals)
    bits = (vals > med).astype(np.uint8)
    # pack bits -> uint64
    out = 0
    for i, b in enumerate(bits):
        if b:
            out |= (1 << i)
    return int(out)

def hsv_histogram(img_bgr: np.ndarray, bins_h=16, bins_s=8, bins_v=8) -> np.ndarray:
    """3D HSV histogram (H×S×V), normalized to sum=1."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2],
                        None, [bins_h, bins_s, bins_v],
                        [0, 180, 0, 256, 0, 256]).astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist.flatten()

def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Correct Bhattacharyya distance on probability vectors.
    d = sqrt(max(0, 1 - sum_i sqrt(p_i) * sqrt(q_i)))
    Range 0 (similar) ... 1 (dissimilar)
    """
    # guard against negatives due to numerical noise
    coef = float(np.sum(np.sqrt(np.clip(p, 0, None)) * np.sqrt(np.clip(q, 0, None))))
    return float(math.sqrt(max(0.0, 1.0 - coef)))

# =========================
# --- feature caching  ----
# =========================

def load_cache_json(path: Path) -> Optional[Dict[str, object]]:
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_cache_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_feature_cache(paths: Dict[str, VariantPaths],
                        which: str,  # "extracted" | "reference"
                        track: str,  # "low"
                        ch_phash: str,  # "gray"
                        ch_hist: str,   # "color"
                        cache_dir: Optional[Path],
                        workers: int,
                        bins_h: int,
                        bins_s: int,
                        bins_v: int) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """
    Returns:
      phash_map: {src_relpath: uint64_int}
      hist_map:  {src_relpath: np.ndarray (H*S*V)}
    """
    # --- load caches if available ---
    cache_p_ph = None
    cache_p_hi = None
    if cache_dir:
        cache_p_ph = cache_dir / f"phash_{which}_{track}_{ch_phash}.json"
        cache_p_hi = cache_dir / f"hist_{which}_{track}_{ch_hist}_{bins_h}_{bins_s}_{bins_v}.json"

    phash_map: Dict[str, int] = {}
    hist_map: Dict[str, np.ndarray] = {}

    # try reading cache
    cache_ph = load_cache_json(cache_p_ph) if cache_p_ph else None
    if cache_ph and cache_ph.get("version") == 1:
        for k, v in cache_ph.get("data", {}).items():
            try:
                phash_map[k] = int(v)
            except Exception:
                pass

    cache_hi = load_cache_json(cache_p_hi) if cache_p_hi else None
    if cache_hi and cache_hi.get("version") == 1:
        # hist entries stored as list -> ndarray
        for k, arr in cache_hi.get("data", {}).items():
            try:
                hist_map[k] = np.asarray(arr, dtype=np.float32)
            except Exception:
                pass

    # compute missing with a thread pool
    tohash: List[Tuple[str, Path]] = []
    tohist: List[Tuple[str, Path]] = []
    for src_rel, vp in paths.items():
        p_gray = getattr(vp, f"{ch_phash}_{track}")
        p_color = getattr(vp, f"{ch_hist}_{track}")
        if p_gray and src_rel not in phash_map:
            tohash.append((src_rel, p_gray))
        if p_color and src_rel not in hist_map:
            tohist.append((src_rel, p_color))

    def _hash_job(job) -> Tuple[str, Optional[int]]:
        src_rel, p = job
        img = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return src_rel, None
        try:
            return src_rel, phash64(img)
        except Exception:
            return src_rel, None

    def _hist_job(job) -> Tuple[str, Optional[np.ndarray]]:
        src_rel, p = job
        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            return src_rel, None
        try:
            return src_rel, hsv_histogram(img, bins_h, bins_s, bins_v)
        except Exception:
            return src_rel, None

    if tohash:
        with futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for k, v in ex.map(_hash_job, tohash):
                if v is not None:
                    phash_map[k] = int(v)

    if tohist:
        with futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for k, v in ex.map(_hist_job, tohist):
                if v is not None:
                    hist_map[k] = np.asarray(v, dtype=np.float32)

    # persist cache
    if cache_p_ph:
        save_cache_json(cache_p_ph, {"version": 1, "data": {k: int(v) for k, v in phash_map.items()}})
    if cache_p_hi:
        save_cache_json(cache_p_hi, {
            "version": 1,
            "meta": {"bins_h": bins_h, "bins_s": bins_s, "bins_v": bins_v, "count": len(hist_map)},
            "data": {k: v.tolist() for k, v in hist_map.items()},
        })
    return phash_map, hist_map

# =========================
# --- Geometry metrics ----
# =========================

def compute_edge_density(path_gray_low: Path) -> Optional[float]:
    img = imread_unicode(path_gray_low, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    try:
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img_blur, 50, 150)
        return float(np.count_nonzero(edges)) / float(edges.size)
    except Exception:
        return None

def build_geom_cache(paths: Dict[str, VariantPaths],
                     which: str,
                     track: str,
                     cache_dir: Optional[Path],
                     workers: int) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      geom_map: {src_relpath: {"edge_density": float}}
    """
    cache_path = None
    if cache_dir:
        cache_path = cache_dir / f"geom_{which}_{track}.json"

    geom_map: Dict[str, Dict[str, float]] = {}
    cache = load_cache_json(cache_path) if cache_path else None
    if cache and cache.get("version") == 1:
        for k, d in cache.get("data", {}).items():
            try:
                geom_map[k] = {"edge_density": float(d.get("edge_density", 0.0))}
            except Exception:
                pass

    todo = []
    for src_rel, vp in paths.items():
        if src_rel not in geom_map:
            p = getattr(vp, f"gray_{track}")
            if p:
                todo.append((src_rel, p))

    if todo:
        with futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for src_rel, ed in ex.map(lambda job: (job[0], compute_edge_density(job[1])), todo):
                if ed is not None:
                    geom_map[src_rel] = {"edge_density": float(ed)}

    if cache_path:
        save_cache_json(cache_path, {
            "version": 1,
            "data": {k: {"edge_density": float(v.get("edge_density", 0.0))} for k, v in geom_map.items()},
        })
    return geom_map

# =========================
# ---- Feedback loader ----
# =========================

def _normalize_ref_key(s: str) -> str:
    """Accepts 'low|<rel>' or '<rel>'; returns '<rel>' with forward slashes."""
    s = (s or "").strip().replace("\\", "/")
    if "|" in s:
        pref, rest = s.split("|", 1)
        if pref in ("low", "high"):
            return rest
    return s

@dataclass
class RefOverride:
    phash_radius_delta: Optional[int] = None
    hist_threshold: Optional[float] = None
    min_cand_per_basis: Optional[int] = None

@dataclass
class FeedbackConfig:
    expand_refs: Set[str]
    global_override: RefOverride
    per_ref_override: Dict[str, RefOverride]
    selected_refs_from_feedback: Set[str]   # for --only-refs-from-feedback

def load_feedback(path: Optional[Path]) -> FeedbackConfig:
    expand_refs: Set[str] = set()
    per_ref_override: Dict[str, RefOverride] = {}
    glob = RefOverride()
    selected_refs_from_feedback: Set[str] = set()

    if not path or not path.exists():
        return FeedbackConfig(expand_refs, glob, per_ref_override, selected_refs_from_feedback)

    try:
        fb = json.load(open(path, "r", encoding="utf-8"))
        actions = fb.get("actions", {})

        # expand_candidates
        ec = actions.get("expand_candidates", {}) or {}
        for s in ec.get("refs", []) or []:
            expand_refs.add(_normalize_ref_key(str(s)))
        # global override
        if "phash_radius_delta" in ec:
            try:
                glob.phash_radius_delta = int(ec["phash_radius_delta"])
            except Exception:
                pass
        if "hist_threshold_new" in ec:
            try:
                glob.hist_threshold = float(ec["hist_threshold_new"])
            except Exception:
                pass
        if "min_cand_per_basis" in ec:
            try:
                glob.min_cand_per_basis = int(ec["min_cand_per_basis"])
            except Exception:
                pass
        # per_ref overrides
        for item in ec.get("per_ref", []) or []:
            ref = _normalize_ref_key(str(item.get("ref", "")))
            if not ref:
                continue
            r = RefOverride()
            if "phash_radius_delta" in item:
                try: r.phash_radius_delta = int(item["phash_radius_delta"])
                except Exception: pass
            if "hist_threshold" in item:
                try: r.hist_threshold = float(item["hist_threshold"])
                except Exception: pass
            if "min_cand_per_basis" in item:
                try: r.min_cand_per_basis = int(item["min_cand_per_basis"])
                except Exception: pass
            per_ref_override[ref] = r

        # also consider redo_preprocess targets as "selected" for subset runs
        rp = actions.get("redo_preprocess", {}) or {}
        for s in (rp.get("refs") or []):
            selected_refs_from_feedback.add(_normalize_ref_key(str(s)))

        # final selection set for --only-refs-from-feedback
        selected_refs_from_feedback |= expand_refs

    except Exception as e:
        print(f"[WARN] feedback parse failed: {e}")

    return FeedbackConfig(expand_refs, glob, per_ref_override, selected_refs_from_feedback)

def load_only_refs_list(path: Optional[Path]) -> Set[str]:
    out: Set[str] = set()
    if not path:
        return out
    if not path.exists():
        print(f"[WARN] only-refs file not found: {path}")
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(_normalize_ref_key(s))
    return out

# =========================
# ---- Candidate build ----
# =========================

@dataclass
class Candidate:
    extracted_rel: str
    phash_dist: int
    hist_dist: float

def build_candidates(ex_paths: Dict[str, VariantPaths],
                     ref_paths: Dict[str, VariantPaths],
                     phash_ex: Dict[str, int],
                     phash_ref: Dict[str, int],
                     hist_ex: Dict[str, np.ndarray],
                     hist_ref: Dict[str, np.ndarray],
                     geom_ex: Dict[str, Dict[str, float]],
                     geom_ref: Dict[str, Dict[str, float]],
                     mp_raw: Dict,
                     phash_threshold: int,
                     hist_threshold: float,
                     min_cand_per_basis: int,
                     phash_step: int,
                     phash_max: int,
                     prefilter_aspect: Optional[float],
                     prefilter_edge: Optional[float],
                     selected_refs: Optional[Set[str]],
                     overrides: FeedbackConfig) -> Tuple[Dict[str, List[Candidate]], Dict[str, object], int]:
    """
    Returns:
      cand_map: mapping key "low|<ref_src_relpath>" -> [Candidate...]
      stats:    summary information (counts, distributions)
      override_count: number of refs for which overrides were applied
    """
    # Aspect ratio (from mapping's orig_wh)
    orig_wh_ex: Dict[str, Tuple[int,int]] = {}
    orig_wh_rf: Dict[str, Tuple[int,int]] = {}
    for src_rel, info in mp_raw["by_src"]["extracted"].items():
        w,h = info.get("orig_wh", [0,0])
        if w and h:
            orig_wh_ex[src_rel] = (int(w), int(h))
    for src_rel, info in mp_raw["by_src"]["reference"].items():
        w,h = info.get("orig_wh", [0,0])
        if w and h:
            orig_wh_rf[src_rel] = (int(w), int(h))

    # BK-tree on EXTRACTED pHash
    bkt = BKTree()
    phash_to_rel: Dict[int, List[str]] = {}
    for rel, hv in phash_ex.items():
        bkt.add(hv)
        phash_to_rel.setdefault(hv, []).append(rel)

    # reference list (optionally filtered)
    ref_list = list(ref_paths.keys())
    if selected_refs:
        ref_list = [r for r in ref_list if r in selected_refs]

    # Stats accumulators
    cand_counts: List[int] = []
    removed_by_aspect = 0
    removed_by_edge   = 0
    override_count    = 0

    def aspect_ratio_of(src_rel: str, kind: str) -> Optional[float]:
        wh = orig_wh_rf.get(src_rel) if kind == "ref" else orig_wh_ex.get(src_rel)
        if not wh:
            return None
        w,h = wh
        if w <= 0 or h <= 0:
            return None
        return float(w) / float(h)

    def edge_density_of(src_rel: str, kind: str) -> Optional[float]:
        d = geom_ref.get(src_rel) if kind == "ref" else geom_ex.get(src_rel)
        if not d:
            return None
        return float(d.get("edge_density", 0.0))

    out: Dict[str, List[Candidate]] = {}
    for i, ref_rel in enumerate(ref_list, 1):
        key = f"low|{ref_rel}"
        ref_hash = phash_ref.get(ref_rel)
        ref_hist = hist_ref.get(ref_rel)
        if ref_hash is None or ref_hist is None:
            out[key] = []
            cand_counts.append(0)
            continue

        # local thresholds
        loc_phash_thr   = int(phash_threshold)
        loc_hist_thr    = float(hist_threshold)
        loc_min_cand    = int(min_cand_per_basis)
        loc_phash_max   = int(phash_max)

        # apply overrides if this ref is in overrides.expand_refs or per_ref_override exists
        if (ref_rel in overrides.expand_refs) or (ref_rel in overrides.per_ref_override):
            override_count += 1
            r = overrides.per_ref_override.get(ref_rel, RefOverride())
            # take per-ref first, fallback to global
            delta = r.phash_radius_delta if r.phash_radius_delta is not None else overrides.global_override.phash_radius_delta
            if isinstance(delta, int):
                loc_phash_thr = min(loc_phash_max, loc_phash_thr + int(delta))
                # allow expansion to at least loc_phash_thr (progressive search still uses phash_step up to loc_phash_thr)

            ht = r.hist_threshold if r.hist_threshold is not None else overrides.global_override.hist_threshold
            if ht is not None:
                try:
                    loc_hist_thr = float(ht)
                except Exception:
                    pass

            mc = r.min_cand_per_basis if r.min_cand_per_basis is not None else overrides.global_override.min_cand_per_basis
            if mc is not None:
                try:
                    loc_min_cand = max(1, int(mc))
                except Exception:
                    pass

        # cheap prefilter references
        ar_ref = aspect_ratio_of(ref_rel, "ref")
        ed_ref = edge_density_of(ref_rel, "ref")

        # progressive radius expansion
        cand_rels: List[str] = []
        for radius in range(min(phash_threshold, loc_phash_thr), loc_phash_thr + 1, max(1, int(phash_step))):
            neighbor_vals = bkt.query(ref_hash, radius)
            for hv in neighbor_vals:
                for ex_rel in phash_to_rel.get(hv, []):
                    # prefilters (aspect/edge)
                    if prefilter_aspect is not None and prefilter_aspect >= 0 and ar_ref is not None:
                        ar_ex = aspect_ratio_of(ex_rel, "ex")
                        if ar_ex is not None:
                            # |log(ar_ex/ar_ref)| <= T
                            dv = abs(math.log(max(1e-8, ar_ex) / max(1e-8, ar_ref)))
                            if dv > prefilter_aspect:
                                removed_by_aspect += 1
                                continue
                    if prefilter_edge is not None and prefilter_edge >= 0 and ed_ref is not None:
                        ed_ex = edge_density_of(ex_rel, "ex")
                        if ed_ex is not None:
                            if abs(ed_ex - ed_ref) > prefilter_edge:
                                removed_by_edge += 1
                                continue
                    cand_rels.append(ex_rel)
            cand_rels = list(dict.fromkeys(cand_rels))  # unique, stable
            if len(cand_rels) >= loc_min_cand or radius >= loc_phash_thr:
                break

        cand_list: List[Candidate] = []
        if cand_rels:
            for ex_rel in cand_rels:
                ex_hist = hist_ex.get(ex_rel)
                if ex_hist is None:
                    continue
                pd = hamming64(ref_hash, phash_ex[ex_rel])
                hd = bhattacharyya_distance(ref_hist, ex_hist)
                if hd <= loc_hist_thr:
                    cand_list.append(Candidate(ex_rel, pd, hd))

        # If not enough after hist filter, fill by smallest hist even if > threshold
        if len(cand_list) < loc_min_cand and cand_rels:
            tmp: List[Candidate] = []
            for ex_rel in cand_rels:
                ex_hist = hist_ex.get(ex_rel)
                if ex_hist is None:
                    continue
                pd = hamming64(ref_hash, phash_ex[ex_rel])
                hd = bhattacharyya_distance(ref_hist, ex_hist)
                tmp.append(Candidate(ex_rel, pd, hd))
            tmp.sort(key=lambda c: (c.hist_dist, c.phash_dist))
            need = loc_min_cand - len(cand_list)
            cand_list.extend(tmp[:need])

        # final order: (hist, phash)
        cand_list.sort(key=lambda c: (c.hist_dist, c.phash_dist))
        out[key] = cand_list
        cand_counts.append(len(cand_list))

        if i % 50 == 0 or i == len(ref_list):
            print(f"[INFO] Candidates progress: {i}/{len(ref_list)}")

    stats = {
        "reference_total": len(ref_list),
        "extracted_total": len(ex_paths),
        "cand_per_ref": {
            "mean": float(np.mean(cand_counts)) if cand_counts else 0.0,
            "median": float(np.median(cand_counts)) if cand_counts else 0.0,
            "min": int(min(cand_counts)) if cand_counts else 0,
            "max": int(max(cand_counts)) if cand_counts else 0,
            "p10": float(np.percentile(cand_counts, 10)) if cand_counts else 0.0,
            "p90": float(np.percentile(cand_counts, 90)) if cand_counts else 0.0,
        },
        "prefilter_removed": {
            "aspect": int(removed_by_aspect),
            "edge": int(removed_by_edge),
        }
    }
    return out, stats, override_count

# =========================
# ------- BK-Tree ---------
# =========================

class BKNode:
    __slots__ = ("val", "children")
    def __init__(self, val: int):
        self.val = val
        self.children: Dict[int, "BKNode"] = {}

class BKTree:
    def __init__(self):
        self.root: Optional[BKNode] = None

    def add(self, val: int) -> None:
        if self.root is None:
            self.root = BKNode(val)
            return
        node = self.root
        while True:
            d = hamming64(val, node.val)
            nxt = node.children.get(d)
            if nxt is None:
                node.children[d] = BKNode(val)
                return
            node = nxt

    def query(self, val: int, radius: int) -> List[int]:
        """
        Return all stored values with Hamming distance <= radius.
        """
        out: List[int] = []
        def _dfs(node: BKNode, d_root: int):
            if d_root <= radius:
                out.append(node.val)
            lo = d_root - radius
            hi = d_root + radius
            for dist, child in node.children.items():
                if lo <= dist <= hi:
                    _dfs(child, hamming64(val, child.val))
        if self.root is None:
            return out
        _dfs(self.root, hamming64(val, self.root.val))
        return out

# =========================
# ------- Verify ----------
# =========================

def verify_integrity(ex_paths: Dict[str, VariantPaths],
                     ref_paths: Dict[str, VariantPaths]) -> None:
    """
    Quick integrity check: presence of low/gray & low/color for all.
    """
    def _check(kind: str, paths: Dict[str, VariantPaths]) -> Tuple[int, int, int]:
        miss_gray = miss_color = 0
        for src_rel, vp in paths.items():
            if not vp.gray_low or not vp.gray_low.exists():
                miss_gray += 1
            if not vp.color_low or not vp.color_low.exists():
                miss_color += 1
        total = len(paths)
        print(f"[VERIFY] {kind}: total={total} miss_gray={miss_gray} miss_color={miss_color}")
        return total, miss_gray, miss_color

    _check("extracted", ex_paths)
    _check("reference", ref_paths)

# =========================
# ---------- CLI ----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate candidates.json using pHash (BK-tree) + HSV Bhattacharyya filtering (+ optional cheap prefilters)."
    )
    ap.add_argument("extracted_out", help="02_preprocess.py generated extracted output root (kept for compatibility; not used)")
    ap.add_argument("reference_out", help="02_preprocess.py generated reference output root (kept for compatibility; not used)")

    ap.add_argument("--mapping", default="preprocess_mapping.json", help="Preprocess mapping JSON")
    ap.add_argument("--out", default="candidates.json", help="Output candidates JSON")
    ap.add_argument("--cache", default=None, help="Feature cache dir (e.g., .cache/features)")

    ap.add_argument("--workers", type=int, default=8, help="Thread workers for feature/geom cache")

    # Feature params
    ap.add_argument("--bins-h", type=int, default=16)
    ap.add_argument("--bins-s", type=int, default=8)
    ap.add_argument("--bins-v", type=int, default=8)

    # pHash search
    ap.add_argument("--phash-threshold", type=int, default=28, help="Initial Hamming radius")
    ap.add_argument("--phash-step", type=int, default=2, help="Expansion step")
    ap.add_argument("--phash-max", type=int, default=36, help="Radius upper bound")

    # Histogram filter
    ap.add_argument("--hist-threshold", type=float, default=0.70, help="Bhattacharyya distance threshold (smaller is more similar)")
    ap.add_argument("--min-cand-per-basis", type=int, default=5, help="Minimum candidates per reference")

    # Cheap prefilters (negative value disables)
    ap.add_argument("--prefilter-aspect", type=float, default=0.50,
                    help="Allow |log(ar_ex/ar_ref)| ≤ T. e.g., 0.5≈±65%%  (set <0 to disable)")
    ap.add_argument("--prefilter-edge", type=float, default=0.30,
                    help="Allow |edge_ex - edge_ref| ≤ T  (set <0 to disable)")

    # Feedback & partial processing
    ap.add_argument("--feedback-in", default=None, help="05 output feedback.json (optional)")
    ap.add_argument("--only-refs-from-feedback", action="store_true",
                    help="Process only refs listed in feedback.expand_candidates.refs (+ redo_preprocess.refs)")
    ap.add_argument("--only-refs", default=None,
                    help="Text file of src_relpath list to process; accepts lines like 'low|<rel>' or '<rel>'")

    ap.add_argument("--verify", action="store_true", help="Run quick integrity check and continue")

    return ap.parse_args()

# =========================
# --------- MAIN ----------
# =========================

def main() -> int:
    args = parse_args()

    mapping_json = Path(args.mapping)
    if not mapping_json.exists():
        print(f"[ERR] mapping JSON not found: {mapping_json}")
        return 2

    ex_paths, ref_paths, mp_raw = load_mapping(mapping_json)

    # optional verify
    if args.verify:
        verify_integrity(ex_paths, ref_paths)

    # feature/geom caches
    cache_dir = Path(args.cache) if args.cache else None

    print("[INFO] Building feature cache (extracted)...")
    phash_ex, hist_ex = build_feature_cache(
        ex_paths, which="extracted", track="low",
        ch_phash="gray", ch_hist="color",
        cache_dir=cache_dir, workers=args.workers,
        bins_h=args.bins_h, bins_s=args.bins_s, bins_v=args.bins_v
    )
    print("[INFO] Building feature cache (reference)...")
    phash_rf, hist_rf = build_feature_cache(
        ref_paths, which="reference", track="low",
        ch_phash="gray", ch_hist="color",
        cache_dir=cache_dir, workers=args.workers,
        bins_h=args.bins_h, bins_s=args.bins_s, bins_v=args.bins_v
    )

    print("[INFO] Building geometry cache (edge density)...")
    geom_ex = build_geom_cache(ex_paths, which="extracted", track="low", cache_dir=cache_dir, workers=args.workers)
    geom_rf = build_geom_cache(ref_paths, which="reference", track="low", cache_dir=cache_dir, workers=args.workers)

    # feedback / selection / overrides
    fb = load_feedback(Path(args.feedback_in) if args.feedback_in else None)
    selected_refs: Optional[Set[str]] = None

    # external only-refs list
    only_refs_file = load_only_refs_list(Path(args.only_refs)) if args.only_refs else set()

    if args.only_refs_from_feedback:
        selected_refs = set(fb.selected_refs_from_feedback)
        if not selected_refs:
            print("[WARN] --only-refs-from-feedback given but feedback has no targets; nothing to do?")
    elif only_refs_file:
        selected_refs = set(only_refs_file)

    # build candidates (maybe partial)
    cand_map, stats, override_count = build_candidates(
        ex_paths, ref_paths,
        phash_ex, phash_rf,
        hist_ex, hist_rf,
        geom_ex, geom_rf,
        mp_raw,
        phash_threshold=args.phash_threshold,
        hist_threshold=args.hist_threshold,
        min_cand_per_basis=args.min_cand_per_basis,
        phash_step=args.phash_step,
        phash_max=args.phash_max,
        prefilter_aspect=args.prefilter_aspect,
        prefilter_edge=args.prefilter_edge,
        selected_refs=selected_refs,
        overrides=fb
    )

    # if partial: merge into existing output
    out_path = Path(args.out)
    merged_cand: Dict[str, List[Dict[str, object]]] = {}
    if selected_refs and out_path.exists():
        try:
            old = json.load(open(out_path, "r", encoding="utf-8"))
            if isinstance(old.get("candidates"), dict):
                merged_cand = {k: v for k, v in old["candidates"].items()}
        except Exception as e:
            print(f"[WARN] failed to read existing candidates for merge: {e}")

    # convert internal Candidate -> dict and merge
    def _serialize_map(cmap: Dict[str, List[Candidate]]) -> Dict[str, List[Dict[str, object]]]:
        return {
            k: [
                {"extracted": c.extracted_rel, "phash_dist": int(c.phash_dist), "hist_dist": float(c.hist_dist)}
                for c in v
            ]
            for k, v in cmap.items()
        }

    ser_new = _serialize_map(cand_map)
    if selected_refs and merged_cand:
        merged_cand.update(ser_new)  # replace processed keys
        final_cand = merged_cand
        # recompute stats on final_cand (for completeness)
        lens = [len(v) for v in final_cand.values()]
        stats = {
            "reference_total": len(final_cand),
            "extracted_total": len(ex_paths),
            "cand_per_ref": {
                "mean": float(np.mean(lens)) if lens else 0.0,
                "median": float(np.median(lens)) if lens else 0.0,
                "min": int(min(lens)) if lens else 0,
                "max": int(max(lens)) if lens else 0,
                "p10": float(np.percentile(lens, 10)) if lens else 0.0,
                "p90": float(np.percentile(lens, 90)) if lens else 0.0,
            },
            "prefilter_removed": stats.get("prefilter_removed", {"aspect": 0, "edge": 0}),
        }
    else:
        final_cand = ser_new

    out_obj = {
        "version": 2,
        "basis": "reference",
        "track": "low",
        "features": {
            "phash_channel": "gray",
            "hist_channel": "color",
            "bins_h": int(args.bins_h),
            "bins_s": int(args.bins_s),
            "bins_v": int(args.bins_v),
        },
        "thresholds": {
            "phash": int(args.phash_threshold),
            "hist": float(args.hist_threshold),
        },
        "min_cand_per_basis": int(args.min_cand_per_basis),
        "prefilter": {
            "aspect_log_abs_max": float(args.prefilter_aspect),
            "edge_density_abs_diff_max": float(args.prefilter_edge),
        },
        "overrides": {
            "count": int(override_count),
            "from_feedback": bool(bool(args.feedback_in)),
        },
        "stats": stats,
        "candidates": final_cand,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[INFO] candidates saved: {out_path} (refs={len(final_cand)}, overrides={override_count})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
