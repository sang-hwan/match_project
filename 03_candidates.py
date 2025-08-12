"""
03_candidates.py

Build candidate pairs (reference -> extracted) using:
- Stage-1: pHash(gray, low) within Hamming radius (BK-tree)
- Stage-2: HSV histogram(color, low) using Bhattacharyya distance
Outputs candidates.json keyed by "low|<reference_src_relpath>"

Usage
-----
python 03_candidates.py EXTRACTED_OUT REFERENCE_OUT \
       --mapping preprocess_mapping.json \
       --out candidates.json \
       --cache .cache/features \
       --phash-threshold 36 \
       --hist-threshold 0.75 \
       --min-cand-per-basis 8 \
       --phash-step 2 --phash-max 48 \
       --verify \
       --workers 8

Notes
-----
- 입력은 02_preprocess.py가 생성한 출력 루트(=전처리본)와 매핑 JSON입니다.
- pHash는 64bit 정수, HSV 히스토그램은 3D(16x8x8) 확률 벡터(합=1)입니다.
- Bhattacharyya distance d = sqrt(1 - sum_i sqrt(p_i * q_i)), 0(유사) ~ 1(상이).
- CPU 환경 최적화를 위해 캐시(JSON)를 사용합니다. 부재 시 즉시 계산 후 저장합니다.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

def bgr_from_any(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# =========================
# ---- Mapping helpers ----
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
# ---- Feature extract ----
# =========================

def phash64(img_gray: np.ndarray) -> int:
    """
    64-bit perceptual hash (DCT based).
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
    # Exclude DC (0,0)
    vals = block.flatten()
    vals = vals[1:]
    med = np.median(vals)
    bits = (vals > med).astype(np.uint8)
    # pack 63 bits + 1 pad bit -> 64
    pad = np.array([0], dtype=np.uint8)
    bits64 = np.concatenate([bits, pad])
    # Compose into 64-bit int (MSB first)
    v = 0
    for b in bits64:
        v = (v << 1) | int(b)
    return int(v)

def hsv_hist(img_bgr: np.ndarray, bins_h: int = 16, bins_s: int = 8, bins_v: int = 8) -> np.ndarray:
    """
    Return normalized HSV histogram (H,S,V in [0,1]/[0,255] ranges).
    Shape: (bins_h*bins_s*bins_v,)
    """
    bgr = bgr_from_any(img_bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2],
                        None,
                        [bins_h, bins_s, bins_v],
                        [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist.flatten()

def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    # BC = sum sqrt(p_i*q_i)
    # d = sqrt(max(0, 1 - BC))
    bc = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.sqrt(np.clip(q, 0, None)))))
    if bc < 0:
        bc = 0.0
    if bc > 1.0:
        bc = 1.0
    d = math.sqrt(1.0 - bc)
    return float(d)

def hamming64(a: int, b: int) -> int:
    x = a ^ b
    try:
        return x.bit_count()  # Py3.10+
    except AttributeError:
        return bin(x).count("1")


# =========================
# ------- BK-Tree ---------
# =========================

class BKNode:
    __slots__ = ("val", "children")
    def __init__(self, val: int):
        self.val = val
        self.children: Dict[int, "BKNode"] = {}

class BKTree:
    """
    BK-tree for Hamming distance on 64-bit integer hashes.
    """
    def __init__(self):
        self.root: Optional[BKNode] = None
        self.values: List[int] = []  # keep order

    def add(self, val: int):
        self.values.append(val)
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

    def build(self, vals: Iterable[int]):
        for v in vals:
            self.add(v)

    def search(self, query: int, radius: int) -> List[int]:
        """
        Returns list of values within radius (inclusive).
        """
        out: List[int] = []
        if self.root is None:
            return out
        stk: List[BKNode] = [self.root]
        while stk:
            node = stk.pop()
            d = hamming64(query, node.val)
            if d <= radius:
                out.append(node.val)
            lo = max(0, d - radius)
            hi = d + radius
            for k, child in node.children.items():
                if lo <= k <= hi:
                    stk.append(child)
        return out


# =========================
# ------ Cache I/O --------
# =========================

def load_cache_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
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
      hist_map:  {src_relpath: np.ndarray (H* S* V)}
    """
    # --- Load caches if available ---
    phash_cache_path = None
    hist_cache_path  = None
    if cache_dir:
        phash_cache_path = cache_dir / f"phash_{which}_{track}_{ch_phash}.json"
        hist_cache_path  = cache_dir / f"hsv_{which}_{track}_{ch_hist}.json"

    phash_map: Dict[str, int] = {}
    hist_map:  Dict[str, np.ndarray] = {}

    phash_cache = load_cache_json(phash_cache_path) if phash_cache_path else None
    hist_cache  = load_cache_json(hist_cache_path) if hist_cache_path else None
    if phash_cache and phash_cache.get("bins") == 64:
        for k, v in phash_cache["data"].items():
            try:
                phash_map[k] = int(v, 16)  # hex string -> int
            except Exception:
                pass
    if hist_cache and hist_cache.get("bins_h") == bins_h and hist_cache.get("bins_s") == bins_s and hist_cache.get("bins_v") == bins_v:
        for k, arr in hist_cache["data"].items():
            hist_map[k] = np.asarray(arr, dtype=np.float32)

    # --- Compute missing ---
    todo_phash = []
    todo_hist  = []
    for src_rel, vp in paths.items():
        # phash target path (gray, low)
        p_ph = getattr(vp, f"{ch_phash}_{track}")
        if p_ph and src_rel not in phash_map:
            todo_phash.append((src_rel, p_ph))
        # hist target path (color, low)
        p_hs = getattr(vp, f"{ch_hist}_{track}")
        if p_hs and src_rel not in hist_map:
            todo_hist.append((src_rel, p_hs))

    def _calc_phash(job):
        src_rel, p = job
        img = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return src_rel, None
        try:
            v = phash64(img)
            return src_rel, v
        except Exception:
            return src_rel, None

    def _calc_hist(job):
        src_rel, p = job
        img = imread_unicode(p, cv2.IMREAD_COLOR)
        if img is None:
            return src_rel, None
        try:
            h = hsv_hist(img, bins_h=bins_h, bins_s=bins_s, bins_v=bins_v)
            return src_rel, h
        except Exception:
            return src_rel, None

    if todo_phash:
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for src_rel, v in ex.map(_calc_phash, todo_phash):
                if v is not None:
                    phash_map[src_rel] = int(v)
    if todo_hist:
        with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for src_rel, h in ex.map(_calc_hist, todo_hist):
                if h is not None:
                    hist_map[src_rel] = h

    # --- Save caches ---
    if cache_dir:
        if phash_cache_path:
            save_cache_json(phash_cache_path, {
                "version": 1,
                "feature": "phash",
                "which": which,
                "track": track,
                "channel": ch_phash,
                "bins": 64,
                "count": len(phash_map),
                "data": {k: f"{v:016x}" for k, v in phash_map.items()},
            })
        if hist_cache_path:
            save_cache_json(hist_cache_path, {
                "version": 1,
                "feature": "hsv",
                "which": which,
                "track": track,
                "channel": ch_hist,
                "bins_h": bins_h, "bins_s": bins_s, "bins_v": bins_v,
                "count": len(hist_map),
                "data": {k: v.tolist() for k, v in hist_map.items()},
            })

    return phash_map, hist_map


# =========================
# ------ Verification -----
# =========================

def verify_integrity(ex_paths: Dict[str, VariantPaths],
                     ref_paths: Dict[str, VariantPaths]) -> None:
    """
    Quick integrity checks: required variants exist for low track.
    """
    def _check(kind: str, paths: Dict[str, VariantPaths]) -> Tuple[int, int, int]:
        miss_gray = miss_color = 0
        for src_rel, vp in paths.items():
            if not vp.gray_low or not vp.gray_low.exists():
                miss_gray += 1
            if not vp.color_low or not vp.color_low.exists():
                miss_color += 1
        total = len(paths)
        print(f"[VERIFY] {kind}: total={total}, missing gray_low={miss_gray}, color_low={miss_color}")
        return total, miss_gray, miss_color

    _check("extracted", ex_paths)
    _check("reference", ref_paths)


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
                     phash_threshold: int,
                     hist_threshold: float,
                     min_cand_per_basis: int,
                     phash_step: int,
                     phash_max: int) -> Dict[str, List[Candidate]]:
    """
    Returns: mapping key "low|<ref_src_relpath>" -> [Candidate...]
    """
    # BK-tree on EXTRACTED pHash
    bkt = BKTree()
    # Also a reverse index: phash value -> list of src_rel (collisions rare but possible)
    phash_to_rel: Dict[int, List[str]] = {}
    for rel, hv in phash_ex.items():
        bkt.add(hv)
        phash_to_rel.setdefault(hv, []).append(rel)

    out: Dict[str, List[Candidate]] = {}
    ref_list = list(ref_paths.keys())

    for i, ref_rel in enumerate(ref_list, 1):
        ref_hash = phash_ref.get(ref_rel)
        ref_hist = hist_ref.get(ref_rel)
        key = f"low|{ref_rel}"

        if ref_hash is None or ref_hist is None:
            out[key] = []
            continue

        # Stage-1: pHash neighbors within radius, widen until >= K or reach max
        radius = phash_threshold
        neigh_vals: List[int] = []
        while True:
            neigh_vals = bkt.search(ref_hash, radius)
            # Map BK results to unique extracted rels
            cand_rels = []
            for hv in neigh_vals:
                cand_rels.extend(phash_to_rel.get(hv, []))
            cand_rels = list(dict.fromkeys(cand_rels))  # dedup preserve order

            if len(cand_rels) >= min_cand_per_basis or radius >= phash_max:
                break
            radius = min(phash_max, radius + phash_step)

        # Stage-2: histogram filter
        cand_list: List[Candidate] = []
        for ex_rel in cand_rels:
            ex_hist = hist_ex.get(ex_rel)
            if ex_hist is None:
                continue
            pd = hamming64(ref_hash, phash_ex[ex_rel])
            hd = bhattacharyya_distance(ref_hist, ex_hist)
            if hd <= hist_threshold:
                cand_list.append(Candidate(ex_rel, pd, hd))

        # If not enough after hist filter, fill with best by hist (even if > threshold)
        if len(cand_list) < min_cand_per_basis:
            # compute hd for all and take top by hist then phash
            tmp: List[Candidate] = []
            for ex_rel in cand_rels:
                ex_hist = hist_ex.get(ex_rel)
                if ex_hist is None:
                    continue
                pd = hamming64(ref_hash, phash_ex[ex_rel])
                hd = bhattacharyya_distance(ref_hist, ex_hist)
                tmp.append(Candidate(ex_rel, pd, hd))
            tmp.sort(key=lambda c: (c.hist_dist, c.phash_dist))
            fill_need = min_cand_per_basis - len(cand_list)
            cand_list.extend(tmp[:fill_need])

        # sort final by (hist, phash)
        cand_list.sort(key=lambda c: (c.hist_dist, c.phash_dist))
        out[key] = cand_list

        if i % 50 == 0 or i == len(ref_list):
            print(f"[INFO] Candidates progress: {i}/{len(ref_list)}")

    return out


# =========================
# ---------- CLI ----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate candidates.json using pHash (BK-tree) + HSV Bhattacharyya filtering."
    )
    ap.add_argument("extracted_out", help="02_preprocess.py에서 생성한 추출 이미지 출력 루트")
    ap.add_argument("reference_out", help="02_preprocess.py에서 생성한 원본 이미지 출력 루트")

    ap.add_argument("--mapping", default="preprocess_mapping.json", help="전처리 매핑 JSON 경로")
    ap.add_argument("--out", default="candidates.json", help="결과 candidates.json 경로")
    ap.add_argument("--cache", default=None, help="특징 캐시 디렉터리(.cache/features 등)")

    ap.add_argument("--workers", type=int, default=8, help="특징 계산 스레드 수")

    # Feature params
    ap.add_argument("--bins-h", type=int, default=16)
    ap.add_argument("--bins-s", type=int, default=8)
    ap.add_argument("--bins-v", type=int, default=8)

    # Thresholding / candidate count
    ap.add_argument("--phash-threshold", type=int, default=36, help="pHash 햄밍 반경 초기값")
    ap.add_argument("--phash-step", type=int, default=2, help="K 미만일 때 반경 증가량")
    ap.add_argument("--phash-max", type=int, default=48, help="반경 상한")
    ap.add_argument("--hist-threshold", type=float, default=0.75, help="Bhattacharyya 거리 임계(작을수록 유사)")
    ap.add_argument("--min-cand-per-basis", type=int, default=8, help="기준 이미지당 최소 후보 수 보장")

    ap.add_argument("--verify", action="store_true", help="전처리 산출물(4콤보) 무결성 검사만 실행")

    return ap.parse_args()

def main() -> int:
    args = parse_args()

    mapping_json = Path(args.mapping)
    if not mapping_json.exists():
        print(f"[ERR] 매핑 JSON을 찾을 수 없습니다: {mapping_json}")
        return 2

    ex_paths, ref_paths, mp = load_mapping(mapping_json)

    # Optional verify
    if args.verify:
        verify_integrity(ex_paths, ref_paths)
        # verify만 하고 종료하려면 여기서 return 0
        # return 0

    # Build or load feature caches
    cache_dir = Path(args.cache) if args.cache else None

    print("[INFO] 특징 캐시 구성 (extracted)...")
    phash_ex, hist_ex = build_feature_cache(
        ex_paths, which="extracted", track="low",
        ch_phash="gray", ch_hist="color",
        cache_dir=cache_dir, workers=args.workers,
        bins_h=args.bins_h, bins_s=args.bins_s, bins_v=args.bins_v
    )
    print(f"[INFO] extracted: pHash={len(phash_ex)}, HSV={len(hist_ex)}")

    print("[INFO] 특징 캐시 구성 (reference)...")
    phash_ref, hist_ref = build_feature_cache(
        ref_paths, which="reference", track="low",
        ch_phash="gray", ch_hist="color",
        cache_dir=cache_dir, workers=args.workers,
        bins_h=args.bins_h, bins_s=args.bins_s, bins_v=args.bins_v
    )
    print(f"[INFO] reference: pHash={len(phash_ref)}, HSV={len(hist_ref)}")

    # Build candidates
    print("[INFO] 후보 생성 시작...")
    cand_map = build_candidates(
        ex_paths=ex_paths, ref_paths=ref_paths,
        phash_ex=phash_ex, phash_ref=phash_ref,
        hist_ex=hist_ex, hist_ref=hist_ref,
        phash_threshold=args.phash_threshold,
        hist_threshold=args.hist_threshold,
        min_cand_per_basis=args.min_cand_per_basis,
        phash_step=args.phash_step, phash_max=args.phash_max
    )

    # Serialize
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {
        "version": 1,
        "basis": "reference",
        "track": "low",
        "features": {"phash_channel": "gray", "hist_channel": "color",
                     "bins_h": args.bins_h, "bins_s": args.bins_s, "bins_v": args.bins_v},
        "thresholds": {"phash": args.phash_threshold, "hist": args.hist_threshold},
        "min_cand_per_basis": args.min_cand_per_basis,
        "stats": {
            "reference_total": len(ref_paths),
            "extracted_total": len(ex_paths),
        },
        "candidates": {
            k: [
                {"extracted": c.extracted_rel, "phash_dist": c.phash_dist, "hist_dist": round(c.hist_dist, 6)}
                for c in v
            ]
            for k, v in cand_map.items()
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"[INFO] candidates 저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
