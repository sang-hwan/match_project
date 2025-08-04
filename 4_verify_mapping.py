# 4_verify_mapping.py
"""
4_verify_mapping.py

Verify a 1 : 1 mapping between pre-HWP **original photos** and HWP-extracted
images, using pre-processed inputs. The script chooses the best match for each
original by ORB (with optional SIFT fallback) feature matching and RANSAC
filtering.

Key updates in this edition:
• Detect category (`processed/extracted` vs `processed/original`) from `"BinData"`.
• Lookup keys: `(category, track, channel)`.
• `select_preprocessed()` prefers preferred category, then fallback.
• `read_gray()` returns None on missing or decode failure.
• Disabled OpenCV threading (`cv2.setNumThreads(0)`).

Usage example:
  python 4_verify_mapping.py \
      -c candidates.json \
      -m preprocess_mapping.json \
      -i processed \
      -o mapping_result.json \
      -w 8
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np

# Disable OpenCV’s internal threading when using multiprocessing
cv2.setNumThreads(0)

# Matching parameters
LOWE_ORB = 0.75
LOWE_SIFT = 0.70
ORB_FEATURES = 700
SIFT_FEATURES = 1000
MIN_MATCHES = 8
RANSAC_THRESHOLD = 5.0
RANSAC_MIN_INLIERS = 6
RANSAC_MIN_INLIER_RATIO = 0.15
USE_SIFT_FALLBACK = True


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_gray(path: Path) -> np.ndarray | None:
    """Read an image file as grayscale; return None on failure."""
    if not path.is_file():
        return None
    try:
        buf = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def build_lookup(mapping: dict) -> dict:
    """
    Build a lookup:
        original_path -> {(category, track, channel): relative_preproc_path}
    """
    lut: dict[str, dict[tuple[str, str, str], str]] = {}
    for fname, meta in mapping.items():
        origin = meta["원본_전체_경로"]
        category = "extracted" if "BinData" in origin else "original"
        track = meta["트랙"].lower()
        channel = meta["채널"].lower()
        rel = f"{category}/{track}/{channel}/{fname}"
        lut.setdefault(origin, {})[(category, track, channel)] = rel
    return lut


def pick_preproc(
    origin: str,
    preferred_cat: str,
    track: str,
    prefer_gray: bool,
    lut: dict,
) -> str | None:
    """Return a relative pre-processed path. Prefer gray, then color."""
    options = lut.get(origin, {})
    if not options:
        return None
    order = ["gray", "color"] if prefer_gray else ["color", "gray"]

    # Same category first
    for ch in order:
        key = (preferred_cat, track, ch)
        if key in options:
            return options[key]

    # Fallback to opposite category
    for (cat, trk, ch), rel in options.items():
        if trk == track and ch in order:
            return rel
    return None


# Feature extractors and matchers
orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=10)
sift = (
    cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    if USE_SIFT_FALLBACK and hasattr(cv2, "SIFT_create")
    else None
)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) if sift else None


def good_matches(d1, d2):
    """Return good feature matches using ORB, fallback to SIFT if needed."""
    m = bf_orb.knnMatch(d1, d2, k=2)
    g = [a for a, b in m if a.distance < LOWE_ORB * b.distance]
    if len(g) < MIN_MATCHES and sift:
        m = bf_sift.knnMatch(d1, d2, k=2)
        g = [a for a, b in m if a.distance < LOWE_SIFT * b.distance]
    return g


def ransac_score(kp1, kp2, matches):
    """Return (inliers, inlier_ratio) from a homography RANSAC test."""
    if len(matches) < MIN_MATCHES:
        return 0, 0.0
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_THRESHOLD)
    if mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / len(matches)


def evaluate(task):
    origin, info, lut, root = task
    track = info.get("track", "low").lower()
    origin_cat = "extracted" if "BinData" in origin else "original"

    # Target image: try gray first, then color
    tgt_rel = (
        pick_preproc(origin, origin_cat, track, True, lut)
        or pick_preproc(origin, origin_cat, track, False, lut)
    )
    if not tgt_rel:
        return origin, None
    img_t = read_gray(root / tgt_rel)
    if img_t is None:
        return origin, None
    kp_t, desc_t = orb.detectAndCompute(img_t, None)
    if desc_t is None:
        return origin, None

    best_score = None
    best_candidate_origin = None

    for cand in info.get("candidates", []):
        cand_origin = cand["name"]
        # Skip same-side matches
        if ("BinData" in cand_origin) == ("BinData" in origin):
            continue
        cand_cat = "extracted" if "BinData" in cand_origin else "original"

        # Candidate image: gray first, then color
        cand_rel = (
            pick_preproc(cand_origin, cand_cat, track, True, lut)
            or pick_preproc(cand_origin, cand_cat, track, False, lut)
        )
        if not cand_rel:
            continue
        img_c = read_gray(root / cand_rel)
        if img_c is None:
            continue
        kp_c, desc_c = orb.detectAndCompute(img_c, None)
        if desc_c is None:
            continue

        matches = good_matches(desc_t, desc_c)
        inliers, ratio = ransac_score(kp_t, kp_c, matches)
        if inliers >= RANSAC_MIN_INLIERS and ratio >= RANSAC_MIN_INLIER_RATIO:
            score = (inliers, ratio, len(matches))
            if best_score is None or score > best_score:
                best_score = score
                best_candidate_origin = cand_origin

    return origin, best_candidate_origin


def main():
    parser = argparse.ArgumentParser(
        description="Verify 1-to-1 mapping between original and extracted images"
    )
    parser.add_argument("-c", "--candidates", required=True)
    parser.add_argument("-m", "--mapping", required=True)
    parser.add_argument("-i", "--img-root", required=True)
    parser.add_argument("-o", "--output", default="mapping_result.json")
    parser.add_argument("-w", "--workers", type=int, default=max(cpu_count() - 1, 1))
    args = parser.parse_args()

    candidates = load_json(args.candidates)
    mapping = load_json(args.mapping)
    lookup = build_lookup(mapping)
    root = Path(args.img_root).resolve()

    tasks = [(orig, info, lookup, root) for orig, info in candidates.items()]
    print(f"[INFO] matching {len(tasks)} originals with {args.workers} workers…")

    with Pool(args.workers) as pool:
        results = pool.map(evaluate, tasks)

    matched = {k: v for k, v in results if v}
    unmatched = [k for k, v in results if not v]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(matched, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY]")
    print(f"total originals : {len(tasks)}")
    print(f"matched         : {len(matched)}")
    print(f"unmatched       : {len(unmatched)}")
    if unmatched:
        print("first unmatched examples:")
        for u in unmatched[:10]:
            print("  -", u)


if __name__ == "__main__":
    main()
