# images_mapper.py
"""
Map pre‑processed extracted images to pre‑processed originals, using pHash + ORB(+FLANN)
with optional color-histogram refinement.

Usage
-----
python images_mapper.py \
  extract.json \
  origin.json \
  path/to/processed_extracted \
  path/to/processed_original \
  out_map.json \
  [-k TOP_K] \
  [-t PHASH_THRESHOLD]
"""
from __future__ import annotations

import argparse
import json
import base64
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

# defaults
PHASH_THRESHOLD = 10
TOP_K = 10
FLANN_KNN = 2
RATIO = 0.75

# FLANN matcher for ORB
FLANN = cv2.FlannBasedMatcher(
    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
    dict(checks=50),
)

def hamming(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex phash strings."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def orb_similarity(e_info: dict, o_info: dict) -> float:
    """Compute RANSAC inlier ratio between ORB features (using pre-decoded descriptors)."""
    d1 = e_info.get("orb_desc")
    d2 = o_info.get("orb_desc")
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return 0.0
    matches = FLANN.knnMatch(d1, d2, k=FLANN_KNN)
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    if len(good) < 4:
        return 0.0
    kp1 = np.float32(e_info["kp"])[[m.queryIdx for m in good]]
    kp2 = np.float32(o_info["kp"])[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)
    return float(mask.sum() / len(good)) if mask is not None else 0.0

def hist_similarity(e_info: dict, o_info: dict) -> float:
    """Compute cosine similarity between two normalized color histograms."""
    h1 = np.array(e_info.get("color_hist", []), dtype=np.float32)
    h2 = np.array(o_info.get("color_hist", []), dtype=np.float32)
    denom = np.linalg.norm(h1) * np.linalg.norm(h2)
    return float(np.dot(h1, h2) / denom) if denom > 0 else 0.0

def load_json(path: Path) -> dict[str, dict]:
    """Load JSON file with UTF-8 decoding."""
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    p = argparse.ArgumentParser(description="Map images via pHash, ORB, and histogram")
    p.add_argument("extract_json", help="JSON with extracted-image features")
    p.add_argument("origin_json", help="JSON with original-image features")
    p.add_argument("extracted_dir", help="folder of processed extracted images")
    p.add_argument("original_dir", help="folder of processed original images")
    p.add_argument("out_map", help="output JSON mapping file")
    p.add_argument("-k", "--topk", type=int, default=TOP_K,
                   help="number of pHash candidates to consider")
    p.add_argument("-t", "--phash-th", type=int, default=PHASH_THRESHOLD,
                   help="maximum Hamming distance for pHash filter")
    args = p.parse_args()

    # Load feature dictionaries
    print(f"[INFO] Loading features from JSON")
    ext_infos = load_json(Path(args.extract_json))
    ori_infos = load_json(Path(args.origin_json))
    print(f"[INFO] {len(ext_infos)} extracted / {len(ori_infos)} originals loaded")

    # Pre-decode ORB descriptors to arrays
    for info in ext_infos.values():
        orb_b64 = info.get("orb")
        if orb_b64:
            arr = base64.b64decode(orb_b64)
            info["orb_desc"] = np.frombuffer(arr, np.uint8).reshape(-1, 32)
    for info in ori_infos.values():
        orb_b64 = info.get("orb")
        if orb_b64:
            arr = base64.b64decode(orb_b64)
            info["orb_desc"] = np.frombuffer(arr, np.uint8).reshape(-1, 32)

    mapping: dict[str, str] = {}
    total = len(ext_infos)

    # Iterate over extracted images
    for idx, (e_key, e_info) in enumerate(ext_infos.items(), start=1):
        print(f"[{idx}/{total}] Mapping {e_key}…")
        # 1) pHash coarse filter
        dists = [(o_key, hamming(e_info["phash"], o_info["phash"]))
                 for o_key, o_info in ori_infos.items()]
        dists.sort(key=lambda x: x[1])
        # Debug: show top-5 pHash distances
        print(f"  ▶ Top-5 pHash distances: {dists[:5]}")
        candidates = [o for o, d in dists[: args.topk] if d <= args.phash_th]
        print(f"  pHash candidates (≤{args.phash_th}): {candidates}")
        if not candidates:
            min_o, min_d = dists[0]
            print(f"  [DEBUG] no candidates ≤{args.phash_th}. Min distance is {min_d} for {min_o}")

        # 2) ORB + histogram refinement
        best_score, best_o = -1.0, None
        for o_key in candidates:
            o_info = ori_infos[o_key]
            orb_score = orb_similarity(e_info, o_info)
            hist_score = hist_similarity(e_info, o_info)
            total_score = orb_score + hist_score
            print(f"    {o_key}: ORB={orb_score:.3f}, Hist={hist_score:.3f} → Total={total_score:.3f}")
            if total_score > best_score:
                best_score, best_o = total_score, o_key

        if best_o:
            mapping[e_key] = best_o
            print(f"[MAP] {e_key} → {best_o} (score={best_score:.3f})")
        else:
            print(f"[WARN] no match for {e_key}")

    # Write mapping result
    out_path = Path(args.out_map)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] mapped {len(mapping)}/{total} images → {out_path}")

if __name__ == "__main__":
    main()
