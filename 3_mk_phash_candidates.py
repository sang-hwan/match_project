# 3_mk_phash_candidates.py
"""
3_mk_phash_candidates.py
========================
Generate pHash + HSV‑histogram‑based candidate lists for each original image.

Key points
----------
* `preprocess_mapping.json` is the **single source of truth** for track (low/high),
  channel (gray/color) and original‑image identity (`원본_전체_경로`).
* A file belongs to **processed/extracted** if its identity path contains
  `"BinData"`, otherwise to **processed/original**.
* Images are split by `(track, channel)`; tracks are processed independently
  so that low/high resolutions never mix.
* For every gray image in the **smaller side** (extracted vs original), find
  candidates on the opposite side that satisfy:

  1. pHash Hamming distance ≤ `--phash-threshold`
  2. Bhattacharyya distance between HSV histograms ≤ `--hist-threshold`
  3. Different original identity (이미 매핑된 동일 원본은 제외)

* Results are written as JSON mapping original‑ID → candidate list::

    {
        "C:/…/1.jpg": {
            "track": "high",
            "candidates": [
                { "name": "C:/…/9.jpg",  "phash": 24, "hist": 0.62 },
                { "name": "C:/…/27.jpg", "phash": 31, "hist": 0.73 }
            ]
        },
        …
    }

Usage
-----
python 3_mk_phash_candidates.py \
       processed/extracted  processed/original  out/candidates.json \
       --mapping preprocess_mapping.json \
       [--phash-threshold 38] [--hist-threshold 0.81]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate candidate list keyed by original photo path"
    )
    p.add_argument("extracted_root")
    p.add_argument("original_root")
    p.add_argument("out_json")
    p.add_argument("--mapping", required=True)
    p.add_argument("--phash-threshold", type=int, default=38)
    p.add_argument("--hist-threshold", type=float, default=0.81)
    return p.parse_args()


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#
def norm_channel(ch: str) -> str:
    """Normalize channel name to 'gray' or 'color'."""
    ch = ch.lower().split("_")[0]
    return "gray" if ch.startswith(("gray", "grey")) else "color"


def compute_phash(p: Path):
    with Image.open(p) as im:
        return imagehash.phash(im)


def compute_hist(p: Path):
    with Image.open(p) as im:
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = [cv2.calcHist([hsv], [c], None, [32], [0, 256]) for c in range(3)]
    for h in feats:
        cv2.normalize(h, h)
    return np.concatenate([h.flatten() for h in feats])


# ---------------------------------------------------------------------------#
# Main                                                                       #
# ---------------------------------------------------------------------------#
def main():
    args = parse_args()
    ex_root = Path(args.extracted_root).resolve()
    or_root = Path(args.original_root).resolve()
    print("[PARAM]", vars(args))

    # --------------------------------------------------------------------- #
    # Load mapping JSON                                                     #
    # --------------------------------------------------------------------- #
    with open(args.mapping, encoding="utf-8") as f:
        mp = json.load(f)

    # Split files into extracted / original dictionaries keyed by (track,ch)
    extracted: dict[tuple[str, str], list[Path]] = defaultdict(list)
    original:  dict[tuple[str, str], list[Path]] = defaultdict(list)

    for name, meta in mp.items():
        track = meta.get("트랙", "").lower()
        ch = norm_channel(meta.get("채널", ""))
        if not track or not ch:
            continue

        rel_path = Path(track) / ch / name
        is_extracted = "BinData" in meta.get("원본_전체_경로", "")
        full_path = (ex_root if is_extracted else or_root) / rel_path
        if full_path.is_file():
            (extracted if is_extracted else original)[(track, ch)].append(full_path)

    # Quick lookup: (원본_전체_경로, track, channel) → pre‑processed filename
    id2name = {
        (m["원본_전체_경로"], m["트랙"].lower(), norm_channel(m["채널"])): n
        for n, m in mp.items()
    }

    out: dict[str, dict] = {}
    basis_cnt = cand_cnt = 0

    # --------------------------------------------------------------------- #
    # Process each track independently                                      #
    # --------------------------------------------------------------------- #
    for track in {"low", "high"}:
        gray_ext = extracted.get((track, "gray"), [])
        gray_ori = original.get((track, "gray"), [])
        if not gray_ext or not gray_ori:
            continue

        # Always iterate over the smaller gray set for efficiency
        basis_gray, basis_side = (
            (gray_ori, "original") if len(gray_ori) <= len(gray_ext) else (gray_ext, "extracted")
        )
        compare_gray = gray_ext if basis_side == "original" else gray_ori
        compare_color_root = (ex_root if basis_side == "original" else or_root) / track / "color"

        # Pre‑compute pHash for both sides
        h_basis = {p: compute_phash(p) for p in basis_gray}
        h_compare = {p: compute_phash(p) for p in compare_gray}

        # Cache HSV histograms for color images (index by gray & color names)
        hist_compare: dict[str, np.ndarray] = {}
        for p in compare_color_root.iterdir():
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            hv = compute_hist(p)
            hist_compare[p.name] = hv
            m = mp.get(p.name)
            if m:
                gname = id2name.get((m["원본_전체_경로"], track, "gray"))
                if gname:
                    hist_compare[gname] = hv

        # ---------------------------------------------------------------- #
        # Candidate search per gray basis image                            #
        # ---------------------------------------------------------------- #
        for bpath, bhash in h_basis.items():
            basis_cnt += 1
            mid = mp[bpath.name]["원본_전체_경로"]
            entry = out.setdefault(mid, {"track": track, "candidates": []})

            # Need histogram of the basis image's *color* counterpart
            c_name = id2name.get((mid, track, "color"))
            if not c_name:
                continue
            c_path = (
                (or_root if basis_side == "original" else ex_root)
                / track / "color" / c_name
            )
            basis_hist = compute_hist(c_path)

            # pHash filter then histogram check
            pdists = [(cp, bhash - h_compare[cp]) for cp in h_compare]
            pdists.sort(key=lambda x: x[1])  # ascending distance

            for cp, pdist in pdists:
                if pdist > args.phash_threshold:
                    break  # remaining pairs are even farther

                c_id = mp[cp.name]["원본_전체_경로"]
                if c_id == mid:  # skip same original
                    continue

                hvec = hist_compare.get(cp.name)
                if hvec is None:
                    continue

                hdist = cv2.compareHist(basis_hist, hvec, cv2.HISTCMP_BHATTACHARYYA)
                if hdist <= args.hist_threshold:
                    entry["candidates"].append(
                        {"name": c_id, "phash": int(pdist), "hist": round(float(hdist), 4)}
                    )
                    cand_cnt += 1

    # --------------------------------------------------------------------- #
    # Save results                                                          #
    # --------------------------------------------------------------------- #
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    now_kst = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    print(
        f"[SAVE] {args.out_json}  |  basis={basis_cnt:,}  "
        f"candidates={cand_cnt:,}  |  {now_kst}"
    )


if __name__ == "__main__":
    main()
