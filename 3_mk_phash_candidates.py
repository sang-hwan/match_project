# 3_mk_phash_candidates.py
"""
pHash + HSV 히스토그램 기반 후보군 생성 (CPU-only)

입력:
  processed/
    extracted/{low|high}/{gray|color}/<name>.png
    reference/{low|high}/{gray|color}/<name>.png
  preprocess_mapping.json  (키: '트랙','채널','카테고리','원본_전체_경로')

출력:
  out_json (candidates.json 유사 포맷):
    { "<원본_전체_경로>": { "track": "low|high",
                            "candidates": [{ "name": "<원본_전체_경로>",
                                             "phash": <int>,
                                             "hist": <float> }, ...] }, ... }

사용 예:
  python 3_mk_phash_candidates.py processed/extracted processed/reference out/candidates.json \
         --mapping preprocess_mapping.json --phash-threshold 38 --hist-threshold 0.81 \
         --cache-dir .cache/features --workers 8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글 안전

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

# ---- Consts ----
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


# ---- Normalizers ----
def _norm_channel(ch: str) -> str:
    t = ch.lower().strip().split("_")[0]
    return "gray" if t.startswith(("gray", "grey", "그레이")) else "color"


def _normalize_category(cat: str | None, origin_path: str | None) -> str:
    """매핑의 '카테고리' 우선, 없으면 'BinData' 포함 여부로 보정."""
    c = (cat or "").strip().lower()
    if c == "original":
        c = "reference"
    if c in {"extracted", "reference"}:
        return c
    op = (origin_path or "").lower()
    return "extracted" if "bindata" in op else "reference"


# ---- Feature cache (filesystem-based, with compute fallback) ----
class FSCache:
    """
    0_A_cache_features.py가 만든 디렉터리 캐시 사용:
      phash/<cat>/<track>/gray/<name>.txt  (16-hex)
      hist/<cat>/<track>/color/<name>.npy  (96D float32)
    없으면 즉시 계산으로 대체.
    """
    def __init__(self, base: Path, ex_root: Path, ref_root: Path):
        self.base = base
        self.ex_root = ex_root
        self.ref_root = ref_root

    def _rel_info(self, p: Path) -> tuple[str, str, str, str]:
        # (category, track, channel, name)
        name = p.name
        try:
            rel = p.resolve().relative_to(self.ex_root.resolve())
            return "extracted", rel.parts[0], rel.parts[1], name
        except Exception:
            pass
        rel = p.resolve().relative_to(self.ref_root.resolve())
        return "reference", rel.parts[0], rel.parts[1], name

    def get_phash(self, p: Path) -> imagehash.ImageHash:
        from imagehash import hex_to_hash
        cat, track, _ch, name = self._rel_info(p)
        f = self.base / "phash" / cat / track / "gray" / (name + ".txt")
        if f.is_file():
            try:
                return hex_to_hash(f.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        # fallback compute
        with Image.open(p) as im:
            return imagehash.phash(im)

    def get_hist(self, p: Path) -> np.ndarray:
        cat, track, _ch, name = self._rel_info(p)
        f = self.base / "hist" / cat / track / "color" / (name + ".npy")
        if f.is_file():
            try:
                v = np.load(f)
                return np.asarray(v, dtype="float32").reshape(-1)
            except Exception:
                pass
        # fallback compute
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        feats = []
        for c in range(3):
            h = cv2.calcHist([hsv], [c], None, [32], [0, 256])
            cv2.normalize(h, h)
            feats.append(h.flatten())
        return np.concatenate(feats).astype("float32")


# ---- Batch helpers (thread-pooled) ----
def _batch_phash(paths: List[Path], cache: FSCache, workers: int) -> Dict[Path, imagehash.ImageHash]:
    out: Dict[Path, imagehash.ImageHash] = {}
    if not paths:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(cache.get_phash, p): p for p in paths}
        for f in as_completed(fut):
            p = fut[f]
            try:
                out[p] = f.result()
            except Exception:
                try:
                    with Image.open(p) as im:
                        out[p] = imagehash.phash(im)
                except Exception:
                    # skip unreadable
                    pass
    return out


def _batch_hist(paths: List[Path], cache: FSCache, workers: int) -> Dict[str, np.ndarray]:
    """
    반환: basename -> 96D float32.
    preprocess_mapping.json의 키가 파일명 기준이므로 basename으로 맞춘다.
    """
    out: Dict[str, np.ndarray] = {}
    if not paths:
        return out
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut = {ex.submit(cache.get_hist, p): p for p in paths}
        for f in as_completed(fut):
            p = fut[f]
            try:
                v = f.result()
                out[p.name] = np.asarray(v, dtype="float32").reshape(-1)
            except Exception:
                try:
                    # 최후의 보루: 직접 계산
                    with Image.open(p) as im:
                        im = ImageOps.exif_transpose(im)
                        bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    feats = []
                    for c in range(3):
                        h = cv2.calcHist([hsv], [c], None, [32], [0, 256])
                        cv2.normalize(h, h)
                        feats.append(h.flatten())
                    out[p.name] = np.concatenate(feats).astype("float32")
                except Exception:
                    pass
    return out


# ---- CLI ----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make pHash+HSV candidates by (track,channel)")
    p.add_argument("extracted_root")
    p.add_argument("reference_root")
    p.add_argument("out_json")
    p.add_argument("--mapping", required=True, help="preprocess_mapping.json")
    p.add_argument("--phash-threshold", type=int, default=38)
    p.add_argument("--hist-threshold", type=float, default=0.81)
    p.add_argument("--cache-dir", default=".cache/features", help="filesystem feature cache root")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return p.parse_args()


# ---- Main ----
def main() -> None:
    args = parse_args()
    ex_root = Path(args.extracted_root).resolve()
    ref_root = Path(args.reference_root).resolve()
    cache = FSCache(Path(args.cache_dir).resolve(), ex_root, ref_root)

    print("[PARAM]", {
        "extracted_root": str(ex_root),
        "reference_root": str(ref_root),
        "out_json": args.out_json,
        "mapping": args.mapping,
        "phash_threshold": args.phash_threshold,
        "hist_threshold": args.hist_threshold,
        "cache_dir": args.cache_dir,
        "workers": args.workers,
    })

    # 매핑 로드 (단일 진실원)
    with open(args.mapping, encoding="utf-8") as f:
        mp = json.load(f)  # {processed_name: {...}}

    # (track,channel) 단위로 분리
    extracted: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    reference: Dict[Tuple[str, str], List[Path]] = defaultdict(list)

    for proc_name, meta in mp.items():
        track = str(meta.get("트랙", "")).lower()
        ch = _norm_channel(str(meta.get("채널", "")))
        if not track or ch not in {"gray", "color"}:
            continue
        cat = _normalize_category(meta.get("카테고리"), meta.get("원본_전체_경로"))
        root = ex_root if cat == "extracted" else ref_root
        full = root / track / ch / proc_name
        if full.is_file():
            (extracted if cat == "extracted" else reference)[(track, ch)].append(full)

    # identity → processed filename 조회 테이블
    id2name: Dict[Tuple[str, str, str], str] = {
        (m["원본_전체_경로"], str(m["트랙"]).lower(), _norm_channel(str(m["채널"]))): n
        for n, m in mp.items()
    }

    out: Dict[str, Dict] = {}
    basis_cnt = cand_cnt = 0

    # 트랙 분리 처리(low/high 혼합 금지)
    for track in ("low", "high"):
        gray_ext = extracted.get((track, "gray"), [])
        gray_ref = reference.get((track, "gray"), [])
        if not gray_ext or not gray_ref:
            continue

        # 작은 쪽을 기준(basis)으로 삼음
        if len(gray_ref) <= len(gray_ext):
            basis_gray, basis_side = gray_ref, "reference"
            compare_gray = gray_ext
            compare_color_dir = ex_root / track / "color"
        else:
            basis_gray, basis_side = gray_ext, "extracted"
            compare_gray = gray_ref
            compare_color_dir = ref_root / track / "color"

        print(f"[TRACK] {track}: basis={basis_side} gray={len(basis_gray)} vs other={len(compare_gray)}")

        # pHash 일괄(스레드+캐시)
        h_basis = _batch_phash(basis_gray, cache, args.workers)
        h_compare = _batch_phash(compare_gray, cache, args.workers)

        # 비교측 컬러 히스토그램(파일명/그레이동명이름 키)
        hist_compare: Dict[str, np.ndarray] = {}
        if compare_color_dir.is_dir():
            color_files = [p for p in compare_color_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
            color_hists = _batch_hist(color_files, cache, args.workers)
            for c_name, hv in color_hists.items():
                hist_compare[c_name] = hv
                meta = mp.get(c_name)
                if meta:
                    gname = id2name.get((meta["원본_전체_경로"], track, "gray"))
                    if gname:
                        hist_compare[gname] = hv
        else:
            print(f"[WARN] Missing color dir: {compare_color_dir} — histogram step will be sparse.")

        # 기준 이미지별 후보 탐색
        for bpath, bhash in h_basis.items():
            basis_cnt += 1

            b_meta = mp.get(bpath.name)
            if not b_meta:
                continue
            mid = b_meta["원본_전체_경로"]

            entry = out.setdefault(mid, {"track": track, "candidates": []})

            # 기준의 컬러 히스토그램
            c_name = id2name.get((mid, track, "color"))
            if not c_name:
                continue
            c_root = ref_root if basis_side == "reference" else ex_root
            c_path = c_root / track / "color" / c_name
            if not c_path.is_file():
                continue
            basis_hist = cache.get_hist(c_path)

            # pHash 거리 오름차순으로 필터링
            pdists: List[Tuple[Path, int]] = []
            for cp, chash in h_compare.items():
                try:
                    d = int(bhash - chash)
                except Exception:
                    with Image.open(cp) as im:
                        d = int(bhash - imagehash.phash(im))
                pdists.append((cp, d))
            pdists.sort(key=lambda x: x[1])

            for cp, pdist in pdists:
                if pdist > args.phash_threshold:
                    break

                c_meta = mp.get(cp.name)
                if not c_meta:
                    continue
                cand_id = c_meta["원본_전체_경로"]

                # 동일 원본은 스킵
                if cand_id == mid:
                    continue

                hvec = hist_compare.get(cp.name)
                if hvec is None:
                    continue

                hdist = float(cv2.compareHist(basis_hist, hvec, cv2.HISTCMP_BHATTACHARYYA))
                if hdist <= args.hist_threshold:
                    entry["candidates"].append(
                        {"name": cand_id, "phash": int(pdist), "hist": round(hdist, 4)}
                    )
                    cand_cnt += 1

    # 저장
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {args.out_json} | basis={basis_cnt:,}  candidates={cand_cnt:,}")


if __name__ == "__main__":
    main()
