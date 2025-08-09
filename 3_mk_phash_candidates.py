# 3_mk_phash_candidates.py
"""
pHash + HSV 히스토그램 기반 후보군 생성 (CPU-only)

입력:
  processed/
    extracted/{low|high}/{gray|color}/<name>.png
    reference/{low|high}/{gray|color}/<name>.png
  preprocess_mapping.json  (키: '트랙','채널','카테고리','원본_전체_경로')

출력 (기본: 트랙 분리 키, low/high 혼합 방지):
  out_json:
    {
      "low|<원본_전체_경로>": {
        "original_path": "<원본_전체_경로>",
        "track": "low",
        "candidates": [
          {"name": "<원본_전체_경로>", "phash": <int>, "hist": <float>},
          ...
        ]
      },
      "high|<원본_전체_경로>": { ... }
    }

CLI 예:
  python 3_mk_phash_candidates.py processed/extracted processed/reference out/candidates.json \
         --mapping preprocess_mapping.json --phash-threshold 28 --hist-threshold 0.30 \
         --cache-dir .cache/features --workers 8 [--key-mode track_path|path] [--no-dedup-within-basis]
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
      phash/{extracted|reference}/{low|high}/gray/<name>.txt  (16-hex)
      hist/{extracted|reference}/{low|high}/color/<name>.npy  (96D float32)
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
        try:
            rel = p.resolve().relative_to(self.ref_root.resolve())
            return "reference", rel.parts[0], rel.parts[1], name
        except Exception:
            pass
        raise ValueError(f"unknown root: {p}")

    # ----- pHash -----
    def get_phash(self, p: Path) -> imagehash.ImageHash:
        cat, track, ch, name = self._rel_info(p)
        if ch != "gray":
            raise ValueError("pHash는 gray 채널만 지원")
        cfile = self.base / "phash" / cat / track / ch / (name + ".txt")
        try:
            with open(cfile, "r", encoding="utf-8") as f:
                return imagehash.hex_to_hash(f.read().strip())
        except Exception:
            pass
        with Image.open(p) as im:
            h = imagehash.phash(ImageOps.exif_transpose(im))
        try:
            cfile.parent.mkdir(parents=True, exist_ok=True)
            with open(cfile, "w", encoding="utf-8") as f:
                f.write(str(h))
        except Exception:
            pass
        return h

    # ----- HSV hist (color) -----
    def get_hist(self, p: Path) -> np.ndarray:
        cat, track, ch, name = self._rel_info(p)
        if ch != "color":
            raise ValueError("HSV hist는 color 채널만 지원")
        cfile = self.base / "hist" / cat / track / ch / (name + ".npy")
        try:
            v = np.load(cfile)
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
            hist = cv2.calcHist([hsv], [c], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            feats.append(hist.flatten())
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
                    pass  # skip unreadable
    return out


def _batch_hist(paths: List[Path], cache: FSCache, workers: int) -> Dict[str, np.ndarray]:
    """
    반환: basename -> 96D float32.
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
    p.add_argument("--key-mode", choices=["track_path","path"], default="track_path",
                   help='basis key: "track|<원본_전체_경로>" (default) or "<원본_전체_경로>" (legacy, not recommended)')
    p.add_argument("--dedup-within-basis", dest="dedup_within_basis", action="store_true", default=True,
                   help="basis 내 동일 candidate name은 (min hist, then min phash) 기준으로 1개만 유지")
    p.add_argument("--no-dedup-within-basis", dest="dedup_within_basis", action="store_false")
    return p.parse_args()


# ---- Key utility ----
def _make_basis_id(mid: str, track: str, mode: str) -> str:
    """Return unique basis ID. Default: "track|<원본_전체_경로>" to avoid low/high collapsing."""
    if mode == "path":
        return mid
    return f"{track}|{mid}"


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
        "key_mode": args.key_mode,
        "dedup_within_basis": args.dedup_within_basis,
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

        # --- 핵심 수정: compare_gray → 정확한 color 파일명 역추적 후 히스토그램 구축 ---
        needed_color_files: List[Path] = []
        gray_to_color: Dict[str, str] = {}  # gray파일명 -> color파일명

        for gp in compare_gray:  # gp.name 예: 000123_g.png
            meta = mp.get(gp.name)
            if not meta:
                continue
            cid = meta.get("원본_전체_경로")
            if not cid:
                continue
            cname = id2name.get((cid, track, "color"))  # 대응 color 파일명 (예: 000123.png)
            if not cname:
                continue
            needed_color_files.append(compare_color_dir / cname)
            gray_to_color[gp.name] = cname

        color_hists = _batch_hist(needed_color_files, cache, args.workers)  # key: color파일명 -> 벡터

        # gray·color 양쪽 키로 조회 가능하도록 사전 구성
        hist_compare: Dict[str, np.ndarray] = {}
        for gname, cname in gray_to_color.items():
            hv = color_hists.get(cname)
            if hv is not None:
                hist_compare[gname] = hv   # cp.name(=gray)로 조회
                hist_compare[cname] = hv   # 필요시 color명으로도 조회

        print(f"[HIST] {track}: compare_gray={len(compare_gray)}  map_ok={len(gray_to_color)}  "
              f"hist_vecs={len(color_hists)}")

        # 기준 이미지별 후보 탐색
        track_phash_pass = 0
        track_hist_pass = 0

        for bpath, bhash in h_basis.items():
            basis_cnt += 1

            b_meta = mp.get(bpath.name)
            if not b_meta:
                continue
            mid = b_meta["원본_전체_경로"]

            basis_id = _make_basis_id(mid, track, args.key_mode)
            entry = out.setdefault(basis_id, {"track": track, "original_path": mid, "candidates": []})

            # 기준(color) 히스토그램
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
                    try:
                        with Image.open(cp) as im:
                            d = int(bhash - imagehash.phash(im))
                    except Exception:
                        continue
                pdists.append((cp, d))
            pdists.sort(key=lambda x: x[1])

            for cp, pdist in pdists:
                if pdist > args.phash_threshold:
                    break
                track_phash_pass += 1

                c_meta = mp.get(cp.name)
                if not c_meta:
                    continue
                cand_id = c_meta["원본_전체_경로"]

                # 동일 원본은 스킵
                if cand_id == mid:
                    continue

                # 히스토그램 조회 (gray 이름 우선 → color 이름 백업)
                hvec = hist_compare.get(cp.name)
                if hvec is None:
                    cname = id2name.get((cand_id, track, "color"))
                    if cname:
                        hvec = hist_compare.get(cname)
                if hvec is None:
                    continue  # 여전히 없으면 스킵

                hdist = float(cv2.compareHist(basis_hist, hvec, cv2.HISTCMP_BHATTACHARYYA))
                if hdist > args.hist_threshold:
                    continue
                track_hist_pass += 1

                new_c = {"name": cand_id, "phash": int(pdist), "hist": round(hdist, 4)}
                if args.dedup_within_basis:
                    # dedup by candidate name; keep best (min hist, then min phash)
                    replaced = False
                    for i, _c in enumerate(entry["candidates"]):
                        if _c["name"] == cand_id:
                            if (new_c["hist"], new_c["phash"]) < (_c["hist"], _c["phash"]):
                                entry["candidates"][i] = new_c
                            replaced = True
                            break
                    if not replaced:
                        entry["candidates"].append(new_c)
                        cand_cnt += 1
                else:
                    entry["candidates"].append(new_c)
                    cand_cnt += 1

        print(f"[PASS] {track}: pHash_pass={track_phash_pass:,}  hist_pass={track_hist_pass:,}")

    # 저장
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {args.out_json} | basis={basis_cnt:,}  candidates={cand_cnt:,}")


if __name__ == "__main__":
    main()
