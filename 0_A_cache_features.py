# 0_A_cache_features.py
"""
HWP 이미지 매칭 파이프라인용 특징 캐시 (CPU-only)

입력(전처리 산출물):
  processed/
    extracted/{low|high}/{gray|color}/<name>.png
    reference/{low|high}/{gray|color}/<name>.png
  preprocess_mapping.json  (키: '트랙','채널','카테고리','원본_전체_경로')

출력:
  {out-dir}/
    phash/{extracted|reference}/{low|high}/gray/<name>.txt
    hist/{extracted|reference}/{low|high}/color/<name>.npy
    orb/{extracted|reference}/{low|high}/{gray|color}/<name>.npz
    index.json

주의:
  • 매핑 JSON의 "카테고리" 레이블을 우선 사용(없으면 'BinData' 휴리스틱).
  • 레거시 "original" 레이블은 자동으로 "reference"로 정규화.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import cv2
from PIL import Image, ImageOps
import imagehash

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
except Exception:  # pragma: no cover
    ProcessPoolExecutor = None  # type: ignore

WhichT = Literal["all", "gray", "color"]
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


# ---------------- Args ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache pHash/HSV/ORB features for processed images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("extracted_root", help="processed/extracted")
    p.add_argument("reference_root", help="processed/reference")
    p.add_argument("--mapping", required=True, help="preprocess_mapping.json")
    p.add_argument("--out-dir", default=".cache/features")
    p.add_argument(
        "--which",
        choices=["all", "gray", "color"],
        default="all",
        help="feature subset (gray=pHash+ORB, color=HSV)",
    )
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--force", action="store_true", help="recompute even if cache exists")
    p.add_argument("--orb-nfeatures", type=int, default=1500)
    return p.parse_args()


# ---------------- IO helpers ----------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_imread_gray(p: Path) -> np.ndarray | None:
    """안전한 그레이스케일 로더(실패 시 None)."""
    if not p.is_file():
        return None
    try:
        buf = np.fromfile(str(p), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def _pil_open_rgb(p: Path) -> np.ndarray | None:
    """PIL 기반 RGB 로더(실패 시 None)."""
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            return np.array(im.convert("RGB"))
    except Exception:
        return None


# ---------------- Feature extractors ----------------
def compute_phash_hex(img_path: Path) -> str:
    with Image.open(img_path) as im:
        h = imagehash.phash(im)  # 64-bit
    return str(h)  # 16-hex


def compute_hsv_hist_96(rgb: np.ndarray) -> np.ndarray:
    """정규화 96D HSV 히스토그램(32bins×3ch)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for c in range(3):
        hist = cv2.calcHist([hsv], [c], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.flatten())
    return np.concatenate(feats).astype(np.float32)


def compute_orb(gray: np.ndarray, nfeatures: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """(keypoints Nx2 float32, descriptors Nx32 uint8)"""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, des = orb.detectAndCompute(gray, None)
    pts = np.float32([kp.pt for kp in kps]) if kps else np.empty((0, 2), np.float32)
    des = des if des is not None else np.empty((0, 32), np.uint8)
    return pts, des


# ---------------- Data model ----------------
@dataclass(frozen=True)
class Item:
    category: str   # "extracted" | "reference"
    track: str      # "low" | "high"
    channel: str    # "gray" | "color"
    name: str       # filename (e.g., 000123.png)
    path: Path      # full path


def _normalize_category(cat: str | None, origin_path: str | None) -> str:
    """매핑 레이블 우선 → 레거시/휴리스틱 보정."""
    c = (cat or "").strip().lower()
    if c == "original":
        c = "reference"
    if c in {"extracted", "reference"}:
        return c
    op = (origin_path or "").lower()
    if "bindata" in op:
        return "extracted"
    return "reference"


def _norm_channel(ch: str) -> str:
    t = ch.lower().split("_")[0]
    return "gray" if t.startswith(("gray", "grey", "그레이")) else "color"


def build_item_list(ex_root: Path, ref_root: Path, mapping_json: Path) -> list[Item]:
    mp = json.loads(mapping_json.read_text(encoding="utf-8"))
    items: list[Item] = []
    for fname, meta in mp.items():
        track = str(meta.get("트랙", "")).lower()
        ch = _norm_channel(str(meta.get("채널", "")))
        if not track or ch not in {"gray", "color"}:
            continue
        cat = _normalize_category(meta.get("카테고리"), meta.get("원본_전체_경로"))
        root = ex_root if cat == "extracted" else ref_root
        p = root / track / ch / fname
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        items.append(Item(cat, track, ch, fname, p))
    return items


# ---------------- Worker ----------------
def _dest_paths(base: Path, it: Item) -> dict[str, Path]:
    return {
        "phash": base / "phash" / it.category / it.track / "gray" / (it.name + ".txt"),
        "hist":  base / "hist"  / it.category / it.track / "color" / (it.name + ".npy"),
        "orb":   base / "orb"   / it.category / it.track / it.channel / (it.name + ".npz"),
    }


def worker(it: Item, out_dir: Path, which: WhichT, force: bool, nfeatures: int) -> tuple[str, bool, str]:
    """
    결과: (relative_id, success, message)
    relative_id = f"{it.category}/{it.track}/{it.channel}/{it.name}"
    """
    rel_id = f"{it.category}/{it.track}/{it.channel}/{it.name}"
    dest = _dest_paths(out_dir, it)
    try:
        # pHash: gray만
        if which in ("all", "gray") and it.channel == "gray":
            _ensure_dir(dest["phash"].parent)
            if force or not dest["phash"].is_file():
                ph = compute_phash_hex(it.path)
                dest["phash"].write_text(ph, encoding="utf-8")

        # HSV: color만
        if which in ("all", "color") and it.channel == "color":
            _ensure_dir(dest["hist"].parent)
            if force or not dest["hist"].is_file():
                rgb = _pil_open_rgb(it.path)
                if rgb is None:
                    return rel_id, False, "read-fail-rgb"
                hv = compute_hsv_hist_96(rgb)
                np.save(dest["hist"], hv)

        # ORB: 양 채널 허용(보통 gray면 충분)
        if which in ("all", "gray") and it.channel in ("gray", "color"):
            _ensure_dir(dest["orb"].parent)
            if force or not dest["orb"].is_file():
                gray = _safe_imread_gray(it.path)
                if gray is None:
                    return rel_id, False, "read-fail-gray"
                kps, des = compute_orb(gray, nfeatures=nfeatures)
                np.savez_compressed(dest["orb"], kps=kps, des=des)

        return rel_id, True, "ok"
    except Exception as e:
        return rel_id, False, f"error:{e}"


# ---------------- Main ----------------
def main() -> None:
    args = parse_args()
    ex_root = Path(args.extracted_root).resolve()
    ref_root = Path(args.reference_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    items = build_item_list(ex_root, ref_root, Path(args.mapping))
    if args.which == "gray":
        items = [it for it in items if it.channel == "gray"]
    elif args.which == "color":
        items = [it for it in items if it.channel == "color"]

    print(f"[PARAM] extracted_root : {ex_root}")
    print(f"[PARAM] reference_root : {ref_root}")
    print(f"[PARAM] mapping        : {args.mapping}")
    print(f"[PARAM] out_dir        : {out_dir}")
    print(f"[PARAM] which          : {args.which}")
    print(f"[INFO ] total candidates: {len(items):,}")

    ok = ng = 0
    logs: list[tuple[str, str]] = []

    if ProcessPoolExecutor and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, it, out_dir, args.which, args.force, args.orb_nfeatures) for it in items]
            for f in as_completed(futs):
                rel, succ, msg = f.result()
                if succ:
                    ok += 1
                else:
                    ng += 1
                    logs.append((rel, msg))
    else:
        for it in items:
            rel, succ, msg = worker(it, out_dir, args.which, args.force, args.orb_nfeatures)
            if succ:
                ok += 1
            else:
                ng += 1
                logs.append((rel, msg))

    index = {
        "roots": {"extracted": str(ex_root), "reference": str(ref_root)},
        "out_dir": str(out_dir),
        "which": args.which,
        "counts": {"total": len(items), "ok": ok, "fail": ng},
        "fails": [{"id": r, "reason": m} for r, m in logs[:200]],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] ok={ok:,} fail={ng:,} → {out_dir/'index.json'}")


if __name__ == "__main__":
    main()
