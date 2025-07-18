# hwp_image_mapper.py
"""
HWP/HWPX → 이미지 추출·매핑(Color‑Hist + ORB + EdgeSSIM + RANSAC)
"""
from __future__ import annotations

import argparse
import base64
import csv
import random
import shutil
import time
import zlib
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import olefile
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from hwp_extract import HWPExtractor           # 외부 모듈
from utils import IMG_EXTS, Tee                # 공통 유틸

# ─────────────── 가중치/상수 ───────────────────────────────────
W_COLOR = 0.3
W_ORB   = 0.4
W_EDGE  = 0.3
RATIO   = 0.75

_ORB        = cv2.ORB_create(500)
_EDGE_SIZE  = (256, 256)
_HB, _SB    = 50, 60

# ─────────────── 데이터 구조 ───────────────────────────────────
@dataclass(slots=True)
class ImgFeat:
    color_hist: np.ndarray
    orb_kp    : list[cv2.KeyPoint]
    orb_desc  : np.ndarray | None
    edge      : np.ndarray

# ─────────────── 도우미 함수 ───────────────────────────────────
def detect_type(hwp: Path) -> str:
    """헤더 서명으로 HWP5 / HWPX 구분"""
    with hwp.open("rb") as f:
        sig = f.read(4)
    return "hwpx" if sig == b"PK\x03\x04" else "hwp5"


def compute_features(fp: Path) -> ImgFeat:
    """Color‑Hist, ORB, Edge 특징 추출"""
    arr = np.frombuffer(fp.read_bytes(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:  # 대안: PIL
        img = np.array(Image.open(fp).convert("RGB"))[:, :, ::-1]

    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_HB, _SB], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    kp, desc = _ORB.detectAndCompute(img, None)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(cv2.resize(gray, _EDGE_SIZE), 100, 200)

    return ImgFeat(hist, kp, desc, edge)


def color_score(h1, h2) -> float:
    return (cv2.compareHist(h1.astype("float32"),
                            h2.astype("float32"),
                            cv2.HISTCMP_CORREL) + 1) / 2


def orb_score(kp1, d1, kp2, d2) -> float:
    """ORB 매칭 + RANSAC 인라이어 비율"""
    if d1 is None or d2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < RATIO * n.distance]
    if not good:
        return 0.0

    # 좌표가 없는 경우는 매칭 비율만
    if not kp1 or not kp2:
        return len(good) / max(len(matches), 1)

    if len(good) > 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            return int(mask.sum()) / len(good)
    return len(good) / max(len(matches), 1)


def edge_score(e1, e2) -> float:
    return ssim(e1, e2, data_range=255)


def sim_scores(f1: ImgFeat, f2: ImgFeat):
    c = color_score(f1.color_hist, f2.color_hist)
    o = orb_score(f1.orb_kp, f1.orb_desc, f2.orb_kp, f2.orb_desc)
    e = edge_score(f1.edge, f2.edge)
    return c, o, e, W_COLOR * c + W_ORB * o + W_EDGE * e


# ─────────────── 이미지 추출 ───────────────────────────────────
def extract_images(hwp: Path, out_dir: Path, sample: int | None = None) -> list[Path]:
    """HWP/HWPX 내부 이미지를 out_dir에 저장"""
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = detect_type(hwp)
    extracted: list[Path] = []

    if kind == "hwpx":
        with zipfile.ZipFile(hwp) as zf:
            names = [n for n in zf.namelist() if n.startswith("Contents/Resources/")]
            for i, n in enumerate(names, 1):
                if "PrvImage" in n or "preview" in n.lower():
                    continue
                (p := out_dir / f"{i:03d}_{Path(n).name}").write_bytes(zf.read(n))
                extracted.append(p)
                print(f"  - {p.name}")
                if sample and len(extracted) >= sample:
                    break
    else:
        doc = HWPExtractor(data=hwp.read_bytes())
        for i, obj in enumerate(doc.extract_files(), 1):
            if "PrvImage" in obj.name or "preview" in obj.name.lower():
                continue
            data = obj.data
            for wb in (None, -zlib.MAX_WBITS):
                try:
                    data = zlib.decompress(data, wb) if wb is not None else zlib.decompress(data)
                    break
                except zlib.error:
                    pass
            img_data = None
            try:
                Image.open(BytesIO(data)).verify()
                img_data = data
            except Exception:
                try:
                    ole = olefile.OleFileIO(BytesIO(data))
                    raw = ole.openstream("Ole10Native").read()
                    idx = raw.find(b"\xff\xd8\xff")
                    img_data = raw[idx:] if idx != -1 else None
                except Exception:
                    img_data = None
            if img_data:
                (p := out_dir / f"{i:03d}_{obj.name.replace('/', '_')}").write_bytes(img_data)
                extracted.append(p)
                print(f"  - {p.name}")
                if sample and len(extracted) >= sample:
                    break
    return extracted


# ─────────────── DB 구축 & 매칭 ─────────────────────────────────
def precompute_db(root: Path, sample: int | None = None) -> dict[Path, ImgFeat]:
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if sample:
        imgs = random.sample(imgs, min(sample, len(imgs)))

    db: dict[Path, ImgFeat] = {}
    for fp in imgs:
        try:
            db[fp] = compute_features(fp)
            print(f"[DB] {fp.name}")
        except Exception as e:
            print(f"[WARN] {fp} ({e})")
    return db


def match_images(extracted: list[Path], db: dict[Path, ImgFeat]) -> list[tuple]:
    if not db:
        print("[WARN] 원본 DB가 비어 있습니다. 매칭을 생략합니다.")
        return []

    rows = []
    for img in extracted:
        f = compute_features(img)
        best_fp: Path | None = None
        best_sc: tuple[float, float, float] | None = None
        best_ov = -1.0
        for fp, of in db.items():
            c, o, e, ov = sim_scores(f, of)
            if ov > best_ov:
                best_fp, best_sc, best_ov = fp, (c, o, e), ov
        c, o, e = best_sc if best_sc else (0.0, 0.0, 0.0)  # type: ignore
        print(f"[MATCH] {img.name} → {best_fp.name if best_fp else '---'} (Overall={best_ov:.4f})")
        rows.append(
            (
                img.name,
                str(img),
                str(best_fp) if best_fp else "",
                "Color+ORB+Edge",
                f"{c:.4f}",
                f"{o:.4f}",
                f"{e:.4f}",
                f"{best_ov:.4f}",
            )
        )
    return rows


# ─────────────── 특징 직렬화 / 복원 ─────────────────────────────
def feat_to_json(feat: ImgFeat) -> dict:
    edge_bits = np.packbits((feat.edge > 0).astype(np.uint8))
    kp_coords = [(float(k.pt[0]), float(k.pt[1])) for k in feat.orb_kp]
    return {
        "color": feat.color_hist.tolist(),
        "kp": kp_coords,
        "orb": feat.orb_desc.tolist() if feat.orb_desc is not None else None,
        "edge": base64.b64encode(edge_bits).decode(),
    }


def json_to_feat(d: dict) -> ImgFeat:
    ch       = np.array(d["color"], np.float32)
    orb_desc = np.array(d["orb"], np.uint8) if d["orb"] is not None else None

    kp_coords = d.get("kp", [])
    orb_kp = [cv2.KeyPoint(x, y, 1) for x, y in kp_coords]

    bits = np.frombuffer(base64.b64decode(d["edge"]), dtype=np.uint8)
    edge = (
        np.unpackbits(bits)[: _EDGE_SIZE[0] * _EDGE_SIZE[1]]
        .reshape(_EDGE_SIZE)
        .astype(np.uint8)
        * 255
    )

    return ImgFeat(ch, orb_kp, orb_desc, edge)


# ─────────────── CLI ───────────────────────────────────────────
def main() -> None:
    pa = argparse.ArgumentParser(description="HWP ↔ 사진 매핑(Color+ORB+EdgeSSIM)")
    pa.add_argument("hwp", help="HWP/HWPX 파일 경로")
    pa.add_argument("photos_root", help="원본 이미지 폴더 경로")
    pa.add_argument("--sample", "-s", type=int, help="추출/DB 샘플 개수 제한")
    pa.add_argument("--fast", action="store_true", help="색상 히스토그램만 사용하여 빠르게 실행")
    args = pa.parse_args()

    # 로그 파일
    log_dir = Path(__file__).parent / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tee(log_dir / "hwp_image_mapper_log.txt")

    print(f"[START] HWP 파일     : {args.hwp}")
    print(f"[START] 사진 루트    : {args.photos_root}")
    print(f"[START] 샘플 제한    : {args.sample if args.sample else '없음'}")
    print(f"[START] FAST 모드    : {args.fast}")
    t0 = time.time()

    global W_COLOR, W_ORB, W_EDGE
    if args.fast:
        W_COLOR, W_ORB, W_EDGE = 1.0, 0.0, 0.0
        print("[INFO] FAST 모드: Color 히스토그램만 사용합니다.")

    tmp = Path("_tmp_extract")
    shutil.rmtree(tmp, ignore_errors=True)
    extracted = extract_images(Path(args.hwp), tmp, args.sample)
    db = precompute_db(Path(args.photos_root), args.sample)

    rows = match_images(extracted, db)
    out_csv = Path("image_mapping.csv")
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["문서이미지명", "추출파일", "원본경로", "방식", "Color", "ORB", "EdgeSSIM", "Overall"]
            )
            writer.writerows(rows)
        print(f"[DONE] {out_csv} 저장 ({len(rows)}행, {time.time() - t0:.1f}s)")
    else:
        print("[DONE] 매핑 결과가 없어 CSV를 만들지 않았습니다.")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
