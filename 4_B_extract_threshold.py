# 4_B_extract_threshold.py
"""
ORB + RANSAC 임계값 추정기 (간소화 버전)

- 입력: candidates.json, preprocess_mapping.json, (옵션) hand_label.csv
- 출력: outputs/match_scores.csv, outputs/thresholds.json, orb_hist.png, orb_cdf.png

예)
python 4_B_extract_threshold.py ^
  --candidates-json candidates.json ^
  --mapping preprocess_mapping.json ^
  --images-root processed ^
  --out-dir outputs ^
  --pos-mode top1 ^
  --neg-mode hard ^
  --neg-ratio 1.0
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory
from tqdm import tqdm
from scipy.stats import beta

# 선택 의존성: 없으면 분위수로 폴백
try:
    from kneed import KneeLocator  # type: ignore
except Exception:
    KneeLocator = None

cv2.setNumThreads(0)
CACHE = Memory(".cache/descriptors", verbose=0)

# ─────────────── I/O & 경로 ───────────────
def norm_identity(p: str) -> str:
    return os.path.normpath(p).lower()

def safe_imread(path: str, flag=cv2.IMREAD_GRAYSCALE):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, flag)
    if img is not None:
        return img
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flag)
    except Exception:
        return None

# ─────────────── ORB + RANSAC ───────────────
@CACHE.cache
def extract_orb(path: str, n: int = 1500):
    img = safe_imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.empty((0, 2), dtype=np.float32), None
    orb = cv2.ORB_create(nfeatures=n)
    kps, des = orb.detectAndCompute(img, None)
    pts = np.float32([kp.pt for kp in kps]) if kps else np.empty((0, 2), dtype=np.float32)
    return pts, des

def orb_score(a: str, b: str, nfeatures: int = 1500, ratio: float = 0.75, ransac_thresh: float = 5.0) -> Tuple[int, int, int]:
    ptsA, desA = extract_orb(a, n=nfeatures)
    ptsB, desB = extract_orb(b, n=nfeatures)
    if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
        return 0, len(ptsA), len(ptsB)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desA, desB, k=2)
    good = [m for m, n in matches if n is not None and m.distance < ratio * n.distance]
    if len(good) < 4:
        return 0, len(ptsA), len(ptsB)

    src = np.float32([ptsA[m.queryIdx] for m in good])
    dst = np.float32([ptsB[m.trainIdx] for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inl = int(mask.sum()) if mask is not None else 0
    return inl, len(ptsA), len(ptsB)

def knee_or_quantile(values: List[float], q: float = 0.97) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if KneeLocator is not None and len(xs) >= 10:
        cdf = np.linspace(0, 1, len(xs))
        try:
            k = KneeLocator(xs, cdf, curve="concave", direction="increasing")
            if k.knee is not None:
                return float(k.knee)
        except Exception:
            pass
    return float(np.quantile(xs, q))

# ─────────────── 매핑/전처리 선택 ───────────────
def build_lookup(pre_map: Dict, images_root: Path) -> Dict[str, Dict[Tuple[str, str, str], str]]:
    """
    identity(원본 풀 경로 정규화) -> {(category, track, channel): relative processed path}
    """
    lut: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for proc_name, meta in pre_map.items():
        origin = meta.get("원본_전체_경로", "")
        track = str(meta.get("트랙", "")).lower()
        ch = str(meta.get("채널", "")).lower()
        cat = "extracted" if "BinData" in origin else "original"
        rel = f"{cat}/{track}/{ch}/{proc_name}"
        lut.setdefault(norm_identity(origin), {})[(cat, track, ch)] = rel
    return lut

def pick_preprocessed(identity_path: str, track: str, prefer_gray: bool,
                      lut: Dict[str, Dict[Tuple[str, str, str], str]], images_root: Path) -> str:
    """
    동일 identity 내에서 (category 동일 → track 동일 → 채널 선호) 우선, 없으면 원본 경로 사용.
    """
    key = norm_identity(identity_path)
    options = lut.get(key, {})
    if not options:
        return identity_path

    order = ["gray", "color"] if prefer_gray else ["color", "gray"]
    cat = "extracted" if "BinData" in identity_path else "original"

    for ch in order:
        tup = (cat, track, ch)
        if tup in options:
            return str(images_root / options[tup])

    for (c2, t2, ch2), rel in options.items():
        if t2 == track and ch2 in order:
            return str(images_root / rel)

    return identity_path

# ─────────────── 페어 구성 ───────────────
def load_positive_pairs(cand_json: Dict, mode: str = "top1") -> List[Tuple[str, str, str]]:
    """
    candidates.json → [(extracted_identity, original_identity, track)]
    """
    pos: List[Tuple[str, str, str]] = []
    for origin_path, info in cand_json.items():
        track = str(info.get("track", "low")).lower()
        items = info.get("candidates", info.get("extracted_candidates", [])) or []
        chosen = [items[0]] if (mode == "top1" and items) else items
        for c in chosen:
            name = c.get("name")
            if not name:
                continue
            a, b = origin_path, name
            if ("BinData" in a) == ("BinData" in b):
                continue
            ex = a if "BinData" in a else b
            ori = b if "BinData" not in b else a
            pos.append((ex, ori, track))
    return pos

def sample_negatives(pos_pairs: List[Tuple[str, str, str]], ratio: float = 1.0,
                     mode: str = "hard", seed: int = 0) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    ex_ids = sorted({ex for ex, _, _ in pos_pairs})
    or_ids = sorted({ori for _, ori, _ in pos_pairs})
    ex2tracks: Dict[str, List[str]] = {}
    for ex, _, tr in pos_pairs:
        ex2tracks.setdefault(ex, [])
        if tr not in ex2tracks[ex]:
            ex2tracks[ex].append(tr)

    pos_set = {(ex, ori) for ex, ori, _ in pos_pairs}
    need = int(len(pos_pairs) * ratio)
    mult = 3 if mode == "hard" else 1
    cand: List[Tuple[str, str, str]] = []

    while len(cand) < need * mult and ex_ids and or_ids:
        ex = rng.choice(ex_ids)
        ori = rng.choice(or_ids)
        if (ex, ori) in pos_set:
            continue
        tr = rng.choice(ex2tracks.get(ex, ["low"]))
        cand.append((ex, ori, tr))

    uniq = list(dict.fromkeys(cand))
    return uniq[:need]

# ─────────────── CLI ───────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Estimate ORB+RANSAC threshold from candidates & preprocessed images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--mapping", default="preprocess_mapping.json")
    ap.add_argument("--images-root", default="processed")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--labels-csv", default=None,
                    help="CSV columns: orig_path, extracted_path, label(1/0)")
    ap.add_argument("--pos-mode", choices=["top1", "all"], default="top1")
    ap.add_argument("--neg-mode", choices=["shuffle", "hard"], default="hard")
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--target-tpr", type=float, default=0.97)
    ap.add_argument("--orb-nfeatures", type=int, default=1500)
    ap.add_argument("--lowe-ratio", type=float, default=0.75)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

# ─────────────── Main ───────────────
def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cand_json = json.loads(Path(args.candidates_json).read_text("utf-8"))
    pre_map = json.loads(Path(args.mapping).read_text("utf-8"))
    images_root = Path(args.images_root)

    lut = build_lookup(pre_map, images_root)
    pos = load_positive_pairs(cand_json, args.pos_mode)
    neg = sample_negatives(pos, ratio=args.neg_ratio, mode=args.neg_mode, seed=args.seed)
    print(f"[INFO] positives: {len(pos):,} , negatives: {len(neg):,}")

    rows: List[Dict] = []
    miss = 0
    all_pairs = [("P",) + p for p in pos] + [("N",) + p for p in neg]
    random.Random(args.seed).shuffle(all_pairs)

    for label, ex_id, or_id, track in tqdm(all_pairs, ncols=80, desc="ORB"):
        ex_path = pick_preprocessed(ex_id, track, True, lut, images_root)
        or_path = pick_preprocessed(or_id, track, True, lut, images_root)
        try:
            inl, kA, kB = orb_score(
                ex_path, or_path,
                nfeatures=args.orb_nfeatures,
                ratio=args.lowe_ratio,
                ransac_thresh=args.ransac_thresh,
            )
        except Exception:
            miss += 1
            continue

        score = inl / (kA + 1e-6)
        rows.append(dict(
            label=label,
            ex_id=Path(ex_id).name, or_id=Path(or_id).name,
            ex_path=ex_id, or_path=or_id,
            ex_used=str(ex_path), or_used=str(or_path),
            track=track,
            inliers=inl, kpA=kA, kpB=kB, score=float(score),
        ))

    print(f"[INFO] file-read failures: {miss}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "match_scores.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out / 'match_scores.csv'}  rows={len(df):,}")

    # ── 임계값 ──
    thr_info: Dict[str, float | None] = {"orb_score": None, "ci_low": None, "ci_high": None}

    if args.labels_csv and Path(args.labels_csv).is_file():
        lab = pd.read_csv(args.labels_csv, encoding="utf-8-sig")
        cols = {c.lower(): c for c in lab.columns}
        ocol = cols.get("orig_path", cols.get("orig"))
        ecol = cols.get("extracted_path", cols.get("extracted"))
        lcol = cols.get("label", cols.get("y"))
        if not (ocol and ecol and lcol):
            raise ValueError("labels-csv must contain columns: orig_path/extracted_path/label")
        lab = lab.rename(columns={ocol: "orig", ecol: "extracted", lcol: "label"})
        lab["orig_b"] = lab["orig"].apply(lambda s: Path(str(s)).name)
        lab["extr_b"] = lab["extracted"].apply(lambda s: Path(str(s)).name)

        d2 = df.copy()
        d2["orig_b"] = d2["or_id"]
        d2["extr_b"] = d2["ex_id"]
        merged = d2.merge(lab[["orig_b", "extr_b", "label"]], on=["orig_b", "extr_b"], how="left")
        merged["label"] = merged["label"].fillna(-1).astype(int)
        labeled = merged[merged["label"] >= 0].copy()

        if labeled.empty:
            print("[WARN] no labeled pairs matched; falling back to no-label mode.")
        else:
            y = (labeled["label"] == 1).astype(int).values
            s = labeled["score"].values
            thr = None
            for t in np.linspace(0, 1, 101):
                tp = int(((s >= t) & (y == 1)).sum())
                fn = int((y == 1).sum()) - tp
                tpr = (tp + 1) / (tp + fn + 2)  # Laplace smoothing
                if tpr >= args.target_tpr:
                    ci_low = beta.ppf(0.025, tp + 1, fn + 1)
                    ci_high = beta.ppf(0.975, tp + 1, fn + 1)
                    thr_info = {"orb_score": float(t), "ci_low": float(ci_low), "ci_high": float(ci_high)}
                    thr = t
                    break
            if thr is None:
                tp = int((y == 1).sum())
                fallback = float(np.quantile(s[y == 1], 0.05)) if (y == 1).any() else 0.0
                thr_info = {
                    "orb_score": fallback,
                    "ci_low": float(beta.ppf(0.025, 1, max(tp, 1))),
                    "ci_high": float(beta.ppf(0.975, 1, max(tp, 1))),
                }

    if thr_info["orb_score"] is None:
        pos_scores = df.loc[df["label"] == "P", "score"].tolist()
        thr_info = {"orb_score": knee_or_quantile(pos_scores, q=0.97), "ci_low": None, "ci_high": None}

    (out / "thresholds.json").write_text(json.dumps(thr_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[KEY] threshold = {thr_info['orb_score']:.4f}")

    # ── 플롯 ──
    plt.figure()
    plt.hist(df["score"].tolist(), bins=100, log=True)
    plt.xlabel("score (inliers / keypointsA)")
    plt.ylabel("freq (log)")
    plt.tight_layout()
    plt.savefig(out / "orb_hist.png", dpi=150)

    xs = sorted(df["score"].tolist())
    plt.figure()
    plt.plot(xs, np.linspace(0, 1, len(xs)))
    plt.axvline(float(thr_info["orb_score"]), ls="--")
    plt.xlabel("score")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(out / "orb_cdf.png", dpi=150)

    print("[DONE] analysis complete.")

if __name__ == "__main__":
    main()
