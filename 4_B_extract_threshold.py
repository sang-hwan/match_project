# 4_B_extract_threshold.py
"""
ORB + RANSAC 임계값 추정기 (정리판, label 충돌 수정반영)

- 요구 입력:
  1) candidates.json   : 각 타깃(=원본 identity)에 대한 후보(extracted) 목록.
     * 필수 필드: "original_path", "track"(low|high), "candidates":[{"name": <경로>}]
  2) preprocess_mapping.json : 전처리 메타. 필수 필드:
     - "원본_전체_경로", "카테고리"(extracted|reference), "트랙"(low|high), "채널"(gray|color)
  3) (선택) hand_label.csv : 컬럼 고정 ["orig_path","extracted_path","label"(1/0)]

- 출력:
  <out>/match_scores.csv, <out>/thresholds.json, <out>/orb_hist.png, <out>/orb_cdf.png

예)
python 4_B_extract_threshold.py ^
  --candidates-json .\candidates.json ^
  --mapping .\preprocess_mapping.json ^
  --images-root .\processed ^
  --out-dir .\feat_dist ^
  --pos-mode top1 ^
  --neg-mode hard ^
  --neg-ratio 1.0 ^
  --target-tpr 0.97
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory
from tqdm import tqdm
from scipy.stats import beta

cv2.setNumThreads(0)
CACHE = Memory(".cache/descriptors", verbose=0)

# ────────────────────────────── 유틸 ──────────────────────────────
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

# ─────────────────────── ORB + RANSAC 점수 ───────────────────────
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

    src = np.float32([ptsA[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ptsB[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, len(ptsA), len(ptsB)

def quantile_threshold(values: List[float], q: float = 0.97) -> float:
    if not values:
        return 0.0
    return float(np.quantile(sorted(values), q))

# ─────────────────────── 매핑 LUT 생성/조회 ───────────────────────
def build_lookup(pre_map: Dict, images_root: Path) -> Dict[str, Dict[Tuple[str, str, str], str]]:
    """
    identity(원본 풀 경로 정규화) -> {(category(extracted|reference), track, channel): relative processed path}
    ※ 카테고리/트랙/채널은 매핑 메타의 값을 그대로 사용(소문자 정규화)
    """
    lut: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    for proc_name, meta in pre_map.items():
        origin = meta.get("원본_전체_경로")
        cat = str(meta.get("카테고리")).lower()
        trk = str(meta.get("트랙")).lower()
        ch = str(meta.get("채널")).lower()

        if origin is None or cat not in ("extracted", "reference"):
            raise ValueError("preprocess_mapping.json: '원본_전체_경로' 또는 '카테고리(extracted|reference)'가 올바르지 않습니다.")

        rel = f"{cat}/{trk}/{ch}/{proc_name}"
        lut.setdefault(norm_identity(origin), {})[(cat, trk, ch)] = rel
    return lut

def category_of_identity(identity_path: str, track: str,
                         lut: Dict[str, Dict[Tuple[str, str, str], str]]) -> Optional[str]:
    """
    매핑 LUT에 등록된 카테고리로 판별.
    동일 identity에 대해 주어진 track에 매칭되는 카테고리가 있으면 반환.
    """
    key = norm_identity(identity_path)
    options = lut.get(key, {})
    cats = {c for (c, t, _), _rel in options.items() if t == track}
    if "extracted" in cats:
        return "extracted"
    if "reference" in cats:
        return "reference"
    return None

def pick_preprocessed(identity_path: str, track: str, prefer_gray: bool,
                      lut: Dict[str, Dict[Tuple[str, str, str], str]], images_root: Path) -> str:
    """
    동일 identity 내에서 (카테고리→트랙→채널) 우선순위로 전처리본 경로 반환.
    카테고리는 LUT 기반으로 판별(등록된 값만 사용).
    """
    key = norm_identity(identity_path)
    options = lut.get(key, {})
    if not options:
        return identity_path

    cat = category_of_identity(identity_path, track, lut)
    if cat is None:
        return identity_path

    order = ["gray", "color"] if prefer_gray else ["color", "gray"]
    for ch in order:
        tup = (cat, track, ch)
        if tup in options:
            return str(images_root / options[tup])

    # 동일 트랙 내 임의 채널 대체
    for (c2, t2, ch2), rel in options.items():
        if c2 == cat and t2 == track:
            return str(images_root / rel)

    return identity_path

# ────────────────────────── 페어 구성 ──────────────────────────
def load_positive_pairs(cand_json: Dict, lut: Dict[str, Dict[Tuple[str, str, str], str]],
                        mode: str = "top1") -> List[Tuple[str, str, str]]:
    """
    candidates.json → [(extracted_identity, reference_identity, track)]
    - 필수: info["original_path"], info["track"], info["candidates"]
    - 카테고리 판별은 LUT 기반
    """
    pos: List[Tuple[str, str, str]] = []
    for _key, info in cand_json.items():
        origin_path = info.get("original_path")
        track = str(info.get("track", "")).lower()
        items = info.get("candidates")

        if not origin_path or track not in ("low", "high") or not isinstance(items, list):
            raise ValueError("candidates.json: 'original_path', 'track(low|high)', 'candidates' 필드가 필요합니다.")

        chosen = [items[0]] if (mode == "top1" and items) else items
        for c in chosen:
            name = c.get("name")
            if not name:
                continue

            # LUT로 카테고리 판별
            cat_a = category_of_identity(origin_path, track, lut)
            cat_b = category_of_identity(name, track, lut)
            if cat_a is None or cat_b is None or cat_a == cat_b:
                continue

            if cat_a == "extracted" and cat_b == "reference":
                ex, ori = origin_path, name
            elif cat_a == "reference" and cat_b == "extracted":
                ex, ori = name, origin_path
            else:
                continue

            pos.append((ex, ori, track))
    return pos

def sample_negatives(pos_pairs: List[Tuple[str, str, str]], ratio: float = 1.0,
                     mode: str = "hard", seed: int = 0) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)
    if not pos_pairs or ratio <= 0:
        return []

    # ex/ori 별 집합 및 정답 매핑
    ex_ids = sorted({ex for ex, _, _ in pos_pairs})
    or_ids = sorted({ori for _, ori, _ in pos_pairs})
    truth_map: Dict[str, set[str]] = {}
    ex_tracks: Dict[str, List[str]] = {}
    for ex, ori, tr in pos_pairs:
        truth_map.setdefault(ex, set()).add(ori)
        ex_tracks.setdefault(ex, [])
        if tr not in ex_tracks[ex]:
            ex_tracks[ex].append(tr)

    target = int(len(pos_pairs) * ratio)
    neg: List[Tuple[str, str, str]] = []
    tries = 0

    if mode == "shuffle":
        while len(neg) < target and tries < target * 20:
            tries += 1
            e = rng.choice(ex_ids)
            o = rng.choice(or_ids)
            if o in truth_map.get(e, set()):
                continue
            tr = rng.choice(ex_tracks.get(e, ["low"]))
            neg.append((e, o, tr))
        return neg

    # hard: 정답 ORI 제외 + 같은 ex의 트랙을 유지
    while len(neg) < target and tries < target * 50:
        tries += 1
        e, _o_true, tr = rng.choice(pos_pairs)
        o = rng.choice(or_ids)
        if o in truth_map.get(e, set()):
            continue
        neg.append((e, o, tr))
    return neg

# ────────────────────────── CLI ──────────────────────────
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

# ────────────────────────── Main ──────────────────────────
def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cand_json = json.loads(Path(args.candidates_json).read_text("utf-8"))
    pre_map = json.loads(Path(args.mapping).read_text("utf-8"))
    images_root = Path(args.images_root)

    # LUT 생성(메타 기반, reference 명칭 고정)
    lut = build_lookup(pre_map, images_root)

    # 양성/음성 페어 구성
    pos = load_positive_pairs(cand_json, lut, args.pos_mode)
    neg = sample_negatives(pos, ratio=args.neg_ratio, mode=args.neg_mode, seed=args.seed)
    print(f"[INFO] positives: {len(pos):,} , negatives: {len(neg):,}")

    rows: List[Dict] = []
    miss = 0
    all_pairs = [("P",) + p for p in pos] + [("N",) + p for p in neg]
    random.Random(args.seed).shuffle(all_pairs)

    for pair, ex_id, or_id, track in tqdm(all_pairs, ncols=80, desc="ORB"):
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
            pair=pair,  # ← 'label' 대신 'pair' 사용(P/N)
            ex_id=Path(ex_id).name, or_id=Path(or_id).name,
            ex_path=ex_id, or_path=or_id,
            ex_used=str(ex_path), or_used=str(or_path),
            track=track,
            inliers=inl, kpA=kA, kpB=kB, score=float(score),
        ))

    print(f"[INFO] file-read failures: {miss}")

    # 저장
    df = pd.DataFrame(rows)
    df.to_csv(out / "match_scores.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] {out/'match_scores.csv'}")

    # ── 임계값 산정 ──
    thr_info: Dict[str, Optional[float]] = {"orb_score": None, "ci_low": None, "ci_high": None}

    if args.labels_csv and Path(args.labels_csv).is_file():
        lab = pd.read_csv(args.labels_csv, encoding="utf-8-sig")
        must = {"orig_path", "extracted_path", "label"}
        if (must - set(lab.columns)):
            raise ValueError("labels-csv must contain EXACTLY these columns: orig_path, extracted_path, label")

        lab = lab[["orig_path", "extracted_path", "label"]].copy()
        lab["orig_b"] = lab["orig_path"].apply(lambda s: Path(str(s)).name)
        lab["extr_b"] = lab["extracted_path"].apply(lambda s: Path(str(s)).name)
        lab["label"] = lab["label"].astype(int)

        d2 = df.copy()
        d2["orig_b"] = d2["or_id"]
        d2["extr_b"] = d2["ex_id"]

        # 충돌 방지를 위해 df에는 'pair' 컬럼만 존재(라벨명 미충돌)
        merged = d2.merge(lab[["orig_b", "extr_b", "label"]], on=["orig_b", "extr_b"], how="left")
        labeled = merged.dropna(subset=["label"]).copy()

        if labeled.empty:
            print("[WARN] no labeled pairs matched; falling back to no-label mode.")
        else:
            y = (labeled["label"] == 1).astype(int).values
            s = labeled["score"].values
            thr: Optional[float] = None
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
                pos_scores = s[y == 1]
                fallback = float(np.quantile(pos_scores, 0.05)) if pos_scores.size else 0.0
                thr_info = {"orb_score": fallback, "ci_low": None, "ci_high": None}

    if thr_info["orb_score"] is None:
        pos_scores = df.loc[df["pair"] == "P", "score"].tolist()
        thr_info = {"orb_score": quantile_threshold(pos_scores, q=0.97), "ci_low": None, "ci_high": None}

    (out / "thresholds.json").write_text(json.dumps(thr_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[KEY] threshold = {thr_info['orb_score']:.4f}")

    # ── 플롯 ──
    plt.figure()
    plt.hist(df["score"].tolist(), bins=50)
    plt.axvline(float(thr_info["orb_score"]), ls="--")
    plt.xlabel("score (inliers/kpA)")
    plt.ylabel("count")
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
