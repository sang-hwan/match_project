# 4_B_extract_threshold.py
"""ORB + RANSAC threshold estimator for the HWP-image matching project.

Images live in the following **fixed** structure (all PNG):

processed/extracted/{low|high}/{gray|color}/<추출 PNG>
processed/original/{low|high}/{gray|color}/<원본 PNG>

PowerShell example
------------------
python 4_B_extract_threshold.py `
    --candidates-json ".\\candidates.json" `
    --images-root     ".\\processed" `
    --out-dir         ".\\outputs"
"""

from __future__ import annotations
import argparse, json, os, random
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple

import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import Memory
from tqdm import tqdm
from scipy.stats import beta
from kneed import KneeLocator

# ───────────────────────── cache ────────────────────────────
CACHE = Memory(".cache/descriptors", verbose=0)

# ───────────────────── helper: robust imread ────────────────
def _safe_imread(p: str, flag=cv2.IMREAD_GRAYSCALE):
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, flag)
    if img is not None:
        return img
    try:
        buf = np.fromfile(p, np.uint8)
        return cv2.imdecode(buf, flag)
    except Exception:
        return None


@CACHE.cache
def _extract_orb(path: str, n=1500):
    img = _safe_imread(path)
    if img is None:
        raise FileNotFoundError(path)
    orb = cv2.ORB_create(nfeatures=n)
    kps, des = orb.detectAndCompute(img, None)
    return np.float32([kp.pt for kp in kps]), des


def _orb_score(a: str, b: str, ratio=0.75) -> Tuple[int, int, int]:
    ptsA, desA = _extract_orb(a); ptsB, desB = _extract_orb(b)
    if desA is None or desB is None:
        return 0, len(ptsA), len(ptsB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in bf.knnMatch(desA, desB, 2)
            if m.distance < ratio * n.distance]
    if len(good) < 4:
        return 0, len(ptsA), len(ptsB)
    src = np.float32([ptsA[m.queryIdx] for m in good])
    dst = np.float32([ptsB[m.trainIdx] for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = int(mask.sum()) if mask is not None else 0
    return inl, len(ptsA), len(ptsB)


def _knee(values: List[float]) -> float:
    cdf = np.linspace(0, 1, len(values))
    k = KneeLocator(values, cdf, curve="concave", direction="increasing")
    return float(k.knee) if k.knee else float(np.quantile(values, 0.97))


# ───────────────────────── CLI ──────────────────────────────
def _argparser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--images-root", default="processed")
    ap.add_argument("--labels-csv", default=None)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--pos-mode", choices=["top1", "all"], default="top1")
    ap.add_argument("--neg-mode", choices=["shuffle", "hard"], default="hard")
    ap.add_argument("--target-tpr", type=float, default=0.97)
    return ap


# ────────────────────── data helpers ────────────────────────
def _load_candidates(fp: Path, mode: str):
    js = json.loads(fp.read_text("utf-8")); pairs = []
    for orig_path, info in js.items():
        cands = info.get("candidates", info.get("extracted_candidates", []))
        if not cands:
            continue
        orig = PureWindowsPath(orig_path).name
        if mode == "top1":
            pairs.append((PureWindowsPath(cands[0]["name"]).name, orig))
        else:
            pairs += [(PureWindowsPath(c["name"]).name, orig) for c in cands]
    return pairs


def _negatives(pos, ids, mode="hard", ratio=1.0):
    pos_set, neg, rng = set(pos), [], random.Random(0)
    mult = 3 if mode == "hard" else 1
    while len(neg) < int(len(pos) * ratio * mult):
        a, b = rng.sample(ids, 2)
        if (a, b) not in pos_set:
            neg.append((a, b))
    return neg[: int(len(pos) * ratio)]


# ────────────────────── path mapping ────────────────────────
def _build_paths(meta: dict, root: str) -> Dict[str, str]:
    id2path: Dict[str, str] = {}
    for img_id, row in meta.items():
        track, ch = row["트랙"], row["채널"]
        # 추출 전처리본
        id2path[img_id] = str(Path(root, "extracted", track, ch, img_id))
        # 대응되는 원본 전처리본(있으면) 또는 RAW
        oname = Path(row["원본_전체_경로"]).name
        p_orig = Path(root, "original", track, ch, oname)
        id2path.setdefault(oname, str(p_orig if p_orig.exists() else row["원본_전체_경로"]))
    return id2path


# ───────────────────────── main ─────────────────────────────
def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    pos = _load_candidates(Path(a.candidates_json), a.pos_mode)
    print(f"[INFO] positives: {len(pos):,}")

    meta = json.loads(Path("preprocess_mapping.json").read_text("utf-8"))
    id2path = _build_paths(meta, a.images_root)
    print(f"[INFO] mapping entries: {len(id2path):,}")

    # 추가 id 확보(candidates 내 절대경로)
    raw = json.loads(Path(a.candidates_json).read_text("utf-8"))
    for o, v in raw.items():
        id2path.setdefault(PureWindowsPath(o).name, o)
        for c in v.get("candidates", v.get("extracted_candidates", [])):
            id2path.setdefault(PureWindowsPath(c["name"]).name, c["name"])

    neg = _negatives(pos, list(id2path), a.neg_mode)
    print(f"[INFO] negatives: {len(neg):,}")

    pairs = [("P",)+p for p in pos] + [("N",)+p for p in neg]
    random.shuffle(pairs)

    rows, miss = [], 0
    for lb, ex, ori in tqdm(pairs, ncols=80, desc="ORB"):
        pa, pb = id2path[ex], id2path[ori]
        try:
            inl, kA, kB = _orb_score(pa, pb)
        except FileNotFoundError:
            miss += 1; continue
        rows.append(dict(label=lb, ex_id=ex, or_id=ori,
                         inliers=inl, kpA=kA, kpB=kB,
                         score=inl/(kA+1e-6)))
    print(f"[INFO] file-read failures: {miss}")
    df = pd.DataFrame(rows)
    df.to_csv(out/"match_scores.csv", index=False)
    print(f"[INFO] ORB rows: {len(df):,}")

    scores, ssorted = df["score"].tolist(), sorted(df["score"])
    if a.labels_csv:
        lab = pd.read_csv(a.labels_csv, encoding="utf-8-sig")
        lmap = {(Path(r.extracted_path).name, Path(r.orig_path).name): r.label
                for r in lab.itertuples(index=False)}
        y = [1 if lmap.get((r.ex_id, r.or_id), 0)==1 else 0
             for r in df.itertuples(index=False)]
        tp, fn = sum(y), len(y)-sum(y)
        thr = next((t for t in np.linspace(0,1,101)
                    if (sum((s>=t and y[i]) for i,s in enumerate(scores))+1)/(tp+2)
                    >= a.target_tpr), 0.0)
        thr_dict = dict(orb_score=thr,
                        ci_low=beta.ppf(.025,tp+1,fn+1),
                        ci_high=beta.ppf(.975,tp+1,fn+1))
    else:
        thr_dict = dict(orb_score=_knee(ssorted),
                        ci_low=None, ci_high=None)

    (out/"thresholds.json").write_text(json.dumps(thr_dict, indent=2, ensure_ascii=False))
    print(f"[KEY] threshold = {thr_dict['orb_score']:.4f}")

    plt.figure(); plt.hist(ssorted,100,log=True)
    plt.title("ORB score histogram"); plt.xlabel("score"); plt.ylabel("freq(log)")
    plt.savefig(out/"orb_hist.png", dpi=150)

    plt.figure(); plt.plot(ssorted, np.linspace(0,1,len(ssorted)))
    plt.axvline(thr_dict["orb_score"], color="r", ls="--")
    plt.title("ORB score CDF"); plt.xlabel("score"); plt.ylabel("CDF")
    plt.savefig(out/"orb_cdf.png", dpi=150)
    print("Done.")


if __name__ == "__main__":
    main(_argparser().parse_args())
