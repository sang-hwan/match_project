# 5_A_evaluate_mapping.py
"""
Evaluate mapping with optional labels, collisions, per-track metrics, and PR/ROC.

Inputs
------
--mapping-json : JSON mapping (orig -> extracted) from 4_verify_mapping.py
--scores-csv   : match/pair score CSV (optional)
--labels-csv   : manual labels CSV (optional; columns: orig[_path], extracted[_path], label|y)
--mapping-meta : preprocess_mapping.json for per-track metrics (optional)
--suggest-n    : suggestions count for no-label active learning (default: 20)

Outputs
-------
evaluation_report.md
collisions.csv
metrics_by_track.csv          (when --mapping-meta is given)
error_scatter.png, error_pairs.html
pr_curve.png, roc_curve.png, threshold_sweep.csv  (when labels+scores given)
to_label.csv                  (no-label path)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────── I/O + Path ───────────────
def load_mapping(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def norm_path(s: str) -> str:
    """OS 무관 경로 비교를 위한 정규화(구분자 통일 + 소문자)."""
    return str(Path(str(s))).replace("\\", "/").lower()

def load_labels_df(path: Path) -> pd.DataFrame:
    """유연한 컬럼 처리 → ['orig','extracted','label'] 및 키 열 생성."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "orig_path" in cols: rename_map[cols["orig_path"]] = "orig"
    elif "orig" in cols:    rename_map[cols["orig"]] = "orig"
    else: raise ValueError("Labels CSV must contain 'orig_path' or 'orig'")

    if "extracted_path" in cols: rename_map[cols["extracted_path"]] = "extracted"
    elif "extracted" in cols:    rename_map[cols["extracted"]] = "extracted"
    else: raise ValueError("Labels CSV must contain 'extracted_path' or 'extracted'")

    if "label" in cols: rename_map[cols["label"]] = "label"
    elif "y" in cols:   rename_map[cols["y"]] = "label"
    else: raise ValueError("Labels CSV must contain 'label' (or 'y')")

    df = df.rename(columns=rename_map)[["orig", "extracted", "label"]]
    df["label"] = df["label"].astype(int)
    df["orig"] = df["orig"].astype(str)
    df["extracted"] = df["extracted"].astype(str)
    df["orig_base"] = df["orig"].apply(lambda s: Path(str(s)).name)
    df["extracted_base"] = df["extracted"].apply(lambda s: Path(str(s)).name)
    df["key_full"] = df["orig"].apply(norm_path) + "||" + df["extracted"].apply(norm_path)
    df["key_base"] = df["orig_base"].str.lower() + "||" + df["extracted_base"].str.lower()
    return df

def read_scores_csv(p: Path) -> pd.DataFrame:
    """
    4_B_extract_threshold.py / 4_verify_mapping.py 산출물 호환:
      - or_id/ex_id (basename) 또는 or_path/ex_path (fullpath)
      - 없을 시 score = inliers / kpA 로 유도
    """
    df = pd.read_csv(p, encoding="utf-8-sig")
    low = {c.lower(): c for c in df.columns}
    rename = {}
    if "or_id"   in low: rename[low["or_id"]]   = "orig"
    if "ex_id"   in low: rename[low["ex_id"]]   = "extracted"
    if "or_path" in low: rename[low["or_path"]] = "orig"
    if "ex_path" in low: rename[low["ex_path"]] = "extracted"
    df = df.rename(columns=rename)

    if not {"orig", "extracted"}.issubset(df.columns):
        raise ValueError("scores-csv must include 'orig' and 'extracted' (or or_id/ex_id, or or_path/ex_path).")

    if "score" not in df.columns:
        if {"inliers", "kpA"}.issubset(df.columns):
            kp = df["kpA"].replace(0, np.nan).astype(float)
            df["score"] = df["inliers"].astype(float) / kp
            df["score"] = df["score"].fillna(0.0)
        else:
            raise ValueError("scores-csv missing 'score' and cannot derive from 'inliers'/'kpA'.")

    for c, default in (("inliers", 0), ("kpA", 1)):
        if c not in df.columns:
            df[c] = default

    df["orig"] = df["orig"].astype(str)
    df["extracted"] = df["extracted"].astype(str)
    df["orig_base"] = df["orig"].apply(lambda s: Path(str(s)).name)
    df["extracted_base"] = df["extracted"].apply(lambda s: Path(str(s)).name)
    df["key_full"] = df["orig"].apply(norm_path) + "||" + df["extracted"].apply(norm_path)
    df["key_base"] = df["orig_base"].str.lower() + "||" + df["extracted_base"].str.lower()
    return df


# ─────────────── Metrics ───────────────
def confusion_from_labels_df(labels: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[int, int, int, int]:
    TP = FP = FN = TN = 0
    preds = {(norm_path(o), norm_path(e)) for o, e in mapping.items()}
    for row in labels.itertuples(index=False):
        o, e, lab = norm_path(row.orig), norm_path(row.extracted), int(row.label)
        pred_pos = (o, e) in preds
        if lab == 1 and pred_pos: TP += 1
        elif lab == 1 and not pred_pos: FN += 1
        elif lab == 0 and pred_pos: FP += 1
        else: TN += 1
    return TP, FP, FN, TN

def metrics(TP: int, FP: int, FN: int, TN: int):
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    acc  = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, acc, f1

def ci_proportion(p: float, n: int, z: float = 1.96):
    """Wilson score interval [lo, hi]."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    dev = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = max(0.0, (centre - dev) / denom)
    hi = min(1.0, (centre + dev) / denom)
    return lo, hi


# ─────────────── Collisions ───────────────
def find_collisions(mapping: Dict[str, str]) -> pd.DataFrame:
    """1:N(N:1) 충돌 목록 반환."""
    inv = defaultdict(list)  # extracted -> [orig...]
    for o, e in mapping.items():
        inv[norm_path(e)].append(norm_path(o))
    rows = []
    for e, origins in inv.items():
        if len(origins) > 1:
            for o in origins:
                rows.append({"type": "extracted_used_by_multiple", "extracted": e, "original": o})
    fwd = defaultdict(list)  # original -> [extracted...]
    for o, e in mapping.items():
        fwd[norm_path(o)].append(norm_path(e))
    for o, exs in fwd.items():
        if len(exs) > 1:
            for e in exs:
                rows.append({"type": "original_mapped_to_multiple", "original": o, "extracted": e})
    return pd.DataFrame(rows)


# ─────────────── Track-wise ───────────────
def build_origin_track_index(meta_json: Path) -> Dict[str, Set[str]]:
    """preprocess_mapping.json → origin(norm) -> {tracks}"""
    mp = json.loads(meta_json.read_text(encoding="utf-8"))
    idx: Dict[str, Set[str]] = defaultdict(set)
    for m in mp.values():
        o = norm_path(m.get("원본_전체_경로", ""))
        trk = str(m.get("트랙", "")).lower()
        if o and trk:
            idx[o].add(trk)
    return idx

def per_track_metrics(labels: pd.DataFrame, mapping: Dict[str, str],
                      origin_tracks: Dict[str, Set[str]]) -> pd.DataFrame:
    rows = []
    for track in ("low", "high"):
        mask = labels["orig"].apply(lambda s: track in origin_tracks.get(norm_path(s), set()))
        sub = labels[mask].copy()
        TP, FP, FN, TN = confusion_from_labels_df(sub, mapping)
        prec, rec, acc, f1 = metrics(TP, FP, FN, TN)
        rows.append(dict(track=track, TP=TP, FP=FP, FN=FN, TN=TN,
                         precision=prec, recall=rec, accuracy=acc, f1=f1, n_labels=len(sub)))
    return pd.DataFrame(rows)


# ─────────────── Diagnostics (labels+scores) ───────────────
def join_labels_scores(labels: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """key_full 우선, 실패 시 key_base로 폴백."""
    df = labels.merge(scores, on="key_full", how="left")
    if df["score"].isna().all():
        df = labels.merge(scores, on="key_base", how="left")
    return df

def sweep_threshold(y_true: np.ndarray, y_score: np.ndarray, steps: int = 201) -> pd.DataFrame:
    thr_list = np.linspace(0.0, 1.0, steps)
    rows = []
    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    for t in thr_list:
        pred = (y_score >= t).astype(int)
        TP = int(np.sum((pred == 1) & (y_true == 1)))
        FP = int(np.sum((pred == 1) & (y_true == 0)))
        FN = int(np.sum((pred == 0) & (y_true == 1)))
        TN = int(np.sum((pred == 0) & (y_true == 0)))
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec  = TP / (TP + FN) if TP + FN else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        tpr  = TP / P if P else 0.0
        fpr  = FP / N if N else 0.0
        rows.append(dict(thr=t, precision=prec, recall=rec, f1=f1, tpr=tpr, fpr=fpr, tp=TP, fp=FP, fn=FN, tn=TN))
    return pd.DataFrame(rows)

def plot_pr_roc(sweep: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(sweep["recall"], sweep["precision"])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curve")
    plt.tight_layout()
    plt.savefig("pr_curve.png", dpi=200)

    plt.figure()
    plt.plot(sweep["fpr"], sweep["tpr"])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=200)


# ─────────────── Active learning (no labels) ───────────────
def active_learning(scores_df: pd.DataFrame, n: int) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    center = None
    thr_path = Path("thresholds.json")
    if thr_path.is_file():
        try:
            center = float(json.loads(thr_path.read_text(encoding="utf-8")).get("orb_score", 0.0)) or None
        except Exception:
            center = None
    if center is None:
        center = float(scores_df["score"].median())

    sdf = scores_df.copy()
    sdf["abs_gap"] = (sdf["score"] - center).abs()
    cols = [c for c in ["orig", "extracted", "score", "inliers", "kpA"] if c in sdf.columns]
    rec = sdf.sort_values("abs_gap").head(n)[cols]
    rec.to_csv("to_label.csv", index=False, encoding="utf-8-sig")
    print(f"[SUGGEST] wrote {len(rec)} rows to to_label.csv")
    return rec


# ─────────────── CLI ───────────────
def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser(description="Evaluate (or suggest labels for) mapping")
    pa.add_argument("--mapping-json", default="mapping_result.json")
    pa.add_argument("--scores-csv", default=None)
    pa.add_argument("--labels-csv", default=None)
    pa.add_argument("--mapping-meta", default=None)
    pa.add_argument("--suggest-n", type=int, default=20)
    return pa.parse_args()


# ─────────────── Main ───────────────
def main():
    args = parse_args()
    mapping = load_mapping(Path(args.mapping_json))

    # Collisions
    col_df = find_collisions(mapping)
    if not col_df.empty:
        col_df.to_csv("collisions.csv", index=False, encoding="utf-8-sig")

    labeled = False
    TP = FP = FN = TN = 0
    prec = rec = acc = f1 = 0.0
    prec_ci = (0.0, 0.0)
    rec_ci = (0.0, 0.0)
    metrics_track_df = None

    # Labels path
    if args.labels_csv and Path(args.labels_csv).is_file():
        labeled = True
        labels = load_labels_df(Path(args.labels_csv))
        TP, FP, FN, TN = confusion_from_labels_df(labels, mapping)
        prec, rec, acc, f1 = metrics(TP, FP, FN, TN)
        n_pos = TP + FN
        n_pred_pos = TP + FP
        prec_ci, rec_ci = ci_proportion(prec, n_pred_pos), ci_proportion(rec, n_pos)

        # Per-track (optional)
        if args.mapping_meta and Path(args.mapping_meta).is_file():
            idx = build_origin_track_index(Path(args.mapping_meta))
            metrics_track_df = per_track_metrics(labels, mapping, idx)
            metrics_track_df.to_csv("metrics_by_track.csv", index=False, encoding="utf-8-sig")

        # Diagnostics with scores
        if args.scores_csv and Path(args.scores_csv).is_file():
            scores = read_scores_csv(Path(args.scores_csv))
            dfj = join_labels_scores(labels, scores)

            # 산포도(정답/오답 시각 점검)
            plt.figure()
            ok = dfj[dfj["label"] == 1]
            ng = dfj[dfj["label"] == 0]
            plt.scatter(ok["score"], ok.get("inliers", pd.Series([0]*len(ok))), label="Label=1", marker="o")
            plt.scatter(ng["score"], ng.get("inliers", pd.Series([0]*len(ng))), label="Label=0", marker="x")
            plt.xlabel("ORB score"); plt.ylabel("Inliers")
            plt.legend(); plt.tight_layout()
            plt.savefig("error_scatter.png", dpi=200)
            dfj.to_html("error_pairs.html", index=False)

            if "score" in dfj.columns and dfj["score"].notna().any():
                y_true = dfj["label"].astype(int).to_numpy()
                y_score = dfj["score"].fillna(0.0).astype(float).to_numpy()
                sweep = sweep_threshold(y_true, y_score)
                sweep.to_csv("threshold_sweep.csv", index=False, encoding="utf-8-sig")
                plot_pr_roc(sweep)
                best = sweep.sort_values("f1", ascending=False).head(1).iloc[0]
                print(f"[THRESHOLD] F1-max threshold = {best['thr']:.3f}  "
                      f"(P={best['precision']:.3f}, R={best['recall']:.3f})")

    # Report
    md = Path("evaluation_report.md")
    with md.open("w", encoding="utf-8") as f:
        f.write("# Mapping Evaluation Report\n\n")
        if labeled:
            f.write("| Metric | Value | 95% CI |\n|---|---|---|\n")
            f.write(f"| Precision | {prec:.3f} | {prec_ci[0]:.3f}–{prec_ci[1]:.3f} |\n")
            f.write(f"| Recall | {rec:.3f} | {rec_ci[0]:.3f}–{rec_ci[1]:.3f} |\n")
            f.write(f"| F1 | {f1:.3f} |  |\n")
            f.write(f"| Accuracy | {acc:.3f} |  |\n")

            f.write("\n## Confusion Matrix\n")
            f.write("| | Pred+ | Pred- |\n|---|---|---|\n")
            f.write(f"| Actual+ | {TP} | {FN} |\n")
            f.write(f"| Actual- | {FP} | {TN} |\n")

            if metrics_track_df is not None and not metrics_track_df.empty:
                f.write("\n## Per-Track Metrics (see metrics_by_track.csv)\n")
                for r in metrics_track_df.to_dict("records"):
                    f.write(f"- {r['track']}: P={r['precision']:.3f}, R={r['recall']:.3f}, "
                            f"F1={r['f1']:.3f}, n={r['n_labels']}\n")
        else:
            f.write("No labels supplied — produced active-learning suggestions (if scores provided).\n")

        f.write("\n## Collisions\n")
        if col_df.empty:
            f.write("No 1:N or N:1 collisions detected.\n")
        else:
            n_rows = len(col_df)
            n_ex = (col_df['type'] == 'extracted_used_by_multiple').sum()
            n_or = (col_df['type'] == 'original_mapped_to_multiple').sum()
            f.write(f"{n_rows} collision rows written to **collisions.csv** "
                    f"(extracted_used_by_multiple={n_ex}, original_mapped_to_multiple={n_or}).\n")

        f.write("\n## Coverage\n")
        f.write(f"Total mapped pairs: {len(mapping)}\n")

    print(f"[REPORT] saved {md}")

    # No-label path: suggestions
    if not labeled and args.scores_csv and Path(args.scores_csv).is_file():
        scores = read_scores_csv(Path(args.scores_csv))
        active_learning(scores, args.suggest_n)


if __name__ == "__main__":
    main()
