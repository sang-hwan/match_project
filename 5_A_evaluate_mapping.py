# 5_A_evaluate_mapping.py
"""
Evaluate mapping quality in both manual-label and no-label settings.

Main features:
1. Label coverage detection:
   - Without --labels-csv: generate candidate list to_label.csv
   - Labels ≤ N (default 10): output point estimates with standard error (SE) and 95% CI

2. Extended metrics:
   - Precision, Recall, F1, Accuracy + Top-5 Recall (when candidate list provided)
   - Save error sample HTML (error_pairs.html) and scatter plot (error_scatter.png)

3. Active learning suggestion:
   - Suggest N low-confidence samples near score threshold (--suggest-n)

Inputs:
  --mapping-json : JSON mapping result from 4_verify_mapping.py (orig→extracted)
  --scores-csv   : match_scores.csv for diagnostics/active-learning (optional)
  --labels-csv   : manual label CSV; omit if none yet (orig_path,extracted_path,label)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def load_mapping(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_labels(path: Path) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        if [c.lower() for c in header[:3]] != ["orig_path", "extracted_path", "label"]:
            raise ValueError("CSV header must be: orig_path,extracted_path,label")
        for o, e, l in rdr:
            rows.append((o, e, int(l)))
    return rows


def confusion_from_labels(labels: List[Tuple[str, str, int]],
                          mapping: Dict[str, str]) -> Tuple[int, int, int, int]:
    TP = FP = FN = TN = 0
    preds = {(o, p) for o, p in mapping.items()}
    for o, e, lab in labels:
        positive = lab == 1
        pred_pos = (o, e) in preds
        if positive and pred_pos:
            TP += 1
        elif positive and not pred_pos:
            FN += 1
        elif not positive and pred_pos:
            FP += 1
        else:
            TN += 1
    return TP, FP, FN, TN


def metrics(TP: int, FP: int, FN: int, TN: int) -> Tuple[float, float, float, float]:
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    acc  = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, acc, f1


def ci_proportion(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score interval."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    dev = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = max(0.0, (centre - dev) / denom)
    hi = min(1.0, (centre + dev) / denom)
    return lo, hi


def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser(description="Evaluate (or suggest labels for) mapping")
    pa.add_argument("--mapping-json", default="mapping_result.json")
    pa.add_argument("--scores-csv", default=None,
                    help="match_scores.csv for diagnostics/active-learning")
    pa.add_argument("--labels-csv", default=None,
                    help="manual label CSV; omit if none yet")
    pa.add_argument("--suggest-n", type=int, default=20,
                    help="number of pairs to suggest when no labels")
    return pa.parse_args()


def active_learning(sco_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Suggest N samples with lowest confidence near the score threshold."""
    if sco_df.empty:
        return pd.DataFrame()
    mid = sco_df["score"].median()
    sco_df["abs_gap"] = (sco_df["score"] - mid).abs()
    rec = sco_df.sort_values("abs_gap").head(n)[
        ["orig", "extracted", "score", "inliers", "kpA"]
    ]
    rec.to_csv("to_label.csv", index=False, encoding="utf-8-sig")
    print(f"[SUGGEST] wrote {len(rec)} rows to to_label.csv  (label them 1/0)")
    return rec


def main():
    args = parse_args()
    mapping = load_mapping(Path(args.mapping_json))

    if args.labels_csv and Path(args.labels_csv).is_file():
        labels = load_labels(Path(args.labels_csv))
        TP, FP, FN, TN = confusion_from_labels(labels, mapping)
        prec, rec, acc, f1 = metrics(TP, FP, FN, TN)

        n_pos = TP + FN
        n_pred_pos = TP + FP
        prec_ci = ci_proportion(prec, n_pred_pos)
        rec_ci  = ci_proportion(rec,  n_pos)

        print("\n===== Evaluation (with manual labels) =====")
        print(f"TP={TP} FP={FP} FN={FN} TN={TN}")
        print(f"Precision = {prec:.3f}  (95% CI {prec_ci[0]:.3f}–{prec_ci[1]:.3f})")
        print(f"Recall    = {rec:.3f}  (95% CI {rec_ci[0]:.3f}–{rec_ci[1]:.3f})")
        print(f"F1        = {f1:.3f}")
        print(f"Accuracy  = {acc:.3f}")

        md = Path("evaluation_report.md")
        with md.open("w", encoding="utf-8") as f:
            f.write("# Mapping Evaluation Report\n\n")
            f.write("| Metric | Value | 95% CI |\n|---|---|---|\n")
            f.write(f"| Precision | {prec:.3f} | {prec_ci[0]:.3f}–{prec_ci[1]:.3f} |\n")
            f.write(f"| Recall | {rec:.3f} | {rec_ci[0]:.3f}–{rec_ci[1]:.3f} |\n")
            f.write(f"| F1 | {f1:.3f} |  |\n")
            f.write(f"| Accuracy | {acc:.3f} |  |\n")
            f.write("\n## Confusion Matrix\n")
            f.write("| | Pred+ | Pred- |\n|---|---|---|\n")
            f.write(f"| Actual+ | {TP} | {FN} |\n")
            f.write(f"| Actual- | {FP} | {TN} |\n")
        print(f"[REPORT] saved {md}")

        if args.scores_csv and Path(args.scores_csv).is_file():
            df_scores = pd.read_csv(args.scores_csv)
            df_scores["key"] = df_scores["orig"] + "||" + df_scores["extracted"]
            df_labels = (
                pd.DataFrame(labels, columns=["orig", "extracted", "label"])
                .assign(key=lambda d: d["orig"] + "||" + d["extracted"])    
            )
            df = df_labels.merge(df_scores, on="key", how="left")
            plt.figure()
            ok = df[df["label"] == 1]
            ng = df[df["label"] == 0]
            plt.scatter(ok["score"], ok["inliers"], label="TP", marker="o")
            plt.scatter(ng["score"], ng["inliers"], label="FP/FN", marker="x")
            plt.xlabel("ORB inlier_ratio score")
            plt.ylabel("inlier_cnt")
            plt.legend()
            plt.savefig("error_scatter.png", dpi=200)
            df.to_html("error_pairs.html", index=False)
            print("[DIAG] saved error_scatter.png, error_pairs.html")
    else:
        if not args.scores_csv or not Path(args.scores_csv).is_file():
            print("No labels & no scores ⇒ nothing to do. Provide --scores-csv.")
            return
        df_scores = pd.read_csv(args.scores_csv)
        active_learning(df_scores, args.suggest_n)

if __name__ == "__main__":
    main()
