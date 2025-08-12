# 5_A_evaluate_mapping.py
"""
매핑 결과 평가 스크립트 (스키마 유연/안정화 버전)

기능
- mapping_result.json(최종 1:1 매핑)을 hand_label.csv(라벨)과 대조하여 Precision/Recall/F1 산출
- pair_scores.csv(전체 후보 점수)에서 상위 N개 제안(suggestions) 생성(매핑에 미포함 + 음성 라벨 제외)
- optional: preprocess_mapping.json에서 reference 타깃 수 집계 및 커버리지 계산

호환성
- scores CSV 컬럼 자동 인식:
  - 우선순위: ('orig','extracted')  → 없으면  ('or_path','ex_path')  → 없으면 ('orig_path','extracted_path')
- labels CSV 컬럼 자동 인식:
  - ('orig_path','extracted_path','label')를 권장, 없으면 ('orig','extracted','label') 또는 ('or_path','ex_path','label')도 허용
- 중복 인덱스/라벨 정렬 이슈 방지: reset_index + 위치기반 대입(to_numpy) 사용

출력
- 표준 출력: 핵심 지표, 통계
- <out_dir>/evaluation_report.json   (mapping_json 위치에 저장)
- <out_dir>/suggestions_top{N}.csv   (scores_csv 기반 N개 제안, --suggest-n>0일 때)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────── IO & 정규화 ────────────────────────────────
def _basename(x: str) -> str:
    return Path(str(x)).name


def read_mapping_json(p: Path) -> pd.DataFrame:
    js = json.loads(p.read_text("utf-8"))
    pairs = js.get("pairs", [])
    # 예상 키: original_path, extracted_path, score, inliers, kpA, kpB, track, ex_used, or_used
    rows = []
    for it in pairs:
        rows.append({
            "orig": it.get("original_path"),
            "extracted": it.get("extracted_path"),
            "score": it.get("score", np.nan),
            "inliers": it.get("inliers", np.nan),
            "kpA": it.get("kpA", np.nan),
            "kpB": it.get("kpB", np.nan),
            "track": str(it.get("track", "")).lower() if it.get("track") is not None else "",
            "ex_used": it.get("ex_used"),
            "or_used": it.get("or_used"),
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    if not {"orig", "extracted"} <= set(df.columns):
        raise ValueError("mapping_result.json에 'pairs[].original_path'와 'pairs[].extracted_path'가 필요합니다.")
    # base columns
    df["orig_b"] = df["orig"].astype(str).map(_basename).to_numpy()
    df["extr_b"] = df["extracted"].astype(str).map(_basename).to_numpy()
    return df


def read_scores_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, encoding="utf-8-sig")
    # 스키마 정규화
    colmap_candidates = [
        {"or_path": "orig", "ex_path": "extracted"},
        {"orig_path": "orig", "extracted_path": "extracted"},
    ]
    for cmap in colmap_candidates:
        overlap = set(cmap.keys()) & set(df.columns)
        if overlap:
            df = df.rename(columns={k: v for k, v in cmap.items() if k in df.columns})
            break
    # 이미 올바른 스키마면 통과
    if not {"orig", "extracted"} <= set(df.columns):
        raise ValueError("scores-csv에는 'orig'와 'extracted' 컬럼(또는 호환 별칭 or_path/ex_path, orig_path/extracted_path)이 필요합니다.")

    # 안정성: RangeIndex 보장
    df = df.reset_index(drop=True)

    # base columns
    df["orig_b"] = df["orig"].astype(str).map(_basename).to_numpy()
    df["extr_b"] = df["extracted"].astype(str).map(_basename).to_numpy()

    # 보조 컬럼이 없을 수 있으니 기본값 처리
    if "score" not in df.columns:
        df["score"] = np.nan
    for c in ("inliers", "kpA", "kpB"):
        if c not in df.columns:
            df[c] = np.nan
    if "track" not in df.columns:
        df["track"] = ""

    return df


def read_labels_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, encoding="utf-8-sig")
    # 라벨 스키마 정규화
    if not {"orig_path", "extracted_path", "label"} <= set(df.columns):
        # 호환 별칭 지원
        alt_maps = [
            {"orig": "orig_path", "extracted": "extracted_path"},
            {"or_path": "orig_path", "ex_path": "extracted_path"},
        ]
        for m in alt_maps:
            if set(m.keys()) <= set(df.columns) and "label" in df.columns:
                df = df.rename(columns=m)
                break

    if not {"orig_path", "extracted_path", "label"} <= set(df.columns):
        raise ValueError("labels-csv는 'orig_path','extracted_path','label' 컬럼(또는 호환 별칭)이 필요합니다.")

    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["orig_b"] = df["orig_path"].astype(str).map(_basename).to_numpy()
    df["extr_b"] = df["extracted_path"].astype(str).map(_basename).to_numpy()
    return df[["orig_path", "extracted_path", "label", "orig_b", "extr_b"]]


def read_meta_targets(preprocess_mapping_json: Optional[Path]) -> Optional[int]:
    if not preprocess_mapping_json or not preprocess_mapping_json.is_file():
        return None
    mp = json.loads(preprocess_mapping_json.read_text("utf-8"))
    # 카테고리 'reference'의 고유 원본_전체_경로 수를 타깃 수로 본다.
    refs = set()
    for _name, meta in mp.items():
        if str(meta.get("카테고리", "")).lower() == "reference":
            origin = meta.get("원본_전체_경로")
            if origin:
                refs.add(origin)
    return len(refs) if refs else None


# ──────────────────────────────── 평가 로직 ────────────────────────────────
def compute_metrics(mapping_df: pd.DataFrame, labels_df: Optional[pd.DataFrame]) -> Dict:
    total_pred = len(mapping_df)
    tp = fp = fn = None
    prec = rec = f1 = None
    labeled_join = None
    pos_total = None

    if labels_df is not None and not labels_df.empty:
        labeled_join = mapping_df.merge(
            labels_df[["orig_b", "extr_b", "label"]],
            on=["orig_b", "extr_b"],
            how="left",
        )

        # 라벨 있는 항목만으로 TP/FP 계산
        has_y = labeled_join["label"].notna()
        labeled_pred = labeled_join[has_y].copy()
        if not labeled_pred.empty:
            tp = int((labeled_pred["label"] == 1).sum())
            fp = int((labeled_pred["label"] == 0).sum())

        # 전체 라벨 양성 수(분모용)
        pos_total = int((labels_df["label"] == 1).sum())
        if pos_total and tp is not None:
            fn = pos_total - tp

        # metrics
        if tp is not None and fp is not None:
            prec = tp / (tp + fp) if (tp + fp) > 0 else None
        if tp is not None and fn is not None:
            rec = tp / (tp + fn) if (tp + fn) > 0 else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)

    return {
        "pred_pairs": total_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "pos_total": pos_total,
    }


def make_suggestions(scores_df: pd.DataFrame,
                     mapping_df: pd.DataFrame,
                     labels_df: Optional[pd.DataFrame],
                     top_n: int) -> pd.DataFrame:
    if top_n <= 0 or scores_df is None or scores_df.empty:
        return pd.DataFrame()

    # 이미 매핑에 들어간 (orig_b, extr_b)는 제외
    mapped_keys = set(zip(mapping_df["orig_b"], mapping_df["extr_b"]))
    mask_not_mapped = ~scores_df.set_index(["orig_b", "extr_b"]).index.isin(mapped_keys)
    cand = scores_df[mask_not_mapped].copy()

    # 라벨이 있으면 음성(0) 라벨은 제외
    if labels_df is not None and not labels_df.empty:
        lab = labels_df[["orig_b", "extr_b", "label"]].copy()
        cand = cand.merge(lab, on=["orig_b", "extr_b"], how="left")
        cand = cand[(cand["label"].isna()) | (cand["label"] == 1)]
        cand = cand.drop(columns=["label"])

    # 점수 높은 순으로 Top-N
    cand = cand.sort_values(["score", "inliers", "kpA", "kpB"], ascending=[False, False, False, False])
    return cand.head(top_n).reset_index(drop=True)


# ──────────────────────────────── CLI ────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate mapping_result.json against labels and scores, with robust CSV schema handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mapping-json", required=True, help="mapping_result.json")
    ap.add_argument("--scores-csv", required=True, help="pair_scores.csv")
    ap.add_argument("--labels-csv", required=False, help="hand_label.csv")
    ap.add_argument("--mapping-meta", required=False, help="preprocess_mapping.json (to count reference targets)")
    ap.add_argument("--suggest-n", type=int, default=0, help="export top-N suggestions from scores (not already mapped)")
    return ap.parse_args()


def main():
    args = parse_args()
    p_map = Path(args.mapping_json)
    p_scores = Path(args.scores_csv)
    p_labels = Path(args.labels_csv) if args.labels_csv else None
    p_meta = Path(args.mapping_meta) if args.mapping_meta else None

    # 읽기
    mapping_df = read_mapping_json(p_map)
    scores_df = read_scores_csv(p_scores)
    labels_df = read_labels_csv(p_labels) if p_labels and p_labels.is_file() else None

    # 평가
    metrics = compute_metrics(mapping_df, labels_df)

    # 타깃 개수(옵션)
    total_targets = read_meta_targets(p_meta)

    # 커버리지(라벨/메타 기준)
    coverage_ref = None
    if total_targets:
        # reference 타깃 대비 커버리지
        coverage_ref = len(mapping_df["orig"].unique()) / total_targets

    coverage_lab = None
    if labels_df is not None:
        # 라벨이 있는 원본 대비 커버리지
        coverage_lab = len(
            pd.Series(mapping_df["orig_b"].unique()).isin(labels_df["orig_b"].unique())
        ) / len(set(labels_df["orig_b"]))

    # 제안 생성
    sugg_df = make_suggestions(scores_df, mapping_df, labels_df, args.suggest_n)
    out_dir = p_map.parent
    report_path = out_dir / "evaluation_report.json"
    sugg_path = out_dir / f"suggestions_top{args.suggest_n}.csv" if args.suggest_n > 0 else None
    if sugg_path is not None and not sugg_df.empty:
        sugg_df.to_csv(sugg_path, index=False, encoding="utf-8-sig")

    # 리포트 저장
    report = {
        "summary": {
            "pred_pairs": metrics["pred_pairs"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "coverage": {
            "reference_targets_total": total_targets,
            "coverage_by_reference": coverage_ref,
            "coverage_by_labels": coverage_lab,
        },
        "files": {
            "mapping_json": str(p_map),
            "scores_csv": str(p_scores),
            "labels_csv": str(p_labels) if p_labels else None,
            "report_json": str(report_path),
            "suggestions_csv": str(sugg_path) if sugg_path and sugg_df is not None else None,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 콘솔 출력 ──
    print("===== EVALUATION SUMMARY =====")
    print(f"Predicted pairs (mapping): {metrics['pred_pairs']}")
    if metrics["tp"] is not None:
        print(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        print(f"Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")
    else:
        print("(No labels provided or matched; precision/recall not computed)")

    if total_targets is not None:
        print(f"Reference targets total: {total_targets}")
        if coverage_ref is not None:
            print(f"Coverage (by reference targets): {coverage_ref*100:.2f}%")
    if coverage_lab is not None:
        print(f"Coverage (by labeled originals): {coverage_lab*100:.2f}%")

    if args.suggest_n > 0:
        if sugg_df is not None and not sugg_df.empty:
            print(f"[SUGGEST] Top-{args.suggest_n} candidates exported → {sugg_path}")
            # 상위 5개만 미리보기
            preview = sugg_df[["orig", "extracted", "score", "inliers", "kpA", "kpB"]].head(min(5, len(sugg_df)))
            print(preview.to_string(index=False))
        else:
            print("[SUGGEST] No suggestions (empty or insufficient scores).")

    print(f"[SAVE] {report_path}")
    print("[DONE] evaluation complete.")


if __name__ == "__main__":
    main()
