"""
05_eval_inspect.py (revised)

Evaluate and package matching results with Positive-Only (PU) labels,
produce suggestions, and export an inspection set for quick human review.

Key changes vs. legacy:
- Positive-only evaluation: do NOT treat unlabeled predictions as errors.
  Reports Recall_L (on labeled positives) and Precision_LB (pessimistic lower bound).
  Optional Precision_hat from a sampled verification CSV.
- Robust I/O compatibility with both legacy and revised mapping formats.
- Suggestions: top-N non-selected pairs derived from pair_scores.
- Inspection set: copy selected pairs (by default accepted mapping) into a folder
  and write a manifest.csv, with safe fallbacks for Windows paths / missing files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:
    raise SystemExit(f"[ERR] pandas import 실패: {e}\n  pip install pandas")

# =========================
# ----- Utilities ---------
# =========================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + ((z * z) / (4 * n * n)))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# =========================
# ---- Mapping helpers ----
# =========================

@dataclass
class VariantPaths:
    color_low: Optional[Path]
    gray_low: Optional[Path]
    color_high: Optional[Path]
    gray_high: Optional[Path]
    src_abspath: Optional[Path]

def load_preprocess_mapping(mapping_json: Path) -> Tuple[Dict[str, VariantPaths], Dict[str, VariantPaths], Dict]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        mp = json.load(f)

    def _collect(cat: str) -> Dict[str, VariantPaths]:
        by_src = {}
        for src_rel, info in mp["by_src"][cat].items():
            v = info.get("variants", {})
            def _get(track: str, ch: str) -> Optional[Path]:
                p = v.get(track, {}).get(ch)
                return Path(p) if p else None
            by_src[src_rel] = VariantPaths(
                color_low=_get("low", "color"),
                gray_low=_get("low", "gray"),
                color_high=_get("high", "color"),
                gray_high=_get("high", "gray"),
                src_abspath=Path(info["src_abspath"]) if info.get("src_abspath") else None,
            )
        return by_src

    return _collect("extracted"), _collect("reference"), mp

def load_mapping_result(mapping_result_json: Path):
    """
    Supports legacy format (mapping: list of objects) and revised format
    (mapping: dict + assignments list).
    Returns:
      pairs: List[dict] with keys: ref_rel, ex_rel, score?, inliers?, inlier_ratio?, phase?
      total_refs: Optional[int]
      raw: original dict
    """
    with open(mapping_result_json, "r", encoding="utf-8") as f:
        M = json.load(f)

    pairs = []
    total_refs = None

    if isinstance(M.get("mapping"), list):
        # Legacy
        for row in M["mapping"]:
            pairs.append({
                "ref_rel": row.get("reference") or row.get("ref") or row.get("ref_rel"),
                "ex_rel": row.get("extracted") or row.get("ex") or row.get("ex_rel"),
                "score": row.get("score"),
                "inliers": row.get("inliers"),
                "inlier_ratio": row.get("inlier_ratio"),
                "phase": row.get("pass") or row.get("phase"),
                "detector": row.get("detector"),
                "ssim": row.get("ssim"),
                "ncc": row.get("ncc"),
            })
        total_refs = (M.get("stats", {}) or {}).get("references_total") or (M.get("stats", {}) or {}).get("references")
    elif isinstance(M.get("mapping"), dict):
        # Revised
        mp = M["mapping"]
        for k, ex in mp.items():
            ref_rel = k.split("|", 1)[-1] if "|" in k else k
            pairs.append({"ref_rel": ref_rel, "ex_rel": ex})
        total_refs = (M.get("stats", {}) or {}).get("references")
        # enrich with assignments
        for a in M.get("assignments", []):
            ref = a.get("reference") or a.get("ref")
            ref_rel = ref.split("|", 1)[-1] if ref and "|" in ref else (ref or a.get("ref_rel"))
            ex = a.get("extracted") or a.get("ex") or a.get("ex_rel")
            for p in pairs:
                if p["ref_rel"] == ref_rel and p["ex_rel"] == ex:
                    p.update({"score": a.get("score"), "phase": a.get("phase")})
                    break
    else:
        raise SystemExit("[ERR] mapping_result.json 에 'mapping'이 없습니다.")

    return pairs, total_refs, M

def read_pair_scores(scores_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if not scores_csv:
        return None
    if not scores_csv.exists():
        print(f"[WARN] pair_scores.csv를 찾을 수 없습니다: {scores_csv}")
        return None
    df = pd.read_csv(scores_csv)
    cols = [c.lower() for c in df.columns]
    # normalize columns
    if "ref" in cols and "ex" in cols:
        df["ref_rel"] = df["ref"].astype(str).str.replace(r"^[^|]*\|", "", regex=True)
        df["ex_rel"] = df["ex"].astype(str)
    elif "ref_rel" in cols and "ex_rel" in cols:
        # already good
        pass
    else:
        # try alternative names
        raise SystemExit(f"[ERR] pair_scores.csv의 컬럼을 해석할 수 없습니다: {df.columns.tolist()}")

    # keep commonly used columns if present
    keep = ["ref_rel", "ex_rel"]
    for c in ["score", "inliers", "inlier_ratio", "phash_dist", "hist_dist", "detector", "ssim", "ncc", "phase"]:
        if c in df.columns:
            keep.append(c)
    return df[keep]

def read_labels_csv(labels_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if not labels_csv:
        return None
    if not labels_csv.exists():
        print(f"[WARN] labels CSV를 찾을 수 없습니다: {labels_csv}")
        return None
    df = pd.read_csv(labels_csv)
    # Expect columns similar to: reference_path, extracted_path, label(1)
    # normalize names
    cols = {c.lower(): c for c in df.columns}
    ref_col = cols.get("reference_path") or cols.get("ref") or cols.get("ref_rel")
    ex_col  = cols.get("extracted_path") or cols.get("ex") or cols.get("ex_rel")
    lbl_col = cols.get("label") or cols.get("y") or cols.get("is_tp")
    if not (ref_col and ex_col):
        raise SystemExit(f"[ERR] labels CSV 컬럼을 해석할 수 없습니다: {df.columns.tolist()}")
    df = df.rename(columns={ref_col: "ref_rel", ex_col: "ex_rel"})
    if lbl_col and lbl_col in df.columns:
        df = df[df[lbl_col].astype(int) == 1]
    df = df[["ref_rel", "ex_rel"]].dropna()
    return df

# =========================
# ---- Suggestions --------
# =========================

def build_suggestions(df_scores: Optional[pd.DataFrame],
                      accepted_pairs: List[Dict],
                      out_csv: Path,
                      topN: int = 20,
                      per_ref_cap: int = 3,
                      only_unmatched: bool = False,
                      unmatched_refs: Optional[Iterable[str]] = None) -> Optional[Path]:
    if df_scores is None or df_scores.empty:
        print("[INFO] pair_scores가 없어 suggestions를 생략합니다.")
        return None
    accepted = {(p["ref_rel"], p["ex_rel"]) for p in accepted_pairs}
    df = df_scores.copy()
    # Filter to unmatched references if requested
    if only_unmatched and unmatched_refs is not None:
        df = df[df["ref_rel"].isin(set(unmatched_refs))]
    df = df.sort_values(by=[c for c in ["score", "inliers", "inlier_ratio"] if c in df.columns],
                        ascending=False)
    picked = []
    per_ref_count: Dict[str, int] = {}
    for _, r in df.iterrows():
        key = (r["ref_rel"], r["ex_rel"])
        if key in accepted:
            continue
        k = per_ref_count.get(r["ref_rel"], 0)
        if k >= per_ref_cap:
            continue
        per_ref_count[r["ref_rel"]] = k + 1
        picked.append(r.to_dict())
        if len(picked) >= topN:
            break
    if not picked:
        print("[INFO] 추천할 후보가 없습니다.")
        return None
    ensure_parent(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
        w.writeheader()
        for r in picked:
            w.writerow(r)
    print(f"[INFO] suggestions 저장: {out_csv} (rows={len(picked)})")
    return out_csv

# =========================
# ----- Inspection --------
# =========================

def resolve_variant_path(vp: VariantPaths, variant: str) -> Optional[Path]:
    if variant == "raw":
        return vp.src_abspath if vp and vp.src_abspath and Path(vp.src_abspath).exists() else None
    tr = "low" if "low" in variant else "high"
    ch = "gray" if "gray" in variant else "color"
    attr = f"{ch}_{tr}"
    p = getattr(vp, attr, None)
    if p and Path(p).exists():
        return Path(p)
    return None

def try_basename_fallback(basename: str, roots: List[str]) -> Optional[Path]:
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for rp, _, files in os.walk(root):
            if basename in files:
                return Path(rp) / basename
    return None

def export_inspection(pairs: List[Dict],
                      extracted_map: Dict[str, VariantPaths],
                      reference_map: Dict[str, VariantPaths],
                      roots: Dict[str, str],
                      out_dir: Path,
                      copy_variant: str,
                      allow_basename_fallback: bool = True) -> Tuple[int, int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(len(pairs))))
    rows = []
    ok = 0

    for i, p in enumerate(pairs, 1):
        ref_rel = p["ref_rel"]
        ex_rel  = p["ex_rel"]
        vp_ref = reference_map.get(ref_rel)
        vp_ex  = extracted_map.get(ex_rel)

        ref_src = resolve_variant_path(vp_ref, copy_variant) if vp_ref else None
        ex_src  = resolve_variant_path(vp_ex, copy_variant) if vp_ex else None

        # fallbacks
        if allow_basename_fallback:
            if not (ref_src and ref_src.exists()):
                cand = try_basename_fallback(os.path.basename(ref_rel), [roots.get("reference_in", ""), roots.get("reference_out", "")])
                if cand: ref_src = cand
            if not (ex_src and ex_src.exists()):
                cand = try_basename_fallback(os.path.basename(ex_rel), [roots.get("extracted_in", ""), roots.get("extracted_out", "")])
                if cand: ex_src = cand

        idx = str(i).zfill(width)
        ref_dst = out_dir / f"{idx}_ref_{os.path.basename(ref_rel)}"
        ex_dst  = out_dir / f"{idx}_ex_{os.path.basename(ex_rel)}"

        status = "ok"
        try:
            if ref_src and Path(ref_src).exists():
                shutil.copy2(ref_src, ref_dst)
            else:
                status = "miss_ref"
            if ex_src and Path(ex_src).exists():
                shutil.copy2(ex_src, ex_dst)
            else:
                status = status + ("|miss_ex" if status != "ok" else "miss_ex")
        except Exception as e:
            status = status + f"|copy_err:{e}"

        if status == "ok":
            ok += 1

        rows.append({
            "index": i,
            "ref_rel": ref_rel,
            "ex_rel": ex_rel,
            "phase": p.get("phase", ""),
            "score": p.get("score", ""),
            "inliers": p.get("inliers", ""),
            "inlier_ratio": p.get("inlier_ratio", ""),
            "ref_src": str(ref_src) if ref_src else "",
            "ex_src": str(ex_src) if ex_src else "",
            "status": status,
        })

    # Write manifest
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["index","ref_rel","ex_rel","phase","score","inliers","inlier_ratio","ref_src","ex_src","status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return ok, len(rows), manifest

# =========================
# --------- CLI -----------
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate mapping (PU labels), produce suggestions and inspection set.")
    ap.add_argument("--mapping-json", required=True, help="mapping_result.json")
    ap.add_argument("--scores-csv", default=None, help="pair_scores.csv (optional; some mapping_result may include its path)")
    ap.add_argument("--mapping-meta", default="preprocess_mapping.json", help="preprocess_mapping.json for path resolution")
    ap.add_argument("--labels-csv", default=None, help="Positive-only labels CSV (columns: reference_path, extracted_path, label=1)")
    ap.add_argument("--sample-csv", default=None, help="Optional sampled verification CSV for precision_hat (columns: ref_rel, ex_rel, is_tp{0/1})")

    ap.add_argument("--export-report", default="evaluation_report.json", help="Where to write evaluation JSON report")
    ap.add_argument("--inspect-out", default="inspect_pairs", help="Folder to export inspection images")
    ap.add_argument("--copy-variant", default="pre_low_color",
                    choices=["raw", "pre_low_gray", "pre_low_color", "pre_high_gray", "pre_high_color"],
                    help="Which image variant to copy for inspection")
    ap.add_argument("--allow-basename-fallback", action="store_true", help="If exact path missing, try basename search under roots")

    ap.add_argument("--suggest-n", type=int, default=20, help="Top-N suggestions to export")
    ap.add_argument("--suggest-out", default="suggestions_topN.csv", help="Output CSV for suggestions")
    ap.add_argument("--suggest-only-unmatched", action="store_true", help="Recommend only for unmatched references")

    return ap.parse_args()

# =========================
# --------- MAIN ----------
# =========================

def main() -> int:
    args = parse_args()

    mapping_json = Path(args.mapping_json)
    if not mapping_json.exists():
        print(f"[ERR] mapping_result.json 없음: {mapping_json}")
        return 2

    pairs, total_refs_from_map, raw_map = load_mapping_result(mapping_json)

    # Optionally derive scores_csv from mapping_result if present
    scores_csv = Path(args.scores_csv) if args.scores_csv else None
    if not scores_csv and isinstance(raw_map.get("scores_csv"), str):
        scores_csv = Path(raw_map["scores_csv"])

    df_scores = read_pair_scores(scores_csv) if scores_csv else None

    # Load mapping meta for copying
    extracted_map, reference_map, mp = load_preprocess_mapping(Path(args.mapping_meta))

    # Determine total references
    total_refs = total_refs_from_map or len(reference_map)
    assigned = len(pairs)
    coverage = float(assigned) / float(max(1, total_refs))

    # Positive-only labels
    labels_df = read_labels_csv(Path(args.labels_csv)) if args.labels_csv else None
    metrics = {
        "labels_used": bool(labels_df is not None and not labels_df.empty)
    }
    if metrics["labels_used"]:
        L = set((r.ref_rel, r.ex_rel) for r in labels_df.itertuples(index=False))
        P = set((p["ref_rel"], p["ex_rel"]) for p in pairs)
        tp_L = len(P & L)
        fn_L = len(L - P)
        # Recall over labeled positives
        recall_L = tp_L / max(1, len(L))
        r_lo, r_hi = wilson_ci(tp_L, max(1, len(L)))
        # Precision lower bound (treat all unlabeled predictions as FP)
        prec_LB = tp_L / max(1, assigned)
        p_lo, p_hi = wilson_ci(tp_L, max(1, assigned))

        metrics.update({
            "recall_L": recall_L,
            "recall_L_ci95": [r_lo, r_hi],
            "precision_LB": prec_LB,
            "precision_LB_ci95": [p_lo, p_hi],
            "tp_L": tp_L,
            "fn_L": fn_L,
            "labels_total": len(L)
        })
    else:
        metrics.update({
            "recall_L": None,
            "recall_L_ci95": None,
            "precision_LB": None,
            "precision_LB_ci95": None,
            "tp_L": None,
            "fn_L": None,
            "labels_total": 0
        })

    # Precision_hat from sampled verification (optional)
    if args.sample_csv and Path(args.sample_csv).exists():
        sdf = pd.read_csv(args.sample_csv)
        cols = {c.lower(): c for c in sdf.columns}
        lblcol = cols.get("is_tp") or cols.get("label")
        if lblcol:
            n = int(len(sdf))
            tp = int((sdf[lblcol].astype(int) == 1).sum())
            prec_hat = tp / max(1, n)
            lo, hi = wilson_ci(tp, max(1, n))
            metrics.update({
                "precision_hat": prec_hat,
                "precision_hat_ci95": [lo, hi],
                "precision_hat_sample_n": n
            })
        else:
            print("[WARN] sample CSV에 is_tp/label 컬럼이 없어 precision_hat을 계산하지 않습니다.")

    # Suggestions
    unmatched_refs = None
    if isinstance(raw_map.get("unmatched_reference"), list):
        unmatched_refs = [k.split("|", 1)[-1] if "|" in k else k for k in raw_map["unmatched_reference"]]
    elif isinstance(raw_map.get("unmatched_references"), list):
        unmatched_refs = raw_map["unmatched_references"]

    suggest_csv_path = None
    if args.suggest_n > 0:
        suggest_csv_path = build_suggestions(
            df_scores=df_scores,
            accepted_pairs=pairs,
            out_csv=Path(args.suggest_out),
            topN=int(args.suggest_n),
            per_ref_cap=3,
            only_unmatched=bool(args.suggest_only_unmatched),
            unmatched_refs=unmatched_refs
        )

    # Inspection export (accepted mapping)
    insp_dir = Path(args.inspect_out)
    # choose pairs for inspection = accepted mapping
    ok_count, total_count, manifest_path = export_inspection(
        pairs=pairs,
        extracted_map=extracted_map, reference_map=reference_map,
        roots=mp.get("roots", {}),
        out_dir=insp_dir,
        copy_variant=args.copy_variant,
        allow_basename_fallback=bool(args.allow_basename_fallback)
    )

    # Report
    report = {
        "version": 2,
        "source_files": {
            "mapping_result_json": str(mapping_json.as_posix()),
            "pair_scores_csv": str(scores_csv.as_posix()) if scores_csv else None,
            "preprocess_mapping_json": str(Path(args.mapping_meta).as_posix()),
            "labels_csv": str(Path(args.labels_csv).as_posix()) if args.labels_csv else None,
        },
        "coverage": {
            "assigned": int(assigned),
            "references_total": int(total_refs),
            "coverage": coverage
        },
        "metrics": metrics,
        "suggestions": {
            "suggestions_csv": str(Path(args.suggest_out).as_posix()) if suggest_csv_path else None,
            "topN": int(args.suggest_n),
            "only_unmatched": bool(args.suggest_only_unmatched)
        },
        "inspection": {
            "out_dir": str(insp_dir.as_posix()),
            "manifest_csv": str(manifest_path.as_posix()),
            "copied_ok": int(ok_count),
            "total_rows": int(total_count),
            "copy_variant": args.copy_variant,
            "allow_basename_fallback": bool(args.allow_basename_fallback)
        }
    }

    out_path = Path(args.export_report)
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] coverage: {assigned}/{total_refs} = {coverage:.3f}")
    if metrics.get("labels_used"):
        print(f"[INFO] Recall_L: {metrics['recall_L']:.3f}  (95% CI {metrics['recall_L_ci95'][0]:.3f}–{metrics['recall_L_ci95'][1]:.3f})")
        print(f"[INFO] Precision_LB: {metrics['precision_LB']:.3f}  (95% CI {metrics['precision_LB_ci95'][0]:.3f}–{metrics['precision_LB_ci95'][1]:.3f})")
    if "precision_hat" in metrics:
        print(f"[INFO] Precision_hat(sample): {metrics['precision_hat']:.3f}  (95% CI {metrics['precision_hat_ci95'][0]:.3f}–{metrics['precision_hat_ci95'][1]:.3f}), n={metrics['precision_hat_sample_n']}")

    print(f"[INFO] suggestions: {suggest_csv_path if suggest_csv_path else 'N/A'}")
    print(f"[INFO] inspection: copied_ok={ok_count}/{total_count}, dir={insp_dir}")
    print(f"[INFO] report saved: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
