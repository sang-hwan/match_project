"""
05_eval_inspect.py

Evaluate final mapping quality and prepare a human inspection set.

Functions
---------
1) Evaluation
   - If --labels-csv is given (reference,extracted pairs), compute Precision / Recall / F1.
     * Robust header detection: reference/ref/original, extracted/ex/ext etc.
     * Wilson score intervals (95%) for Precision, Recall.
   - Always report coverage: assigned_references / total_references (from mapping_result.json).

2) Suggestions (optional)
   - From pair_scores.csv, list top-N non-accepted pairs sorted by score (or inliers).
   - Output suggestions_topN.csv (for manual review).

3) Inspection set
   - Copy matched pairs into a flat folder structure for quick eyeballing:
       inspect_out/
         0001_ref_<basename>.<ext>, 0001_ex_<basename>.<ext>
         0002_ref_...,             0002_ex_...
   - Manifest CSV (reference_rel, extracted_rel, ref_abs, ex_abs).
   - Sources:
       * default: raw inputs from preprocess_mapping.json ("roots.extracted_in", "roots.reference_in") + src_rel paths
       * or preprocessed low-track variants when --copy-variant pre_low_gray|pre_low_color is selected.
   - Fallback: if raw absolute path missing & --allow-basename-fallback, search by basename under roots.

Usage
-----
python 05_eval_inspect.py \
  --mapping-json .\map_dist\mapping_result.json \
  --scores-csv   .\map_dist\pair_scores.csv \
  --mapping-meta preprocess_mapping.json \
  --labels-csv   hand_label.csv \
  --suggest-n    20 \
  --inspect-out  inspect_pairs \
  --export-manifest inspect_pairs/manifest.csv \
  --copy-variant raw \
  --allow-basename-fallback

Notes
-----
- All paths are treated as case-sensitive by default. If needed, pre-normalize your label paths.
- This tool does not draw charts; it writes JSON/CSV artifacts and prints a concise summary.
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

# --------------------------
# Basic I/O
# --------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_rel(s: str) -> str:
    # Normalize to POSIX style relative for stable set/dict keys
    return str(Path(s).as_posix())

# --------------------------
# Mapping/meta schema helpers
# --------------------------

@dataclass
class VariantPaths:
    color_low: Optional[str]
    gray_low: Optional[str]
    color_high: Optional[str]
    gray_high: Optional[str]

def load_meta_variants(mp: dict, cat: str) -> Dict[str, VariantPaths]:
    out: Dict[str, VariantPaths] = {}
    by_src = mp.get("by_src", {}).get(cat, {})
    for rel, info in by_src.items():
        v = info.get("variants", {})
        def _get(track: str, ch: str) -> Optional[str]:
            p = v.get(track, {}).get(ch)
            return str(p) if p else None
        out[norm_rel(rel)] = VariantPaths(
            color_low = _get("low","color"),
            gray_low  = _get("low","gray"),
            color_high= _get("high","color"),
            gray_high = _get("high","gray"),
        )
    return out

def build_basename_index(root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    if not root.exists():
        return idx
    for p in root.rglob("*"):
        if p.is_file():
            idx.setdefault(p.name.lower(), []).append(p)
    return idx

# --------------------------
# Labels CSV (ground truth)
# --------------------------

LabelRow = Tuple[str, str]  # (reference_rel, extracted_rel)

def auto_headers(headers: List[str]) -> Tuple[Optional[str], Optional[str]]:
    hs = [h.strip().lower() for h in headers]
    ref_keys = {"reference","ref","original","reference_rel","ref_rel","reference_path","ref_path"}
    ex_keys  = {"extracted","extract","ex","ext","extracted_rel","extracted_path","ex_path"}
    ref_h = next((h for h in hs if h in ref_keys), None)
    ex_h  = next((h for h in hs if h in ex_keys), None)
    return ref_h, ex_h

def read_labels_csv(path: Path,
                    meta: dict,
                    allow_basename_match: bool = True) -> Set[LabelRow]:
    """Read labels CSV -> set of (ref_rel, ex_rel). Robust to header variations."""
    gt: Set[LabelRow] = set()
    ref_root = Path(meta.get("roots",{}).get("reference_in",""))
    ex_root  = Path(meta.get("roots",{}).get("extracted_in",""))

    # Build basename indices for fallback mapping (only if requested)
    ref_base_idx = build_basename_index(ref_root) if allow_basename_match else {}
    ex_base_idx  = build_basename_index(ex_root)  if allow_basename_match else {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        ref_h, ex_h = auto_headers(r.fieldnames or [])
        if not ref_h or not ex_h:
            raise SystemExit(f"[ERR] 라벨 CSV 헤더를 인식할 수 없습니다. headers={r.fieldnames}")

        for row in r:
            ref_raw = (row.get(ref_h) or "").strip()
            ex_raw  = (row.get(ex_h) or "").strip()
            if not ref_raw or not ex_raw:
                continue

            # Try to normalize to meta-relative
            ref_rel = derive_rel_path(ref_raw, ref_root, ref_base_idx)
            ex_rel  = derive_rel_path(ex_raw,  ex_root,  ex_base_idx)
            gt.add((norm_rel(ref_rel), norm_rel(ex_rel)))
    return gt

def derive_rel_path(given: str, root: Path, base_idx: Dict[str, List[Path]]) -> str:
    """
    Try to convert a given path or filename to a root-relative posix path string.
    Strategy:
      1) If absolute under root, make it relative.
      2) If already relative (exists under root), keep as-is.
      3) If plain basename and unique match exists under root, use it.
      4) Fallback: return given as-is (user responsibility).
    """
    p = Path(given)
    if p.is_absolute():
        try:
            return str(p.relative_to(root).as_posix())
        except Exception:
            # fallthrough
            pass
    else:
        # treat as relative from root
        if (root / p).exists():
            return str((root / p).relative_to(root).as_posix())

    # basename fallback
    name = p.name.lower()
    if name and name in base_idx and len(base_idx[name]) == 1:
        return str(base_idx[name][0].relative_to(root).as_posix())

    # give up
    return str(p.as_posix())

# --------------------------
# Metrics
# --------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson CI for binomial proportion (k successes out of n)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z*z/(2*n)) / denom
    half   = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

@dataclass
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    p_ci: Tuple[float, float]
    r_ci: Tuple[float, float]

def compute_prf(pred: Set[LabelRow], gt: Set[LabelRow]) -> PRF:
    tp = len(pred.intersection(gt))
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    p_ci = wilson_interval(tp, tp+fp) if (tp+fp) > 0 else (0.0, 0.0)
    r_ci = wilson_interval(tp, tp+fn) if (tp+fn) > 0 else (0.0, 0.0)
    return PRF(tp, fp, fn, precision, recall, f1, p_ci, r_ci)

# --------------------------
# Mapping & scores loading
# --------------------------

@dataclass
class MappingPair:
    reference: str
    extracted: str
    inliers: int
    inlier_ratio: float
    good_matches: int
    pass_label: str
    score: float

def load_mapping_pairs(mapping_json: Path) -> Tuple[List[MappingPair], dict]:
    mj = load_json(mapping_json)
    pairs = []
    for m in mj.get("mapping", []):
        pairs.append(MappingPair(
            reference    = norm_rel(m["reference"]),
            extracted    = norm_rel(m["extracted"]),
            inliers      = int(m.get("inliers", 0)),
            inlier_ratio = float(m.get("inlier_ratio", 0.0)),
            good_matches = int(m.get("good_matches", 0)),
            pass_label   = str(m.get("pass","")),
            score        = float(m.get("score", 0.0)),
        ))
    return pairs, mj

@dataclass
class ScoredRow:
    reference: str
    extracted: str
    inliers: int
    inlier_ratio: float
    good_matches: int
    phash_dist: int
    hist_dist: float
    score: float
    pass_label: str

def load_scores_csv(scores_csv: Path) -> List[ScoredRow]:
    if not scores_csv.exists():
        return []
    out: List[ScoredRow] = []
    with open(scores_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append(ScoredRow(
                    reference    = norm_rel(row["ref_rel"]),
                    extracted    = norm_rel(row["ex_rel"]),
                    good_matches = int(float(row.get("good_matches", 0))),
                    inliers      = int(float(row.get("inliers", 0))),
                    inlier_ratio = float(row.get("inlier_ratio", 0.0)),
                    phash_dist   = int(float(row.get("phash_dist", -1))),
                    hist_dist    = float(row.get("hist_dist", 1.0)),
                    pass_label   = row.get("pass",""),
                    score        = float(row.get("score", 0.0)),
                ))
            except Exception:
                continue
    return out

# --------------------------
# Suggestions
# --------------------------

def write_suggestions(scores: List[ScoredRow],
                      assigned: Set[LabelRow],
                      out_csv: Path,
                      top_n: int) -> int:
    """
    Pick top-N non-accepted pairs by score (descending), excluding already assigned.
    """
    pool = [s for s in scores if s.pass_label.strip() == ""]
    # Sort primarily by score, then inliers, then (−hist, +phash)
    pool.sort(key=lambda s: (s.score, s.inliers, -s.hist_dist, -s.phash_dist), reverse=True)

    ensure_dir(out_csv.parent)
    cnt = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank","reference","extracted","score","inliers","inlier_ratio","good_matches",
                    "phash_dist","hist_dist"])
        for i, s in enumerate(pool, 1):
            key = (s.reference, s.extracted)
            if key in assigned:
                continue
            w.writerow([i, s.reference, s.extracted,
                        f"{s.score:.6f}", s.inliers, f"{s.inlier_ratio:.6f}",
                        s.good_matches, s.phash_dist, f"{s.hist_dist:.6f}"])
            cnt += 1
            if cnt >= top_n:
                break
    return cnt

# --------------------------
# Inspection copy
# --------------------------

def resolve_abs_paths(ref_rel: str, ex_rel: str,
                      meta: dict,
                      variants_ref: Dict[str, VariantPaths],
                      variants_ex: Dict[str, VariantPaths],
                      copy_variant: str,
                      allow_basename_fallback: bool) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Decide absolute source files to copy for ref/ex according to copy_variant.
    copy_variant: raw | pre_low_gray | pre_low_color
    """
    copy_variant = copy_variant.lower()
    if copy_variant == "raw":
        ref_root = Path(meta.get("roots",{}).get("reference_in",""))
        ex_root  = Path(meta.get("roots",{}).get("extracted_in",""))
        ref_abs = (ref_root / ref_rel) if ref_rel else None
        ex_abs  = (ex_root / ex_rel) if ex_rel else None
    else:
        # preprocessed
        vr = variants_ref.get(ref_rel)
        ve = variants_ex.get(ex_rel)
        if copy_variant == "pre_low_gray":
            ref_abs = Path(vr.gray_low) if (vr and vr.gray_low) else None
            ex_abs  = Path(ve.gray_low) if (ve and ve.gray_low) else None
        else:  # pre_low_color
            ref_abs = Path(vr.color_low) if (vr and vr.color_low) else None
            ex_abs  = Path(ve.color_low) if (ve and ve.color_low) else None

    # Fallback by basename search
    if allow_basename_fallback:
        if (ref_abs is None or not ref_abs.exists()) and copy_variant == "raw":
            ref_root = Path(meta.get("roots",{}).get("reference_in",""))
            ref_abs = try_find_by_basename(ref_rel, ref_root) or ref_abs
        if (ex_abs is None or not ex_abs.exists()) and copy_variant == "raw":
            ex_root = Path(meta.get("roots",{}).get("extracted_in",""))
            ex_abs  = try_find_by_basename(ex_rel, ex_root) or ex_abs

    return ref_abs if ref_abs and ref_abs.exists() else None, \
           ex_abs  if ex_abs  and ex_abs.exists()  else None

def try_find_by_basename(rel: str, root: Path) -> Optional[Path]:
    cand = root.rglob(Path(rel).name)
    try:
        p = next(cand)
        return p
    except StopIteration:
        return None

def copy_pairs_for_inspection(pairs: List[MappingPair],
                              inspect_out: Path,
                              export_manifest: Path,
                              meta: dict,
                              variants_ref: Dict[str, VariantPaths],
                              variants_ex: Dict[str, VariantPaths],
                              copy_variant: str,
                              allow_basename_fallback: bool,
                              limit: int = 0) -> Tuple[int,int]:
    """
    Returns: (copied_pairs, skipped_pairs)
    """
    ensure_dir(inspect_out)
    rows: List[List[str]] = []
    copied = skipped = 0

    for idx, p in enumerate(pairs, 1):
        if limit and copied >= limit:
            break

        ref_abs, ex_abs = resolve_abs_paths(
            p.reference, p.extracted, meta, variants_ref, variants_ex,
            copy_variant, allow_basename_fallback
        )
        if not ref_abs or not ex_abs:
            skipped += 1
            continue

        # Make numbered filenames
        base = f"{copied+1:04d}"
        ref_dst = inspect_out / f"{base}_ref_{Path(ref_abs).name}"
        ex_dst  = inspect_out / f"{base}_ex_{Path(ex_abs).name}"

        try:
            shutil.copy2(ref_abs, ref_dst)
            shutil.copy2(ex_abs,  ex_dst)
            copied += 1
            rows.append([
                base, p.reference, p.extracted,
                str(ref_abs), str(ex_abs),
                str(ref_dst), str(ex_dst),
                p.pass_label, f"{p.inliers}", f"{p.inlier_ratio:.6f}", f"{p.score:.6f}"
            ])
        except Exception:
            skipped += 1

    # Write manifest
    ensure_dir(export_manifest.parent)
    with open(export_manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id","reference_rel","extracted_rel","ref_abs","ex_abs",
                    "ref_dst","ex_dst","pass","inliers","inlier_ratio","score"])
        for r in rows:
            w.writerow(r)

    return copied, skipped

# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate mapping and export a folder for human inspection.")
    ap.add_argument("--mapping-json", required=True, help="mapping_result.json from 04_match.py")
    ap.add_argument("--scores-csv", default="", help="pair_scores.csv (for suggestions)")
    ap.add_argument("--mapping-meta", default="preprocess_mapping.json", help="preprocess mapping meta (for roots/variants)")

    ap.add_argument("--labels-csv", default="", help="Optional ground-truth label CSV to compute P/R/F1")
    ap.add_argument("--suggest-n", type=int, default=0, help="If >0, write suggestions_topN.csv")

    ap.add_argument("--inspect-out", default="inspect_pairs", help="Folder to copy matched pairs")
    ap.add_argument("--export-manifest", default="", help="CSV manifest path (default: <inspect-out>/manifest.csv)")
    ap.add_argument("--copy-variant", choices=["raw","pre_low_gray","pre_low_color"], default="raw", help="Which images to copy for inspection")
    ap.add_argument("--allow-basename-fallback", action="store_true", help="When raw path missing, search by basename under roots")
    ap.add_argument("--limit-inspect", type=int, default=0, help="Max number of pairs to copy (0=all)")

    ap.add_argument("--report-json", default="evaluation_report.json", help="Evaluation report JSON path")

    return ap.parse_args()

# --------------------------
# Main
# --------------------------

def main() -> int:
    args = parse_args()

    mapping_json = Path(args.mapping_json)
    if not mapping_json.exists():
        print(f"[ERR] mapping_result.json not found: {mapping_json}")
        return 2
    pairs, mj = load_mapping_pairs(mapping_json)

    # Load meta for roots/variants
    meta_path = Path(args.mapping_meta)
    if not meta_path.exists():
        print(f"[ERR] preprocess_mapping.json not found: {meta_path}")
        return 2
    meta = load_json(meta_path)
    variants_ref = load_meta_variants(meta, "reference")
    variants_ex  = load_meta_variants(meta, "extracted")

    # Predicted set for metrics
    pred_set: Set[LabelRow] = {(p.reference, p.extracted) for p in pairs}

    # Compute reference coverage
    total_refs = int(mj.get("stats",{}).get("references_total", 0))
    assigned   = int(mj.get("stats",{}).get("assigned", len(pairs)))
    coverage   = (assigned / total_refs) if total_refs > 0 else 0.0

    # Labels-based metrics (optional)
    prf: Optional[PRF] = None
    labels_used = False
    if args.labels_csv:
        labels_path = Path(args.labels_csv)
        if not labels_path.exists():
            print(f"[WARN] labels CSV not found: {labels_path}")
        else:
            gt = read_labels_csv(labels_path, meta, allow_basename_match=True)
            prf = compute_prf(pred_set, gt)
            labels_used = True

    # Suggestions (optional)
    suggestions_written = 0
    scores_csv = Path(args.scores_csv) if args.scores_csv else Path("")
    if args.suggest_n and scores_csv.exists():
        scored = load_scores_csv(scores_csv)
        out_csv = Path("suggestions_topN.csv")
        suggestions_written = write_suggestions(scored, pred_set, out_csv, args.suggest_n)

    # Inspection copy
    inspect_out = Path(args.inspect_out)
    export_manifest = Path(args.export_manifest) if args.export_manifest else (inspect_out / "manifest.csv")
    copied, skipped = copy_pairs_for_inspection(
        pairs, inspect_out, export_manifest, meta,
        variants_ref, variants_ex, args.copy_variant,
        args.allow_basename_fallback, args.limit_inspect
    )

    # Report JSON
    report = {
        "version": 1,
        "source_files": {
            "mapping_json": str(mapping_json),
            "scores_csv": str(scores_csv) if scores_csv else "",
            "mapping_meta": str(meta_path),
            "labels_csv": str(Path(args.labels_csv)) if args.labels_csv else "",
        },
        "coverage": {
            "assigned": assigned,
            "references_total": total_refs,
            "coverage_ref": coverage,
        },
        "metrics": {
            "precision": None if not prf else prf.precision,
            "recall":    None if not prf else prf.recall,
            "f1":        None if not prf else prf.f1,
            "precision_ci95": None if not prf else prf.p_ci,
            "recall_ci95":    None if not prf else prf.r_ci,
            "tp_fp_fn": None if not prf else {"tp": prf.tp, "fp": prf.fp, "fn": prf.fn},
            "labels_used": labels_used,
        },
        "suggestions": {
            "topN": int(args.suggest_n),
            "written": int(suggestions_written),
            "file": "suggestions_topN.csv" if suggestions_written > 0 else ""
        },
        "inspection": {
            "out_dir": str(inspect_out),
            "manifest_csv": str(export_manifest),
            "copy_variant": args.copy_variant,
            "copied_pairs": copied,
            "skipped_pairs": skipped,
            "limit": int(args.limit_inspect),
            "allow_basename_fallback": bool(args.allow_basename_fallback),
        }
    }
    save_json(Path(args.report_json), report)

    # Console summary
    print("=== Evaluation Summary ===")
    print(f"Assigned pairs         : {assigned} / {total_refs} (coverage={coverage:.3f})")
    if labels_used and prf:
        print(f"Precision              : {prf.precision:.3f}  (95% CI {prf.p_ci[0]:.3f}-{prf.p_ci[1]:.3f})")
        print(f"Recall                 : {prf.recall:.3f}     (95% CI {prf.r_ci[0]:.3f}-{prf.r_ci[1]:.3f})")
        print(f"F1                     : {prf.f1:.3f}         TP={prf.tp} FP={prf.fp} FN={prf.fn}")
    else:
        print("Labels CSV             : not provided (P/R/F1 skipped)")
    if suggestions_written:
        print(f"Suggestions            : suggestions_topN.csv (rows={suggestions_written})")
    print(f"Inspect copied/ skipped: {copied} / {skipped}")
    print(f"Report JSON            : {args.report_json}")
    print(f"Manifest CSV           : {export_manifest}")
    print("==========================")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
