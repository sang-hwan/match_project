"""
05_eval_inspect.py  (rev. feedback-loop ready)

What it does
------------
1) Evaluate mapping quality with positive-only labels (optional):
   - Recall_L (on labeled positives), Precision_LB (pessimistic lower bound)
   - Optional Wilson CI for sampled verification file

2) Build inspection pack:
   - Copy paired images (variant-selectable) into an inspect folder
   - Emit manifest.csv with rich metadata

3) Suggest top-N alternatives:
   - From pair_scores.csv, list best unelected candidates per ref
   - Optionally only for unmatched refs

4) Emit feedback.json (for 02/03/04 loop):
   - redo_preprocess.refs        : references needing better preprocessing
   - expand_candidates.(global/per_ref) : refs to search wider (pHash radius / hist threshold / min candidates)
   - tighten_verification.refs   : assigned-but-borderline to tighten A-guard
   - allow_reuse.enabled         : hint for reuse assignment if duplicated content is detected

Inputs
------
- --mapping-json    : mapping_result.json from 04_match.py
- --scores-csv      : pair_scores.csv (optional; fallback to mapping_result.scores_csv)
- --mapping-meta    : preprocess_mapping.json from 02_preprocess.py
- --labels-csv      : (optional) positives only. CSV with columns:
                      'ref','ex'  (preferred) or 'reference','extracted'
                      If 'ex' column missing, label means "ref exists positively in doc" (no ex constraint).
- --sample-verify   : (optional) CSV with columns 'ref','is_correct' (0/1) for Wilson CI estimate of precision

Artifacts
---------
- evaluation_report.json
- inspect_pairs/ (images + manifest.csv)
- suggestions_topN.csv
- feedback.json

Usage
-----
python 05_eval_inspect.py \
  --mapping-json artifacts/mapping_result.json \
  --scores-csv   artifacts/pair_scores.csv \
  --mapping-meta artifacts/preprocess_mapping.json \
  --labels-csv   hand_label.csv \
  --inspect-out  inspect_pairs \
  --copy-variant pre_low_color \
  --allow-basename-fallback \
  --suggest-n 20 --suggest-only-unmatched \
  --export-report  artifacts/evaluation_report.json \
  --export-feedback artifacts/feedback.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

# ============== I/O helpers ==============

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path or not path.exists():
        return []
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({k: (v if v is not None else "") for k, v in row.items()})
    return out

def safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(dst))
        return True
    except Exception:
        return False

# ============== Mapping loaders ==============

@dataclass
class VariantPaths:
    src_abspath: Optional[Path]
    low_color: Optional[Path]
    low_gray: Optional[Path]
    high_color: Optional[Path]
    high_gray: Optional[Path]

def _vp_from_entry(entry: dict) -> VariantPaths:
    v = entry.get("variants", {})
    src_abspath = Path(entry.get("src_abspath")) if entry.get("src_abspath") else None
    def _get(track: str, ch: str) -> Optional[Path]:
        p = v.get(track, {}).get(ch)
        return Path(p) if p else None
    return VariantPaths(
        src_abspath=src_abspath,
        low_color=_get("low", "color"),
        low_gray=_get("low", "gray"),
        high_color=_get("high", "color"),
        high_gray=_get("high", "gray"),
    )

def load_preprocess_mapping(mapping_meta: Path) -> Tuple[Dict[str, VariantPaths], Dict[str, VariantPaths], dict]:
    mp = read_json(mapping_meta)
    ex_map: Dict[str, VariantPaths] = {}
    rf_map: Dict[str, VariantPaths] = {}
    for cat in ("extracted", "reference"):
        by_src = mp.get("by_src", {}).get(cat, {})
        for src_rel, entry in by_src.items():
            vp = _vp_from_entry(entry)
            if cat == "extracted":
                ex_map[src_rel] = vp
            else:
                rf_map[src_rel] = vp
    return ex_map, rf_map, mp

# ============== Labels / Metrics ==============

@dataclass
class Labels:
    # positive-only labels
    ref_to_exset: Dict[str, Set[str]]  # if empty set => "ref exists positively somewhere"
    has_ex_column: bool

def load_labels(path: Optional[Path]) -> Labels:
    ref_to_exset: Dict[str, Set[str]] = {}
    has_ex = False
    if not path or not path.exists():
        return Labels(ref_to_exset, has_ex)
    rows = read_csv_rows(path)
    # column normalization
    def pick(row: dict, *names) -> Optional[str]:
        for n in names:
            if n in row and row[n] != "":
                return row[n]
        return None
    for row in rows:
        ref = pick(row, "ref", "reference", "ref_rel", "reference_rel")
        ex  = pick(row, "ex", "extracted", "ex_rel", "extracted_rel")
        if not ref:
            # allow rows with only ex? ignore
            continue
        ref = ref.strip().replace("\\", "/")
        if ref not in ref_to_exset:
            ref_to_exset[ref] = set()
        if ex:
            has_ex = True
            ref_to_exset[ref].add(ex.strip().replace("\\", "/"))
    return Labels(ref_to_exset, has_ex)

def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + (z**2)/total
    center = (phat + (z**2)/(2*total)) / denom
    margin = z * math.sqrt((phat*(1-phat) + (z**2)/(4*total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ============== Evaluation core ==============

def evaluate(mapping_obj: dict, labels: Labels) -> dict:
    mapping: Dict[str, str] = mapping_obj.get("mapping", {})
    assignments = mapping_obj.get("assignments", [])
    assigned_refs = set(mapping.keys())
    all_refs: Set[str] = set()
    # derive all refs from mapping, unassigned too if present
    if "unassigned" in mapping_obj:
        all_refs |= set(mapping_obj["unassigned"])
    all_refs |= assigned_refs

    labeled_refs = set(labels.ref_to_exset.keys())
    L = len(labeled_refs)
    correct_on_labeled = 0
    for ref in labeled_refs:
        ex_mapped = mapping.get(ref)
        if ex_mapped is None:
            continue
        exset = labels.ref_to_exset.get(ref, set())
        if (not labels.has_ex_column) or (ex_mapped in exset):
            correct_on_labeled += 1

    # Recall on labeled positives only
    recall_L = (correct_on_labeled / L) if L > 0 else None
    # Precision lower bound: assume unlabeled are incorrect (pessimistic)
    total_assigned = len(assigned_refs)
    precision_LB = (correct_on_labeled / total_assigned) if total_assigned > 0 and L > 0 else None

    # suspects as reported (from 04) if present
    suspects = mapping_obj.get("suspects", [])
    stats = {
        "references": len(all_refs),
        "assigned": total_assigned,
        "unassigned": int(max(0, len(all_refs) - total_assigned)),
        "suspects_reported": len(suspects),
        "labels_total": L,
        "correct_on_labeled": int(correct_on_labeled),
        "recall_L": recall_L,
        "precision_LB": precision_LB,
        "assign_mode": mapping_obj.get("assign_mode", "greedy"),
    }
    return stats

# ============== Suggestions (Top-N) ==============

def build_suggestions(scores_rows: List[dict],
                      mapping: Dict[str, str],
                      N: int,
                      only_unmatched: bool) -> List[Dict[str, str]]:
    # Group by ref
    per_ref: Dict[str, List[dict]] = {}
    for row in scores_rows:
        ref = (row.get("ref") or "").strip().replace("\\", "/")
        if not ref:
            continue
        per_ref.setdefault(ref, []).append(row)

    out: List[Dict[str, str]] = []
    for ref, lst in per_ref.items():
        if only_unmatched and (ref in mapping):
            continue
        # sort by numeric score desc, then inliers desc
        def to_f(x, k, default=0.0):
            try:
                return float(x.get(k, default))
            except Exception:
                return default
        lst.sort(key=lambda r: (to_f(r, "score"), to_f(r, "inliers")), reverse=True)
        rank = 0
        for r in lst:
            ex = (r.get("ex") or "").strip().replace("\\", "/")
            if ex == "" or (ref in mapping and ex == mapping.get(ref)):
                continue
            rank += 1
            out.append({
                "ref": ref,
                "rank": str(rank),
                "ex": ex,
                "score": f"{to_f(r, 'score'):.6f}",
                "inliers": str(int(to_f(r, "inliers"))),
                "inlier_ratio": f"{to_f(r, 'inlier_ratio'):.6f}",
                "phase": r.get("phase") or "",
                "ssim": f"{to_f(r, 'ssim'):.6f}",
                "ncc": f"{to_f(r, 'ncc'):.6f}",
                "top2_margin": f"{to_f(r, 'top2_margin'):.6f}",
                "top2_multiplier": f"{to_f(r, 'top2_multiplier'):.6f}",
            })
            if rank >= N:
                break
    return out

def write_suggestions_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    cols = ["ref","rank","ex","score","inliers","inlier_ratio","phase","ssim","ncc","top2_margin","top2_multiplier"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ============== Inspect pack builder ==============

def choose_variant(vp: VariantPaths, copy_variant: str) -> Optional[Path]:
    # Mapping:
    #   pre_low_color  -> vp.low_color
    #   pre_low_gray   -> vp.low_gray
    #   pre_high_color -> vp.high_color
    #   pre_high_gray  -> vp.high_gray
    #   orig           -> vp.src_abspath
    if copy_variant == "pre_low_color":
        return vp.low_color or vp.low_gray or vp.color_low  # compat
    if copy_variant == "pre_low_gray":
        return vp.low_gray or vp.low_color
    if copy_variant == "pre_high_color":
        return vp.high_color or vp.low_color
    if copy_variant == "pre_high_gray":
        return vp.high_gray or vp.low_gray
    if copy_variant == "orig":
        return vp.src_abspath or vp.low_color or vp.low_gray
    return vp.low_color or vp.low_gray

def build_inspect_pack(inspect_out: Path,
                       mapping: Dict[str, str],
                       ex_map: Dict[str, VariantPaths],
                       rf_map: Dict[str, VariantPaths],
                       copy_variant: str,
                       allow_basename_fallback: bool,
                       roots: dict) -> Tuple[int, int]:
    """
    Copies (ref, ex) images into inspect_out/{ref,ex}/NNNN.* and writes manifest.csv.
    Returns: (copied_pairs, missing_pairs)
    """
    inspect_out.mkdir(parents=True, exist_ok=True)
    ref_dir = inspect_out / "ref"
    ex_dir  = inspect_out / "ex"
    ref_dir.mkdir(exist_ok=True)
    ex_dir.mkdir(exist_ok=True)

    # Manifest rows
    mrows: List[Dict[str, str]] = []
    copied = 0
    missing = 0

    # path fallback helper
    def fallback_path(is_ref: bool, rel: str) -> Optional[Path]:
        if not allow_basename_fallback:
            return None
        # Try composing from roots
        base = "reference_in" if is_ref else "extracted_in"
        root = Path(roots.get(base, "")) if roots else None
        if root and (root.exists()):
            p = (root / rel)
            return p if p.exists() else None
        return None

    # deterministic order
    refs_sorted = sorted(mapping.keys())
    for idx, ref_rel in enumerate(refs_sorted, 1):
        ex_rel = mapping.get(ref_rel)
        r_vp = rf_map.get(ref_rel)
        x_vp = ex_map.get(ex_rel, None) if ex_rel else None

        r_src = choose_variant(r_vp, copy_variant) if r_vp else None
        x_src = choose_variant(x_vp, copy_variant) if x_vp else None

        if (not r_src or not r_src.exists()) and allow_basename_fallback:
            r_src = r_src if (r_src and r_src.exists()) else fallback_path(True, ref_rel)
        if (not x_src or not x_src.exists()) and allow_basename_fallback:
            x_src = x_src if (x_src and x_src.exists()) else fallback_path(False, ex_rel)

        if not r_src or not x_src or (not r_src.exists()) or (not x_src.exists()):
            missing += 1
            mrows.append({
                "idx": f"{idx:04d}", "ref_rel": ref_rel, "ex_rel": ex_rel or "",
                "ref_path": str(r_src) if r_src else "", "ex_path": str(x_src) if x_src else "",
                "copied": "0"
            })
            continue

        # destination names
        ext_r = r_src.suffix.lower() or ".png"
        ext_x = x_src.suffix.lower() or ".png"
        ref_dst = ref_dir / f"{idx:04d}{ext_r}"
        ex_dst  = ex_dir  / f"{idx:04d}{ext_x}"

        ok_r = safe_copy(r_src, ref_dst)
        ok_x = safe_copy(x_src, ex_dst)
        if ok_r and ok_x:
            copied += 1
            mrows.append({
                "idx": f"{idx:04d}", "ref_rel": ref_rel, "ex_rel": ex_rel or "",
                "ref_path": str(ref_dst), "ex_path": str(ex_dst),
                "copied": "1"
            })
        else:
            missing += 1
            mrows.append({
                "idx": f"{idx:04d}", "ref_rel": ref_rel, "ex_rel": ex_rel or "",
                "ref_path": str(ref_dst), "ex_path": str(ex_dst),
                "copied": "0"
            })

    # write manifest
    mf = inspect_out / "manifest.csv"
    with open(mf, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx","ref_rel","ex_rel","ref_path","ex_path","copied"])
        w.writeheader()
        for r in mrows:
            w.writerow(r)

    return copied, missing

# ============== Feedback builder ==============

@dataclass
class FeedbackThresholds:
    suspect_ssim: float = 0.90
    suspect_ncc: float = 0.85
    suspect_top2_margin: float = 6.0
    suspect_top2_multiplier: float = 1.40

def build_feedback(mapping_obj: dict,
                   scores_rows: List[dict],
                   eval_stats: dict,
                   thresholds: FeedbackThresholds,
                   suggest_n: int) -> dict:
    """
    Heuristics:
      - redo_preprocess.refs:
          * refs with NO scores at all (candidate starvation)
          * OR assigned refs with very low photometric (ssim<ncc<th) -> likely contrast/trim issue
      - expand_candidates.refs:
          * unmatched refs with some scores but none passed -> search wider
          * refs whose best score is much higher than next, but still unassigned (rare)
      - tighten_verification.refs:
          * assigned refs flagged as suspect by 04 OR photo/top2 near boundary
      - allow_reuse.enabled:
          * if the same 'ex' appears frequently in top suggestions across many refs
    """
    mapping: Dict[str, str] = mapping_obj.get("mapping", {})
    assignments = mapping_obj.get("assignments", [])
    assigned_refs = set(mapping.keys())
    unassigned = set(mapping_obj.get("unassigned", []))
    suspects_from_04 = set(mapping_obj.get("suspects", []))

    # Group scores by ref
    by_ref: Dict[str, List[dict]] = {}
    for row in scores_rows:
        ref = (row.get("ref") or "").strip().replace("\\", "/")
        if not ref:
            continue
        by_ref.setdefault(ref, []).append(row)

    redo_refs: Set[str] = set()
    expand_refs: Set[str] = set()
    tighten_refs: Set[str] = set()
    # frequency of 'ex' in top-K ranks (K=3)
    ex_freq: Dict[str, int] = {}

    # numeric converters
    def f(row, k, d=0.0):
        try:
            return float(row.get(k, d))
        except Exception:
            return d

    # analyze per ref
    for ref, lst in by_ref.items():
        lst.sort(key=lambda r: (f(r,"score"), f(r,"inliers")), reverse=True)
        # collect ex frequency from top-3
        for r in lst[:3]:
            ex = (r.get("ex") or "").strip().replace("\\", "/")
            if ex:
                ex_freq[ex] = ex_freq.get(ex, 0) + 1

        if ref not in assigned_refs:
            # unmatched
            if len(lst) == 0:
                redo_refs.add(ref)         # no candidate scored -> likely preprocess issue
            else:
                expand_refs.add(ref)       # had candidates but none accepted -> widen search
            continue

        # assigned: check borderline
        a = None
        for r in lst:
            # find the accepted one: phase A/B or equal to mapping ex
            if (r.get("phase") in ("A","B")) and ((r.get("ex") or "") == mapping.get(ref)):
                a = r
                break
        # if not found, locate the row corresponding to mapped ex
        if a is None:
            for r in lst:
                if (r.get("ex") or "") == mapping.get(ref):
                    a = r
                    break

        if a is None:
            # mapped but not found in scores? unusual -> request tighten for safety
            tighten_refs.add(ref)
            continue

        ssim = f(a, "ssim")
        ncc  = f(a, "ncc")
        t2m  = f(a, "top2_margin")
        t2x  = f(a, "top2_multiplier")
        if (ssim < thresholds.suspect_ssim and ncc < thresholds.suspect_ncc) or (t2m < thresholds.suspect_top2_margin) or (t2x < thresholds.suspect_top2_multiplier):
            tighten_refs.add(ref)

    # merge suspects from 04
    tighten_refs |= suspects_from_04

    # build allow_reuse hint
    # if many refs tend to choose the same ex in top-3 suggestions -> duplication pattern
    reuse_enabled = any(cnt >= 3 for cnt in ex_freq.values())

    # global expansion defaults (conservative)
    expand_defaults = {
        "phash_radius_delta": 4,
        "hist_threshold_new": 0.80,
        "min_cand_per_basis": 12
    }

    feedback = {
        "version": "1.0",
        "summary": {
            "assigned": eval_stats.get("assigned", 0),
            "references": eval_stats.get("references", 0),
            "unassigned": eval_stats.get("unassigned", 0),
            "suspects": int(len(tighten_refs)),
            "labels_total": eval_stats.get("labels_total", 0),
            "recall_L": eval_stats.get("recall_L", None),
            "precision_LB": eval_stats.get("precision_LB", None)
        },
        "actions": {
            "redo_preprocess": {
                "refs": sorted(list(redo_refs)),
                # reserved for extracted-side targeting if needed in future
                "extracted_refs": []
            },
            "expand_candidates": {
                "refs": sorted(list(expand_refs)),
                **expand_defaults,
                # per-ref overrides can be filled later if needed
                "per_ref": []
            },
            "tighten_verification": {
                "refs": sorted(list(tighten_refs)),
                "top2_margin": thresholds.suspect_top2_margin,
                "top2_multiplier": thresholds.suspect_top2_multiplier,
                "ssim_th": thresholds.suspect_ssim,
                "ncc_th": thresholds.suspect_ncc
            },
            "allow_reuse": {
                "enabled": bool(reuse_enabled)
            }
        }
    }
    return feedback

# ============== CLI / Main ==============

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate mapping, build inspect pack, suggestions, and emit feedback for the next loop.")
    ap.add_argument("--mapping-json", required=True, help="mapping_result.json from 04")
    ap.add_argument("--scores-csv", default=None, help="pair_scores.csv (falls back to mapping_result.scores_csv)")
    ap.add_argument("--mapping-meta", required=True, help="preprocess_mapping.json from 02")
    ap.add_argument("--labels-csv", default=None, help="Positive-only labels CSV (columns: ref,ex)")

    ap.add_argument("--sample-verify", default=None, help="Optional CSV with sampled verification results (columns: ref,is_correct[0/1])")

    ap.add_argument("--inspect-out", default="inspect_pairs", help="Output directory for inspection pack")
    ap.add_argument("--copy-variant", choices=[
        "pre_low_color", "pre_low_gray", "pre_high_color", "pre_high_gray", "orig"
    ], default="pre_low_color", help="Which variant to copy for inspection")
    ap.add_argument("--allow-basename-fallback", action="store_true", help="If chosen variant missing, try roots + relpath")

    ap.add_argument("--suggest-n", type=int, default=20, help="Top-N suggestions per ref")
    ap.add_argument("--suggest-only-unmatched", action="store_true", help="Emit suggestions only for unmatched refs")

    ap.add_argument("--export-report", default="evaluation_report.json", help="Where to write evaluation_report.json")
    ap.add_argument("--export-feedback", default=None, help="Where to write feedback.json (optional)")

    # thresholds to flag suspects and drive feedback
    ap.add_argument("--suspect-ssim", type=float, default=0.90)
    ap.add_argument("--suspect-ncc", type=float, default=0.85)
    ap.add_argument("--suspect-top2-margin", type=float, default=6.0)
    ap.add_argument("--suspect-top2-multiplier", type=float, default=1.40)

    return ap.parse_args()

def main() -> int:
    args = parse_args()

    mapping_obj = read_json(Path(args.mapping_json))
    scores_path = Path(args.scores_csv) if args.scores_csv else Path(mapping_obj.get("scores_csv", ""))
    if not scores_path or not scores_path.exists():
        print(f"[WARN] pair_scores.csv not found at {scores_path}. Suggestions & some feedback signals may be limited.")
    scores_rows = read_csv_rows(scores_path) if scores_path and scores_path.exists() else []

    ex_map, rf_map, mapping_meta = load_preprocess_mapping(Path(args.mapping_meta))
    roots = mapping_meta.get("roots", {})

    labels = load_labels(Path(args.labels_csv) if args.labels_csv else None)
    eval_stats = evaluate(mapping_obj, labels)

    # Optional: sample verification Wilson CI
    if args.sample_verify:
        sv_rows = read_csv_rows(Path(args.sample_verify))
        succ = sum(1 for r in sv_rows if (r.get("is_correct") or r.get("correct") or "0") in ("1", "true", "True"))
        tot = len(sv_rows)
        lo, hi = wilson_ci(succ, tot) if tot > 0 else (0.0, 0.0)
        eval_stats["sample_precision_estimate"] = (succ / tot) if tot > 0 else None
        eval_stats["sample_precision_wilson95"] = {"low": lo, "high": hi, "n": tot}

    # Build inspect pack
    mapping: Dict[str, str] = mapping_obj.get("mapping", {})
    copied, missing = build_inspect_pack(
        inspect_out=Path(args.inspect_out),
        mapping=mapping,
        ex_map=ex_map, rf_map=rf_map,
        copy_variant=args.copy_variant,
        allow_basename_fallback=args.allow_basename_fallback,
        roots=roots
    )
    eval_stats["inspect_copied_pairs"] = int(copied)
    eval_stats["inspect_missing_pairs"] = int(missing)

    # Suggestions Top-N
    sugg_rows = build_suggestions(
        scores_rows=scores_rows,
        mapping=mapping,
        N=max(1, int(args.suggest_n)),
        only_unmatched=bool(args.suggest_only_unmatched)
    )
    sugg_path = Path("suggestions_topN.csv").resolve()
    write_suggestions_csv(sugg_path, sugg_rows)
    eval_stats["suggestions_written"] = str(sugg_path) if sugg_rows else ""

    # Export evaluation report
    report_obj = {
        "stats": eval_stats,
        "inputs": {
            "mapping_json": str(Path(args.mapping_json).resolve()),
            "scores_csv": str(scores_path.resolve()) if scores_rows else "",
            "mapping_meta": str(Path(args.mapping_meta).resolve()),
            "labels_csv": str(Path(args.labels_csv).resolve()) if args.labels_csv else "",
        },
        "inspect_out": str(Path(args.inspect_out).resolve()),
        "suggestions_csv": str(sugg_path) if sugg_rows else ""
    }
    write_json(Path(args.export_report), report_obj)
    print(f"[INFO] evaluation_report saved: {args.export_report}")

    # Build & export feedback (optional)
    if args.export_feedback:
        th = FeedbackThresholds(
            suspect_ssim=float(args.suspect_ssim),
            suspect_ncc=float(args.suspect_ncc),
            suspect_top2_margin=float(args.suspect_top2_margin),
            suspect_top2_multiplier=float(args.suspect_top2_multiplier),
        )
        feedback = build_feedback(
            mapping_obj=mapping_obj,
            scores_rows=scores_rows,
            eval_stats=eval_stats,
            thresholds=th,
            suggest_n=int(args.suggest_n)
        )
        write_json(Path(args.export_feedback), feedback)
        print(f"[INFO] feedback saved: {args.export_feedback}")

    print("[INFO] Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
