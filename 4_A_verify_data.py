# 4_A_verify_data.py
"""
Verify one-to-one integrity between the set of candidate mappings (candidates.json)
and the preprocessing mapping (preprocess_mapping.json).

This script performs the following steps:
  1) Load JSON files:
     - candidates.json: mapping of each original image to its candidate list.
     - preprocess_mapping.json: metadata mapping of original images to preprocessing outputs.
  2) Build an index of original image paths to their preprocessing variant tuples (track, channel).
  3) Check that every target and candidate path in candidates.json exists in the mapping.
  4) Ensure each original has all four expected preprocessing combinations:
       (low,color), (low,gray), (high,color), (high,gray).
  5) Print summary statistics and up to 20 samples of missing items.
  6) Exit with status code 0 if all checks pass, or 1 if inconsistencies are found.

Usage:
  python 4_A_verify_data.py --candidates candidates.json --mapping preprocess_mapping.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

EXPECTED_COMBOS: Set[Tuple[str, str]] = {
    ("low", "color"),
    ("low", "gray"),
    ("high", "color"),
    ("high", "gray"),
}


def load_json(path: str) -> Dict:
    """Load a JSON file or exit with error if not found."""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def norm_path(p: str) -> str:
    """Normalize file paths for case-insensitive comparison."""
    return os.path.normpath(p).lower()


def build_origin_index(mapping: Dict) -> Dict[str, List[Tuple[str, str]]]:
    """Build index mapping each original path to a list of (track, channel)."""
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for meta in mapping.values():
        origin = norm_path(meta.get("원본_전체_경로", ""))
        index[origin].append((meta.get("트랙", ""), meta.get("채널", "")))
    return index


def verify_candidates(
    candidates: Dict, origin_index: Dict[str, List[Tuple[str, str]]]
) -> Tuple[List[str], List[str]]:
    """Return lists of missing targets and missing candidate references."""
    missing_targets: List[str] = []
    missing_candidates: List[str] = []

    for target_path, info in candidates.items():
        if norm_path(target_path) not in origin_index:
            missing_targets.append(target_path)
        for cand in info.get("candidates", []):
            if norm_path(cand.get("name", "")) not in origin_index:
                missing_candidates.append(cand.get("name", ""))
    return missing_targets, missing_candidates


def verify_combo_completeness(
    origin_index: Dict[str, List[Tuple[str, str]]]
) -> List[str]:
    """Return list of origins missing any expected preprocessing combos."""
    incomplete: List[str] = []
    for origin, combos in origin_index.items():
        if not EXPECTED_COMBOS.issubset(set(combos)):
            incomplete.append(origin)
    return incomplete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify consistency between candidates and preprocess mapping"
    )
    parser.add_argument(
        "-c", "--candidates",
        default="candidates.json",
        help="Path to candidates.json"
    )
    parser.add_argument(
        "-m", "--mapping",
        default="preprocess_mapping.json",
        help="Path to preprocess_mapping.json"
    )
    args = parser.parse_args()

    print("▶ Loading JSON files …")
    candidates = load_json(args.candidates)
    mapping = load_json(args.mapping)

    print("▶ Building origin index …")
    origin_index = build_origin_index(mapping)

    print("▶ Step 1. Path existence check")
    missing_targets, missing_candidates = verify_candidates(candidates, origin_index)

    print("▶ Step 2. Combo completeness check")
    incomplete_combos = verify_combo_completeness(origin_index)

    total_targets = len(candidates)
    total_candidate_refs = sum(len(v.get("candidates", [])) for v in candidates.values())

    print("\n===== SUMMARY =====")
    print(f"Targets           : {total_targets:,}")
    print(f"Candidate refs    : {total_candidate_refs:,}")
    print(f"Unique originals  : {len(origin_index):,}")
    print(f"Missing targets   : {len(missing_targets):,}")
    print(f"Missing candidates: {len(set(missing_candidates)):,}")
    print(f"Incomplete combos : {len(incomplete_combos):,}")
    print("====================\n")

    def print_samples(title: str, data: List[str]) -> None:
        if not data:
            return
        print(f"-- {title} (up to 20) --")
        for p in data[:20]:
            print("  ", p)
        print()

    print_samples("Missing targets", missing_targets)
    print_samples("Missing candidates", list(dict.fromkeys(missing_candidates)))
    print_samples("Incomplete combos", incomplete_combos)

    if not (missing_targets or missing_candidates or incomplete_combos):
        print("All checks passed – mapping is consistent.")
        sys.exit(0)
    else:
        print("Inconsistencies detected – please review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
