# 4_A_verify_data.py
#
# This script verifies the **one-to-one integrity** between the candidate set
# (candidates.json) and the preprocessing mapping (preprocess_mapping.json).
#
# ──────────────────────────────────────────────────────────────
# 1) Check that every target/candidate image path exists in the mapping
# 2) For each original image, confirm that all four combinations
#    (low/color, low/gray, high/color, high/gray) are present
# 3) Print statistics and lists of missing items (up to 20 samples) to stdout
#    → exit code 1 if inconsistencies are found, 0 otherwise
#
# Usage example
#   $ python 4_A_verify_data.py \
#       --candidates candidates.json \
#       --mapping preprocess_mapping.json
#
# ※ Uses plain print() statements instead of any logging library.
# ──────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

EXPECTED_COMBOS: Set[Tuple[str, str]] = {
    ("low", "color"),
    ("low", "gray"),
    ("high", "color"),
    ("high", "gray"),
}


def load_json(path: str) -> Dict:
    """Load a UTF-8 JSON file and verify that it exists."""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found → {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def norm_path(p: str) -> str:
    """Normalize paths for comparison (handles case/slash differences on Windows)."""
    return os.path.normpath(p).lower()


def build_origin_index(mapping: Dict) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build an index: original image path → list of (track, channel) tuples.

    Returns
    -------
    Dict[str, List[(track, channel)]]
    """
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for preprocess_name, meta in mapping.items():
        origin = norm_path(meta["원본_전체_경로"])
        index[origin].append((meta["트랙"], meta["채널"]))
    return index


def verify_candidates(
    candidates: Dict, origin_index: Dict[str, List[Tuple[str, str]]]
) -> Tuple[List[str], List[str]]:
    """Check that target and candidate paths in candidates.json exist in the mapping."""
    missing_targets: List[str] = []
    missing_candidates: List[str] = []

    for target_path, info in candidates.items():
        n_target = norm_path(target_path)
        if n_target not in origin_index:
            missing_targets.append(target_path)

        for cand in info.get("candidates", []):
            n_cand = norm_path(cand["name"])
            if n_cand not in origin_index:
                missing_candidates.append(cand["name"])

    return missing_targets, missing_candidates


def verify_combo_completeness(
    origin_index: Dict[str, List[Tuple[str, str]]]
) -> List[str]:
    """Return a list of original paths lacking the full low/high × color/gray set."""
    incomplete: List[str] = []
    for origin, combos in origin_index.items():
        if not EXPECTED_COMBOS.issubset(set(combos)):
            incomplete.append(origin)
    return incomplete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify consistency between candidates.json "
        "and preprocess_mapping.json"
    )
    parser.add_argument(
        "-c",
        "--candidates",
        default="candidates.json",
        help="Path to candidates.json (default: ./candidates.json)",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        default="preprocess_mapping.json",
        help="Path to preprocess_mapping.json (default: ./preprocess_mapping.json)",
    )
    args = parser.parse_args()

    print("▶ Loading JSON files …")
    candidates = load_json(args.candidates)
    mapping = load_json(args.mapping)

    print("▶ Building lookup index …")
    origin_index = build_origin_index(mapping)

    print("▶ Step 1. Path existence check")
    miss_t, miss_c = verify_candidates(candidates, origin_index)

    print("▶ Step 2. Track/Channel combo completeness")
    incomplete_combo = verify_combo_completeness(origin_index)

    # ────────────────── Summary ──────────────────
    total_targets = len(candidates)
    total_candidate_refs = sum(len(v["candidates"]) for v in candidates.values())

    print("\n===== SUMMARY =====")
    print(f"Targets in candidates.json          : {total_targets:,}")
    print(f"Candidate references                : {total_candidate_refs:,}")
    print(f"Unique originals in mapping         : {len(origin_index):,}")
    print(f"Missing target paths                : {len(miss_t):,}")
    print(f"Missing candidate paths             : {len(set(miss_c)):,}")
    print(f"Origins w/ incomplete 4-combo set   : {len(incomplete_combo):,}")
    print("====================\n")

    # Detailed samples of missing items (up to 20 each)
    def _print_sample(title: str, data: List[str]) -> None:
        if not data:
            return
        print(f"-- {title} (showing ≤ 20) --")
        for p in data[:20]:
            print("  ", p)
        print()

    _print_sample("Missing target paths", miss_t)
    _print_sample("Missing candidate paths", list(dict.fromkeys(miss_c)))
    _print_sample("Origins lacking full 4-combo", incomplete_combo)

    # Determine exit code
    if not (miss_t or miss_c or incomplete_combo):
        print("All checks passed – mapping is consistent.")
        sys.exit(0)
    else:
        print("Inconsistencies detected – please review the above lists.")
        sys.exit(1)


if __name__ == "__main__":
    main()
