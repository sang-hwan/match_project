# 5_check_map_result.py
"""
매핑 결과 페어 수집/시각 점검용 스크립트 (견고화 버전)

사용 예:
python .\5_check_map_result.py ^
  ".\map_dist\mapping_result.json" ^
  ".\images_output" ^
  ".\target_data\자동등록 사진 모음" ^
  ".\inspect_pairs" ^
  --export-csv .\inspect_pairs\manifest.csv

동작:
- mapping_result.json에서 (original_path, extracted_path) 페어를 읽어
  out_dir/<pair_####>/에 각 파일을 복사합니다.
- --export-csv가 주어지면 매니페스트 CSV를 저장합니다.

안전장치:
- JSON 최상위에 'pairs'(list)와 'map'(dict)이 함께 있어도 안전 처리
- 경로가 절대경로로 존재하지 않으면, 각 root와 basename으로 재해결 시도
- 출력 폴더/CSV의 부모 폴더 자동 생성
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect mapped (orig, extracted) image pairs for visual inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("mapping_json", help="mapping_result.json")
    ap.add_argument("extracted_root", help="root folder of extracted images (e.g., images_output)")
    ap.add_argument("reference_root", help="root folder of reference originals (e.g., target_data/자동등록 사진 모음)")
    ap.add_argument("out_dir", help="output directory to copy pairs into")
    ap.add_argument("--export-csv", default=None, help="path to export manifest CSV")
    return ap.parse_args()


# ──────────────────────────────── JSON 로딩/정규화 ────────────────────────────────
def load_pairs_from_mapping(mapping_path: Path) -> List[Dict]:
    """
    mapping_result.json을 읽어 (orig, extracted) 페어 리스트를 반환.
    - 우선 'pairs' 리스트를 사용(여기에 score 등 메타 포함)
    - 없으면 'map' dict을 변환
    - 둘 다 없으면, 최상위가 dict[str,str]이면 그대로 변환
    반환: [{orig, extracted, score, inliers, kpA, kpB, track, ex_used, or_used}, ...]
    """
    js = json.loads(mapping_path.read_text("utf-8"))

    rows: List[Dict] = []

    if isinstance(js, dict) and isinstance(js.get("pairs"), list):
        for it in js["pairs"]:
            if not isinstance(it, dict):
                continue
            o = it.get("original_path")
            e = it.get("extracted_path")
            if isinstance(o, str) and isinstance(e, str):
                rows.append({
                    "orig": o,
                    "extracted": e,
                    "score": it.get("score"),
                    "inliers": it.get("inliers"),
                    "kpA": it.get("kpA"),
                    "kpB": it.get("kpB"),
                    "track": it.get("track"),
                    "ex_used": it.get("ex_used"),
                    "or_used": it.get("or_used"),
                })
        if rows:
            return rows

    if isinstance(js, dict) and isinstance(js.get("map"), dict):
        for o, e in js["map"].items():
            if isinstance(o, str) and isinstance(e, str):
                rows.append({"orig": o, "extracted": e})
        if rows:
            return rows

    if isinstance(js, dict):  # 구버전: 최상위가 바로 {orig: extracted}
        ok = True
        for k, v in js.items():
            if not (isinstance(k, str) and isinstance(v, str)):
                ok = False
                break
        if ok:
            for o, e in js.items():
                rows.append({"orig": o, "extracted": e})
            return rows

    raise ValueError("mapping_result.json에서 유효한 페어 정보를 찾지 못했습니다. ('pairs' 또는 'map' 필요)")


# ──────────────────────────────── 경로 해석/복사 ────────────────────────────────
def resolve_existing_path(raw: str, root: Path) -> Optional[Path]:
    """
    우선 raw 경로가 존재하면 그대로 사용.
    아니면 root / basename(raw)를 시도.
    그래도 없으면 None.
    """
    if isinstance(raw, str):
        p = Path(raw)
        if p.exists():
            return p
        alt = root / Path(raw).name
        if alt.exists():
            return alt
    return None


def copy_if_exists(src: Optional[Path], dst: Path) -> Tuple[Optional[Path], str]:
    """
    src가 있으면 dst로 복사(부모 mkdir), 상태 문자열 반환('copied' 또는 'missing').
    """
    if src is None or not src.exists():
        return None, "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(dst))
        return dst, "copied"
    except Exception:
        return None, "error"


# ──────────────────────────────── 메인 ────────────────────────────────
def main():
    args = parse_args()
    mapping_path = Path(args.mapping_json)
    ex_root = Path(args.extracted_root)
    or_root = Path(args.reference_root)
    out_dir = Path(args.out_dir)

    pairs = load_pairs_from_mapping(mapping_path)

    manifest_rows: List[Dict] = []
    copied_ok = 0

    for idx, it in enumerate(pairs, start=1):
        o_raw = it.get("orig")
        e_raw = it.get("extracted")
        if not (isinstance(o_raw, str) and isinstance(e_raw, str)):
            continue

        # 경로 해석(존재 우선, 실패시 root + basename 대체)
        o_src = resolve_existing_path(o_raw, or_root)
        e_src = resolve_existing_path(e_raw, ex_root)

        # 출력 경로(페어별 하위 폴더)
        pair_dir = out_dir / f"pair_{idx:04d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        o_dst = pair_dir / f"or_{Path(o_raw).name}"
        e_dst = pair_dir / f"ex_{Path(e_raw).name}"

        e_out, e_status = copy_if_exists(e_src, e_dst)
        o_out, o_status = copy_if_exists(o_src, o_dst)

        if e_status == "copied" and o_status == "copied":
            copied_ok += 1

        manifest_rows.append({
            "pair_id": idx,
            "orig": o_raw,
            "extracted": e_raw,
            "orig_exists": bool(o_src and o_src.exists()),
            "extracted_exists": bool(e_src and e_src.exists()),
            "orig_copied": (o_status == "copied"),
            "extracted_copied": (e_status == "copied"),
            "orig_dst": str(o_out) if o_out else None,
            "extracted_dst": str(e_out) if e_out else None,
            # 메타가 있으면 포함
            "score": it.get("score"),
            "inliers": it.get("inliers"),
            "kpA": it.get("kpA"),
            "kpB": it.get("kpB"),
            "track": it.get("track"),
            "ex_used": it.get("ex_used"),
            "or_used": it.get("or_used"),
        })

    # 매니페스트 저장
    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows).to_csv(export_path, index=False, encoding="utf-8-sig")

    # 콘솔 요약
    total = len(pairs)
    print("===== CHECK RESULT =====")
    print(f"Total pairs in mapping: {total}")
    print(f"Copied OK (both sides): {copied_ok}")
    missing_e = sum(1 for r in manifest_rows if not r["extracted_exists"])
    missing_o = sum(1 for r in manifest_rows if not r["orig_exists"])
    print(f"Missing extracted files: {missing_e}")
    print(f"Missing original files : {missing_o}")
    if args.export_csv:
        print(f"[SAVE] manifest → {args.export_csv}")
    print("[DONE] copy complete.")


if __name__ == "__main__":
    main()
