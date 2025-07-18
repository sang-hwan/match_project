# 3_mk_phash_candidates.py
"""
pHash 후보 추림: 전처리된 추출 이미지와 원본 이미지 중 더 적은 쪽을 기준으로 매핑 후보를 찾아 하나의 JSON 파일로 저장합니다.

Usage
-----
python 3_mk_phash_candidates.py \
  path/to/extracted_dir \
  path/to/originals_dir \
  out.json \
  [--threshold 10]
"""

import argparse
import json
from pathlib import Path
from PIL import Image
import imagehash

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def compute_phash(path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash for the given image file."""
    with Image.open(path) as img:
        return imagehash.phash(img)


def main():
    pa = argparse.ArgumentParser(
        description="pHash 후보 추림: 전처리된 추출 이미지와 원본 이미지 중 더 적은 쪽을 기준으로 매핑 후보 추출"
    )
    pa.add_argument(
        "extracted_dir",
        help="전처리된 추출 이미지 디렉토리 경로"
    )
    pa.add_argument(
        "originals_dir",
        help="전처리된 원본 이미지 디렉토리 경로"
    )
    pa.add_argument(
        "out_json",
        help="출력 JSON 파일 경로"
    )
    pa.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="pHash 해밍 거리 임계값 (기본값: 10)"
    )
    args = pa.parse_args()

    print("[START] pHash 후보 추림 시작")
    extracted_dir = Path(args.extracted_dir)
    originals_dir = Path(args.originals_dir)
    print(f"[INFO] 탐색 경로 - 추출 이미지: {extracted_dir}, 원본 이미지: {originals_dir}")

    ext_files = [p for p in extracted_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    ori_files = [p for p in originals_dir.rglob('*') if p.suffix.lower() in SUPPORTED_EXTS]
    print(f"[INFO] 추출 이미지 수: {len(ext_files)}, 원본 이미지 수: {len(ori_files)}")

    print("[STEP] 추출 이미지 pHash 계산 중...")
    ext_hashes = {p: compute_phash(p) for p in ext_files}
    print(f"[DONE] 추출 이미지 pHash 계산 완료 ({len(ext_hashes)})")
    print("[STEP] 원본 이미지 pHash 계산 중...")
    ori_hashes = {p: compute_phash(p) for p in ori_files}
    print(f"[DONE] 원본 이미지 pHash 계산 완료 ({len(ori_hashes)})")

    if len(ori_hashes) <= len(ext_hashes):
        basis_hashes, compare_hashes = ori_hashes, ext_hashes
        basis_root, compare_root = originals_dir, extracted_dir
        basis_label, compare_label = "original", "extracted"
    else:
        basis_hashes, compare_hashes = ext_hashes, ori_hashes
        basis_root, compare_root = extracted_dir, originals_dir
        basis_label, compare_label = "extracted", "original"
    print(f"[INFO] 기준 세트: {basis_label} ({len(basis_hashes)}) vs 비교 세트: {compare_label} ({len(compare_hashes)})")

    mapping: dict[str, list[str]] = {}
    print("[START] 후보 매핑 생성")
    total = len(basis_hashes)
    for idx, (basis_path, basis_hash) in enumerate(basis_hashes.items(), start=1):
        rel_basis = basis_path.relative_to(basis_root).as_posix()
        print(f"[{idx}/{total}] 처리 중: {rel_basis}")

        # 거리 분포 출력
        dists = [
            (cmp_path.relative_to(compare_root).as_posix(), basis_hash - cmp_hash)
            for cmp_path, cmp_hash in compare_hashes.items()
        ]
        dists.sort(key=lambda x: x[1])
        print(f"  ▶ Top-5 해밍 거리: {dists[:5]}")

        candidates = [o for o, d in dists if d <= args.threshold]
        print(f"  ▶ 임계값({args.threshold}) 이하 후보 수: {len(candidates)}")

        mapping[rel_basis] = candidates
    print(f"[DONE] 후보 매핑 생성 완료 (총 항목: {len(mapping)})")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 매핑 후보 JSON 저장 완료: {out_path}")

if __name__ == '__main__':
    main()
