# test_valid.py
"""
매핑 결과를 기반으로 이미지 쌍 복사‧정렬
"""
import argparse
import shutil
from pathlib import Path

from utils import Tee


def main() -> None:
    pa = argparse.ArgumentParser(description="매핑 결과 이미지 복사/정렬")
    pa.add_argument("mapping_file", help="image_mapping.txt")
    pa.add_argument("extracted_dir", help="추출 이미지 폴더")
    pa.add_argument("output_dir", help="복사 대상 폴더")
    args = pa.parse_args()

    log_dir = Path(__file__).parent / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tee(log_dir / "test_valid_log.txt")

    mapping_file = Path(args.mapping_file)
    extracted_root = Path(args.extracted_dir)
    out_dir = Path(args.output_dir)

    if not mapping_file.exists():
        print(f"[ERR] 매핑 파일이 없습니다: {mapping_file}")
        return
    if not extracted_root.exists():
        print(f"[ERR] 추출 폴더가 없습니다: {extracted_root}")
        return

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"[START] 결과를 {out_dir}에 복사합니다")

    lines = mapping_file.read_text(encoding="utf-8").splitlines()
    copied = 0
    for raw in lines:
        line = raw.strip()
        if not line or "=>" not in line:
            continue
        ex_part, rest = line.split("=>", 1)
        ex_name = ex_part.strip()
        orig_str = rest.rsplit("  (", 1)[0].strip()
        orig_path = Path(orig_str)

        ex_path = extracted_root / ex_name
        prefix = f"{ex_path.stem}__{orig_path.parent.name}__{orig_path.stem}"

        if ex_path.exists():
            shutil.copy2(ex_path, out_dir / f"{prefix}__추출{ex_path.suffix}")
        else:
            print(f"[WARN] 추출 이미지 없음: {ex_path}")

        if orig_path.exists():
            shutil.copy2(orig_path, out_dir / f"{prefix}__원본{orig_path.suffix}")
        else:
            print(f"[WARN] 원본 이미지 없음: {orig_path}")

        copied += 1

    print(f"[DONE] {copied}쌍 복사 완료 → {out_dir}")


if __name__ == "__main__":
    main()
