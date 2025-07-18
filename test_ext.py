# test_ext.py
"""
HWP 이미지 추출 전용 데모
"""
import argparse
import shutil
from pathlib import Path

from hwp_image_mapper import extract_images, detect_type
from utils import Tee

def main() -> None:
    pa = argparse.ArgumentParser(description="HWP 이미지 추출 테스트")
    pa.add_argument("hwp_file", help="HWP/HWPX 파일 경로")
    pa.add_argument("--sample", "-s", type=int, help="추출할 이미지 개수 제한")
    args = pa.parse_args()

    log_dir = Path(__file__).parent / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tee(log_dir / "test_ext_log.txt")

    hwp = Path(args.hwp_file)
    if not hwp.exists():
        print(f"[ERR] 파일 없음: {hwp}")
        return

    print(f"[INFO] HWP 파일   : {hwp}  ({detect_type(hwp)})")
    print(f"[INFO] 샘플 개수  : {args.sample if args.sample else '전체'}")

    tmp = Path("_tmp_extract_test")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()

    imgs = extract_images(hwp, tmp, args.sample)
    print(f"[RESULT] 추출 완료: {len(imgs)}개 이미지가 {tmp}에 저장됨")

if __name__ == "__main__":
    main()
