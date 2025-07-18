# test_info.py
"""
추출·원본 이미지의 Color+ORB+Edge 특징을 JSON으로 저장
"""
import argparse
import json
import shutil
from pathlib import Path

from hwp_image_mapper import compute_features, feat_to_json
from utils import IMG_EXTS, Tee


def collect_feats(img_paths, key_fn):
    res = {}
    for p in img_paths:
        try:
            res[key_fn(p)] = feat_to_json(compute_features(p))
            print(f"[OK ] 특징 추출 성공: {p}")
        except Exception as e:
            print(f"[WARN] 특징 추출 실패: {p} ({e})")
    return res


def main() -> None:
    pa = argparse.ArgumentParser(description="이미지 특징(JSON) 저장 도구")
    pa.add_argument("extracted_dir", help="추출 이미지 폴더 경로")
    pa.add_argument("originals_dir", help="원본 이미지 폴더 경로")
    pa.add_argument(
        "--outfile", "-o", default="image_features.json", help="출력 JSON 파일명"
    )
    args = pa.parse_args()

    log_dir = Path(__file__).parent / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tee(log_dir / "test_info_log.txt")

    ex_dir, orig_dir = Path(args.extracted_dir), Path(args.originals_dir)
    ex_imgs = [p for p in ex_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    orig_imgs = [p for p in orig_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    print(f"[START] 추출 폴더  : {ex_dir} ({len(ex_imgs)}장)")
    print(f"[START] 원본 폴더  : {orig_dir} ({len(orig_imgs)}장)")
    print(f"[START] 출력 파일  : {args.outfile}")

    data = {}
    data.update(collect_feats(ex_imgs, key_fn=lambda p: p.name))
    data.update(collect_feats(orig_imgs, key_fn=lambda p: str(p)))

    out_path = Path(args.outfile)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] JSON 저장 완료: {out_path}  (총 {len(data)}개)")


if __name__ == "__main__":
    main()
