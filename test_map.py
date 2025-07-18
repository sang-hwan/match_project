# test_map.py
"""
JSON 특징으로 추출 ⇄ 원본 매핑(Color+ORB+EdgeSSIM)
"""
import argparse
import json
from pathlib import Path

from hwp_image_mapper import json_to_feat, sim_scores
from utils import Tee


def main() -> None:
    pa = argparse.ArgumentParser(description="특징 JSON 매핑")
    pa.add_argument("--json", "-j", default="image_features.json", help="특징 JSON 파일")
    pa.add_argument("--threshold", "-t", type=float, default=0.7, help="Overall 임계값")
    pa.add_argument("--outfile", default="image_mapping.txt", help="출력 파일명")
    args = pa.parse_args()

    log_dir = Path(__file__).parent / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    Tee(log_dir / "test_map_log.txt")

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    extracted = {k: json_to_feat(v) for k, v in data.items() if "\\" not in k and "/" not in k}
    originals = {k: json_to_feat(v) for k, v in data.items() if "\\" in k or "/" in k}

    print(f"[INFO] 추출 {len(extracted)}건, 원본 {len(originals)}건, 임계값 {args.threshold}")

    lines = []
    for ex_name, ex_feat in extracted.items():
        best_orig, best_scores = "", (0.0, 0.0, 0.0, -1.0)
        for orig_path, orig_feat in originals.items():
            scores = sim_scores(ex_feat, orig_feat)
            if scores[3] > best_scores[3]:
                best_scores, best_orig = scores, orig_path

        c, o, e, ov = best_scores
        if ov < args.threshold:
            print(f"[WARN] {ex_name} 매핑 불확실: Overall={ov:.3f} < {args.threshold}")
            continue

        line = f"{ex_name} => {best_orig}  (Overall={ov:.4f}, C={c:.3f}, O={o:.3f}, E={e:.3f})"
        print(f"[MAP] {line}")
        lines.append(line)

    Path(args.outfile).write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {args.outfile} 저장 ({len(lines)}쌍)")


if __name__ == "__main__":
    main()
