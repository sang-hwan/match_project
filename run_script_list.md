# 1) 추출
python 01_extract_hwp.py doc.hwpx images_output

  ```powershell
  간단 사용 예시)
  # 1) HWPX에서 50개만 샘플 추출
  python 01_extract_hwp.py doc/sample.hwpx .\images_output -s 50

  # 2) HWP(olefile 필요). 프리뷰/썸네일은 기본 제외
  python 01_extract_hwp.py doc/sample.hwp .\images_output
  ```


# 2) 전처리(+ 매핑 메타)
python 02_preprocess.py images_output processed/extracted \
                        target_data/원본 processed/reference \
                        --low-size 640 --enable-high

  ```powershell
  # 전처리(+ 매핑 JSON 생성)
  python 02_preprocess.py ".\images_output" ".\processed\extracted" ^
                          ".\target_data\자동등록 사진 모음" ".\processed\reference" ^
                          --low-size 640 --enable-high --high-size 1280 ^
                          --pad-mode edge --pad-multiple 16 ^
                          --colorspace lab --clahe-clip 2.0 --clahe-grid 8 ^
                          --gray-clahe --map-json preprocess_mapping.json --out-ext png
  ```


# 3) 후보 생성(+ 캐시/무결성)
python 03_candidates.py processed/extracted processed/reference \
                        --mapping preprocess_mapping.json \
                        --out candidates.json --cache .cache/features \
                        --verify

  ```powershell
  python 03_candidates.py ".\processed\extracted" ".\processed\reference" ^
      --mapping preprocess_mapping.json ^
      --out candidates.json ^
      --cache .cache/features --workers 8 ^
      --phash-threshold 36 --hist-threshold 0.75 --min-cand-per-basis 8 ^
      --phash-step 2 --phash-max 48 --verify
  ```


# 4) 매칭(스코어링→필터→전역 1:1)
python 04_match.py --candidates candidates.json \
                   --mapping preprocess_mapping.json \
                   --images-root processed \
                   --scores pair_scores.csv \
                   --output mapping_result.json \
                   --min-inliers 8 --min-inlier-ratio 0.15

  ```powershell
  python 04_match.py ^
    --candidates ".\candidates.json" ^
    --mapping ".\preprocess_mapping.json" ^
    --scores ".\map_dist\pair_scores.csv" ^
    --output ".\map_dist\mapping_result.json" ^
    --orb-nfeatures 1000 --ratio 0.75 --ransac-th 5.0 ^
    --min-inliers 8 --min-inlier-ratio 0.15 ^
    --enable-recall-sweep ^
    --min-inliers-b 6 --min-inlier-ratio-b 0.12 ^
    --top2-margin 4 --top2-multiplier 1.25 ^
    --score-alpha 5.0 ^
    --assign greedy
  ```


# 5) 평가+시각점검
python 05_eval_inspect.py --mapping-json mapping_result.json \
                          --scores-csv pair_scores.csv \
                          --labels-csv hand_label.csv \
                          --mapping-meta preprocess_mapping.json \
                          --inspect-root-ex images_output \
                          --inspect-root-ref "target_data/원본" \
                          --inspect-out inspect_pairs --export-manifest

  ```powershell
  python 03_candidates.py ".\processed\extracted" ".\processed\reference" ^
      --mapping preprocess_mapping.json ^
      --out candidates.json ^
      --cache .cache/features --workers 8 ^
      --phash-threshold 36 --hist-threshold 0.75 --min-cand-per-basis 8 ^
      --phash-step 2 --phash-max 48 --verify
  ```


# ───────────────────────── 6. HWP 스트림 교체
# (코드상 인자명: --map-json / --src-hwp / --dst-hwp)
python 6_replace_stream.py `
    --map-json "mapping_result.json" `
    --src-hwp  "target_data\기계설비 성능점검 결과보고서(종합 1).hwp" `
    --dst-hwp  "target_data\기계설비 성능점검 결과보고서(종합 1)_수정본.hwp"
