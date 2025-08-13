```powershell
# 1) 추출 (HWPX 예시; HWP도 동일하게 가능)
python 01_extract_hwp.py ".\data\doc\기계설비 성능점검 결과보고서(종합 1).hwp" ".\images_output"
# python 01_extract_hwp.py ".\doc\sample.hwpx" ".\images_output"

# 2) 전처리(+ 매핑 메타 생성)
python 02_preprocess.py `
  ".\images_output" ".\processed\extracted" `
  ".\data\target_original" ".\processed\reference" `
  --low-size 640 --enable-high --high-size 1280 `
  --pad-mode edge --pad-multiple 16 `
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 `
  --gray-clahe --map-json ".\artifacts\preprocess_mapping.json" --out-ext png

# 3) 후보 생성(+ 캐시/무결성 확인)  ← 개정본 반영: 보수적 기본값 & 프리필터
python 03_candidates.py `
  ".\processed\extracted" ".\processed\reference" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --out ".\artifacts\candidates.json" `
  --cache ".\.cache\features" --workers 8 `
  --phash-threshold 28 --hist-threshold 0.70 --min-cand-per-basis 5 `
  --phash-step 2 --phash-max 36 `
  --prefilter-aspect 0.50 --prefilter-edge 0.30 `
  --verify

# 4) 매칭(스코어링→필터→전역 1:1)  ← 개정본 반영: 상호검증/USAC/포토메트릭/alpha
python 04_match.py `
  --candidates ".\artifacts\candidates.json" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --scores ".\artifacts\pair_scores.csv" `
  --output ".\artifacts\mapping_result.json" `
  --use-orb --use-sift `
  --orb-nfeatures 1000 --ratio 0.75 --mutual-check `
  --ransac-th 5.0 `
  --min-inliers 12 --min-inlier-ratio 0.22 `
  --enable-recall-sweep `
  --min-inliers-b 8 --min-inlier-ratio-b 0.15 `
  --top2-margin 4 --top2-multiplier 1.25 `
  --alpha 5.0 `
  --verify-photometric --ssim-th 0.80 --ncc-th 0.65 `
  --assign greedy

# 5) 평가 + 육안 검수팩 출력  ← 개정본 반영: PU 평가/추천/리포트 플래그
# labels CSV가 없으면 --labels-csv 줄은 있어도 경고만 출력하고 정상 진행됩니다.
python 05_eval_inspect.py `
  --mapping-json ".\artifacts\mapping_result.json" `
  --scores-csv   ".\artifacts\pair_scores.csv" `
  --mapping-meta ".\artifacts\preprocess_mapping.json" `
  --labels-csv   ".\hand_label.csv" `
  --inspect-out  ".\inspect_pairs" `
  --copy-variant pre_low_color `
  --allow-basename-fallback `
  --suggest-n    20 `
  --suggest-only-unmatched `
  --suggest-out  ".\artifacts\suggestions_topN.csv" `
  --export-report ".\artifacts\evaluation_report.json"

# 6) HWP 스트림 교체(패치)  ※ HWP(OLE) 전용 (변경 없음)
# --dst-hwp 의 부모 폴더가 새 폴더라면 미리 만들어 두세요(같은 폴더면 보통 필요 없음).
python 06_patch_hwp_bindata.py `
  --mapping-json ".\artifacts\mapping_result.json" `
  --mapping-meta ".\artifacts\preprocess_mapping.json" `
  --src-hwp  ".\data\doc\기계설비 성능점검 결과보고서(종합 1).hwp" `
  --dst-hwp  ".\data\doc\기계설비 성능점검 결과보고서(종합 1)_수정본.hwp" `
  --extracted-manifest ".\images_output\extracted_manifest.json"
```