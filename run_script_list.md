A) 초기 1회 실행 세트(보수적 기준)

```powershell
# 1) 추출 (HWPX 예시; HWP도 동일)
python 01_extract_hwp.py ".\data\doc\기계설비 성능점검 결과보고서(종합 1).hwp" ".\images_output"
# python 01_extract_hwp.py ".\doc\sample.hwpx" ".\images_output"

# 2) 전처리(+ 매핑 메타 생성)
python 02_preprocess.py `
  ".\images_output" ".\processed\extracted" `
  ".\data\target_original" ".\processed\reference" `
  --low-size 640 --enable-high --high-size 1280 `
  --pad-mode edge --pad-multiple 16 `
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 `
  --gray-clahe --map-json ".\artifacts\preprocess_mapping.json" `
  --out-ext png

# 3) 후보 생성(+ 캐시/무결성 확인)
python 03_candidates.py `
  ".\processed\extracted" ".\processed\reference" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --out ".\artifacts\candidates.json" `
  --cache ".\.cache\features" --workers 8 `
  --phash-threshold 28 --hist-threshold 0.70 --min-cand-per-basis 5 `
  --phash-step 2 --phash-max 36 `
  --prefilter-aspect 0.50 --prefilter-edge 0.30 `
  --verify

# 4) 매칭(스코어링→필터→전역 1:1)
python 04_match.py `
  --candidates ".\artifacts\candidates.json" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --scores ".\artifacts\pair_scores.csv" `
  --output ".\artifacts\mapping_result.json" `
  --use-orb --use-sift `
  --ratio 0.75 --mutual-check `
  --use-usac --ransac-th 5.0 `
  --min-inliers 12 --min-inlier-ratio 0.22 `
  --enable-recall-sweep `
  --min-inliers-b 8 --min-inlier-ratio-b 0.15 `
  --top2-margin 4 --top2-multiplier 1.25 `
  --verify-photometric --ssim-th 0.80 --ncc-th 0.65 `
  --enable-ecc --use-gssim `
  --assign greedy

# 5) 평가 + 검수팩 + 피드백 생성
# (labels CSV가 없어도 경고 후 정상 진행)
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
  --export-report ".\artifacts\evaluation_report.json" `
  --export-feedback ".\artifacts\feedback.json"

# 6) HWP 스트림 교체(패치) ※ HWP(OLE) 전용 (변경 없음)
python 06_patch_hwp_bindata.py `
  --mapping-json ".\artifacts\mapping_result.json" `
  --mapping-meta ".\artifacts\preprocess_mapping.json" `
  --src-hwp  ".\data\doc\기계설비 성능점검 결과보고서(종합 1).hwp" `
  --dst-hwp  ".\data\doc\기계설비 성능점검 결과보고서(종합 1)_수정본.hwp" `
  --extracted-manifest ".\images_output\extracted_manifest.json"
```

---

B) 피드백 루프 실행 세트(02→05 반복)

```powershell
# 2’) 전처리(피드백 대상만 부분 재생성 권장)
python 02_preprocess.py `
  ".\images_output" ".\processed\extracted" `
  ".\data\target_original" ".\processed\reference" `
  --low-size 640 --enable-high --high-size 1280 `
  --pad-mode edge --pad-multiple 16 `
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 `
  --gray-clahe --map-json ".\artifacts\preprocess_mapping.json" `
  --out-ext png `
  --feedback-in ".\artifacts\feedback.json" `
  --only-refs-from-feedback `
  --auto-trim-plus
  # (--skip-existing 를 추가하면 미대상 산출물은 그대로 재사용)

# 3’) 후보(피드백 대상만 확장)
python 03_candidates.py `
  ".\processed\extracted" ".\processed\reference" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --out ".\artifacts\candidates.json" `
  --cache ".\.cache\features" --workers 8 `
  --phash-threshold 28 --hist-threshold 0.70 --min-cand-per-basis 5 `
  --phash-step 2 --phash-max 36 `
  --prefilter-aspect 0.50 --prefilter-edge 0.30 `
  --feedback-in ".\artifacts\feedback.json" `
  --only-refs-from-feedback `
  --verify

# 4’) 매칭(참조별 가드 오버라이드 반영)
python 04_match.py `
  --candidates ".\artifacts\candidates.json" `
  --mapping ".\artifacts\preprocess_mapping.json" `
  --scores ".\artifacts\pair_scores.csv" `
  --output ".\artifacts\mapping_result.json" `
  --use-orb --use-sift `
  --ratio 0.75 --mutual-check `
  --use-usac --ransac-th 5.0 `
  --min-inliers 12 --min-inlier-ratio 0.22 `
  --enable-recall-sweep `
  --min-inliers-b 8 --min-inlier-ratio-b 0.15 `
  --top2-margin 4 --top2-multiplier 1.25 `
  --verify-photometric --ssim-th 0.80 --ncc-th 0.65 `
  --enable-ecc --use-gssim `
  --assign greedy `
  --feedback-in ".\artifacts\feedback.json"
  # 중복 이미지가 의심되면: --assign reuse  (1:N 허용)

# 5’) 평가+피드백(반복)
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
  --export-report ".\artifacts\evaluation_report.json" `
  --export-feedback ".\artifacts\feedback.json"
```