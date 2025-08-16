# HWP 이미지 추출·매칭·패치 파이프라인

한글(HWP/HWPX) 문서에서 내장 이미지를 **추출 → 전처리 → 후보 생성 → 매칭/스코어링 → 평가/검수 → HWP(OLE) 본문 패치**까지 한번에 수행하는 CPU‑only 파이프라인입니다.

> 핵심 목표: HWP 문서 속 BinData 이미지를 외부 **참조 이미지**와 자동 매칭하고, 검수/피드백 루프를 거쳐 **안전하게 교체(패치)** 하는 것.

---

## 파이프라인 한눈에 보기

```
[01 추출]      01_extract_hwp.py
      → images_output/ + extracted_manifest.json
[02 전처리]    02_preprocess.py
      → processed/{extracted,reference}/ + preprocess_mapping.json
[03 후보]      03_candidates.py
      → candidates.json (+ feature/geom 캐시)
[04 매칭]      04_match.py
      → pair_scores.csv, mapping_result.json
[05 평가/검수] 05_eval_inspect.py
      → evaluation_report.json, feedback.json, inspect_pairs/
[06 패치]      06_patch_hwp_bindata.py (HWP(OLE) 전용)
      → *_수정본.hwp
```

- **증분 실행/피드백 루프**: 05단계가 만든 `feedback.json`을 입력해 02/03/04를 **부분 재실행**하며 결과를 수렴시킵니다.
- **아티팩트 중심 설계**: 모든 단계는 다음 단계를 구동하는 **JSON/CSV 아티팩트**를 생성합니다.

---

## 요구 사항

- Python 3.9+ (권장)
- 필수 패키지
  - `numpy`, `opencv-python`
- 선택/조건부 패키지
  - `olefile` — HWP(OLE) 추출(01)과 패치(06)에 필요
  - `Pillow` — 일부 추출(01) 및 패치(06)에 필요

설치 예시:

```bash
pip install numpy opencv-python olefile pillow
```

---

## 폴더 레이아웃(권장)

```
project-root/
├─ data/
│  ├─ doc/                 # 원본 HWP/HWPX
│  └─ target_original/     # 참조(원본) 이미지
├─ images_output/          # 01 산출(추출 이미지 + manifest)
├─ processed/
│  ├─ extracted/           # 02 산출(추출 기준 변형)
│  └─ reference/           # 02 산출(참조 기준 변형)
├─ .cache/
│  └─ features/            # 03의 feature/geom 캐시
├─ artifacts/              # 02~05의 아티팩트(JSON/CSV)
└─ inspect_pairs/          # 05 검수팩
```

---

## 빠른 시작(Quick Start)

### A) 초기 1회 실행 세트

```powershell
# 1) 추출(HWP 예시; HWPX도 동일 방식)
python 01_extract_hwp.py ".\data\doc\문서.hwp" ".\images_output"

# 2) 전처리(+ 매핑 메타 생성)
python 02_preprocess.py `
  ".\images_output" ".\processed\extracted" `
  ".\data	arget_original" ".\processed
eference" `
  --low-size 640 --enable-high --high-size 1280 `
  --pad-mode edge --pad-multiple 16 `
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 `
  --gray-clahe --map-json ".\artifacts\preprocess_mapping.json" `
  --out-ext png

# 3) 후보 생성(+ 캐시/무결성 확인)
python 03_candidates.py `
  ".\processed\extracted" ".\processed
eference" `
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

# 6) HWP 스트림 교체(패치) ※ HWP(OLE) 전용
python 06_patch_hwp_bindata.py `
  --mapping-json ".\artifacts\mapping_result.json" `
  --mapping-meta ".\artifacts\preprocess_mapping.json" `
  --src-hwp  ".\data\doc\문서.hwp" `
  --dst-hwp  ".\data\doc\문서_수정본.hwp" `
  --extracted-manifest ".\images_output\extracted_manifest.json"
```

### B) 피드백 루프(02→05 반복)

```powershell
# 2’) 전처리(피드백 대상만 부분 재생성 권장)
python 02_preprocess.py `
  ".\images_output" ".\processed\extracted" `
  ".\data	arget_original" ".\processed
eference" `
  --low-size 640 --enable-high --high-size 1280 `
  --pad-mode edge --pad-multiple 16 `
  --colorspace lab --clahe-clip 2.0 --clahe-grid 8 `
  --gray-clahe --map-json ".\artifacts\preprocess_mapping.json" `
  --out-ext png `
  --feedback-in ".\artifacts\feedback.json" `
  --only-refs-from-feedback `
  --auto-trim-plus

# 3’) 후보(피드백 대상만 확장) + 병합
python 03_candidates.py `
  ".\processed\extracted" ".\processed
eference" `
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
# 중복 이미지 의심 시: --assign reuse  (1:N 허용)

# 5’) 평가 + 피드백 재생성
python 05_eval_inspect.py ...
```

> Linux/macOS는 역따옴표 없이 동일한 인자 구성을 Bash로 실행하세요.

---

## 각 스크립트 기능 요약

- **01_extract_hwp.py** — HWP(OLE) / HWPX(ZIP)에서 래스터 이미지를 시그니처 스니핑으로 추출하고, zlib/deflate 원복을 시도하여 바이트 단위로 저장합니다. 프리뷰/썸네일은 휴리스틱으로 기본 제외되며, `extracted_manifest.json`으로 추적성을 제공합니다.

- **02_preprocess.py** — 오토 트림 → 리사이즈+레터박스 → 색 보정(LAB‑CLAHE) 및 Gray‑CLAHE를 적용해 `extracted|reference × low|high × color|gray` 변형을 산출합니다. 결과 경로/파라미터 인덱스를 `preprocess_mapping.json`에 기록합니다. `--feedback-in`/`--only-refs*`로 부분 재생성/병합을 지원합니다.

- **03_candidates.py** — pHash(64‑bit, gray/low) BK‑tree 근접 탐색(반경 점진 확장) 후 HSV 히스토그램 Bhattacharyya 거리 필터를 적용합니다. 필요 시 종횡비/에지밀도 차이 프리필터를 활용하고, 후보 수가 부족하면 상위 유사도로 보강합니다. 캐시(JSON)로 재실행 가속.

- **04_match.py** — ORB/SIFT 특징 매칭(Lowe ratio + mutual check) → USAC/RANSAC 호모그래피로 기하 검증(inliers·ratio·RMSE) → (선택) SSIM/NCC/ECC/G‑SSIM 광학 검증을 통합해 점수화하고, Top‑2 분리 규칙과 Phase A/B 가드로 수용 여부를 결정합니다. 전역 할당은 1:1 greedy(기본) 또는 1:N reuse 지원.

- **05_eval_inspect.py** — (양성 라벨만 있을 때) Recall_L/Precision_LB, (선택) Wilson CI를 계산하고, (ref,ex) 검수팩과 Top‑N 대안을 출력합니다. `feedback.json`을 생성하여 전처리·후보 확장·검증 엄격화·reuse 힌트를 파이프라인에 환류합니다.

- **06_patch_hwp_bindata.py** — **HWP(OLE 전용)** BinData/* 스트림을 외부 이미지로 교체합니다. 원 해상도를 유지하고, 원본 스트림이 **Deflate**면 같은 모드로, **RAW**면 RAW로 기록합니다. JPEG/PNG는 품질·다운스케일 탐색으로 **원본 크기** 제약에 맞춥니다. HWPX(ZIP) 패치는 범위 외입니다.

---

## 생성/소비 아티팩트

- `images_output/` + `extracted_manifest.json` (01)
- `processed/{extracted,reference}/` + `preprocess_mapping.json` (02)
- `candidates.json` + `.cache/features/*` (03)
- `pair_scores.csv`, `mapping_result.json` (04)
- `evaluation_report.json`, `feedback.json`, `inspect_pairs/`, `suggestions_topN.csv` (05)
- `*_수정본.hwp` (06)

---

## 팁

- 대용량 처리 시 **캐시/부분 실행**을 적극 활용하세요: 03의 `--cache`, 02/03의 `--only-refs*`, 02의 `--skip-existing`.
- **검수팩(inspect_pairs/)** 으로 현장 확인을 빠르게 돌리고, `suspects`/Top‑2 지표를 우선 점검하세요.
- 중복 컨텐츠가 의심되면 04단계에서 `--assign reuse`로 1:N 할당을 허용하세요.

---

## 알려진 제한/이슈 & 빠른 해결

1) **04 ↔ 06 매핑 포맷 불일치**  
   - 04의 `mapping_result.json`은 `mapping: { "<ref_rel>": "<ex_rel>", ... }` (dict) 형식이며, 06은 `mapping: [ {"reference": "...","extracted":"..."} ]` (list‑of‑pairs)를 기대합니다. 실제 사용 전 06에서 이 둘을 모두 수용하도록 정규화 로직을 추가하세요.

   예시(06 내부에서 정규화):
   ```python
   mfield = mr.get("mapping", {})
   pairs = []
   if isinstance(mfield, dict):
       for ref_rel, ex_rel in mfield.items():
           pairs.append({"reference": ref_rel, "extracted": ex_rel})
   elif isinstance(mfield, list):
       pairs = mfield
   # 이후 for m in pairs: ...
   ```

2) **02 로그 출력 f-string 오탈자**  
   - 매핑 병합 저장 로그 한 곳에서 `print(f("[INFO] ..."))` 형태의 문법 오류가 있습니다. `print(f"[INFO] ...")`로 교정하세요.

3) **HWPX 패치 미지원**  
   - 06은 HWP(OLE)만 패치합니다(HWPX는 범위 외). 최종 산출이 HWPX인 경우 별도 툴이 필요합니다.

---

## 문제 해결(Troubleshooting)

- `cv2` ImportError → `pip install opencv-python` 설치 확인, Python 버전 호환성 점검.
- `olefile` 관련 에러 → HWP(OLE) 처리/패치 단계에서 필요합니다. `pip install olefile`.
- SIFT 미사용 경고 → OpenCV 빌드에 SIFT가 없으면 ORB만 사용됩니다. 정확도 향상을 원하면 SIFT 가용 빌드를 사용하세요.
- 03단계 후보가 거의 나오지 않음 → 05의 `feedback.json`에서 `expand_candidates`(반경/히스토/최소수) 확장 후 02/03/04 반복.
