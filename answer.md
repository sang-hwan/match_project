## 1) 전처리 정리 & 재현성 강화

### 수정 파일: `2_pre_process_image.py`  (경미 수정)

**이유**: 전처리 산출물의 폴더 규약과 로깅을 더 엄격히 하고, 경계/여백 트림 안정성 개선. 이후 단계에서 재현 가능해야 합니다.&#x20;

**핵심 변경점**

* [버그] 출력 경로 생성: `out_path = base/channel/out_name` 로 고정돼 있어 의도한 `low/gray` 등의 트랙 하위 폴더가 누락될 소지가 있습니다. `base`가 이미 `…/low` 이므로 `base/channel/out_name`은 맞지만, 로그/매핑 키에서도 **(track, channel) 정규화**를 명시적으로 사용하고 누락 시 에러를 내도록 합니다.
* 여백 트림을 Otsu 단일 임계 대신 **형태학적 열림/닫힘 + 가장 큰 연결요소(bbox)** 로 변경(색 배경/그림자 케이스 견고화).
* 저장시 **md5 해시**를 추가 기록(선택): `mapping[...]['md5']`—후단 캐시 키로 재사용.

---

## 2) 특징 캐시 공통화 (CPU 최적화)

### **신규 파일**: `3_cache_features.py`

**이유**: 후보 생성/검증에서 같은 이미지를 반복 해석하지 않도록 pHash, HSV‑hist, ORB‑descriptor를 **디스크 캐시**합니다. `4_B_extract_threshold.py`는 이미 `joblib.Memory(".cache/descriptors")`를 사용 중—이를 공용화합니다.&#x20;

**핵심 변경점**

* 모든 (track, channel)별 이미지에 대해

  * `phash`(64bit)
  * `hsv_hist`(3×32bin, L2 정규화)
  * `orb`: keypoints 수, descriptor(npz)
    를 `processed/.cache/` 하위에 저장.
* API: `load_or_compute_phash(path)`, `load_or_compute_hist(path)`, `load_or_compute_orb(path)`.

---

## 3) 후보 생성 단계 재설계 (강한 1차 필터)

### 수정 파일: `3_mk_phash_candidates.py`

**이유**: 현재 임계값이 느슨해 오탐 후보가 과다합니다. 먼저 **강한 음성 필터**로 후보폭을 좁히고, 이후 ORB/RANSAC으로 정밀 검증합니다.&#x20;

**핵심 변경점**

1. **동적 임계값 자동화**

   * `3_A_search_threshold.py`의 “전체 쌍 분포”가 아니라, **가까운 상위 M개(예: M=200)만 스캔**하여 **로컬 최소치 분포**를 기반으로 트랙별 임계값을 잡습니다(unsupervised). 즉, 각 extracted(gray)에 대해 original(gray)과의 pHash 상위 M 최소 거리 집합을 만들고, 그 **하위 분위수(예: 5~10%)** 를 “같을 가능성이 높은 영역”으로 보고 그 구간의 컷을 사용합니다.
   * 초기 권장 시작값(라벨 없을 때): **pHash ≤ 16~20**, **Hist ≤ 0.35~0.45**. (제공된 전체 분포에서 “다름”이 다수인 점을 감안해 보수적으로 설정)
   * 옵션 `--auto-threshold` 추가: 자동 산정→CLI로 확인 가능.
2. **BK‑Tree(해밍 거리) / LSH(선택)**

   * pHash는 해밍 거리이므로 **BK‑Tree**로 반경 탐색(≤R) 구현. 큰 전체 교차곱을 피하고 **O(log N)** 수준 근방 후보만 뽑습니다(완전 CPU 친화).
3. **Top‑K 제한**

   * (gray 기준) pHash 필터로 **Top‑K(예: 50)** 까지로 후보를 제한 → 이후 color‑hist로 2차 필터 (**AND**) 적용.
4. **출력에 점수 동봉**

   * 각 후보에 `{"name": “…", "phash": d, "hist": h, "pre_score": α·(1−d/64)+β·(1−h)}` 같이 가벼운 **사전 점수**를 저장, 뒤 단계에서 tie‑break로 사용.

> 변경 후 `candidates.json`이 기존 포맷을 유지하되, 후보 수가 현저히 줄어듭니다.

---

## 4) ORB + RANSAC 검증식 보강과 임계 산정 로직 보정

### 수정 파일: `4_B_extract_threshold.py`

**이유**: 현재 `score = inliers / kpA`는 **비대칭**이고, 라벨 기반 TPR 타겟 스캐닝이 빈약할 때 **thr=0.0**으로 붕괴합니다. 대칭 스코어와 강인한 추정 방식을 추가합니다.&#x20;

**핵심 변경점**

1. **대칭 점수**

   * `score = inliers / max(4, min(kpA, kpB))` 또는 `inliers / sqrt(kpA*kpB)` 로 교체(둘 중 CLI로 선택). 극단적 키포인트 편차에 덜 민감.
2. **라벨 없는 경우(무감독)**

   * 기존 knee 탐지 유지 + **반복적 trimming**: 상위 1% 이상치 제거 후 knee 재추정.
3. **라벨 있는 경우(반감독)**

   * 기존 “TPR ≥ target” 대신 **Youden’s J(=TPR−FPR) 최대** 또는 **F1 최대** 지점 선택 옵션(`--criterion {tpr,f1,j_stat}`) 추가.
   * 현재 JSON: `{ "orb_score": 0.0, "ci_low": 0.088..., "ci_high": 0.157... }` → ‘임계=0’은 사실상 무의미하므로 새 방식으로 **>0** 의 합리적 컷이 나오도록 합니다.&#x20;
4. **산출물**

   * `outputs/match_scores.csv` 그대로, `thresholds.json`에 `{"orb_score": t, "method": "...", "track": "low/high"}` 단위로 저장(트랙별 분리 가능).

---

## 5) 정합 및 1:1 제약을 **전역 최적화**로 해결

### 수정 파일: `4_verify_mapping.py` → **이름 유지, 내부 로직 확장**

**이유**: 현재는 원본별 “최선 후보”를 고르는 **로컬 결정**입니다. 서로 다른 원본이 **같은 추출**을 고르는 **중복 할당** 가능성이 있습니다. 전역적으로 1:1을 보장하는 **이분 매칭(헝가리안/최대 가중 매칭)** 이 필요합니다.&#x20;

**핵심 변경점**

1. 후보쌍 만들기

   * 3단계 후보에서 온 리스트를 기반으로, ORB 스코어(§4) ≥ 임계인 쌍만 **유효 엣지**로 채택.
2. **가중치**

   * `w = ORB_score`(정규화) + `γ·pre_score`(pHash/hist 기반 사전 점수)
   * 가중치가 낮은 쌍은 엣지에서 제외(희박화).
3. **전역 1:1 할당**

   * `scipy.optimize.linear_sum_assignment`(가중 비용 최소화)이면 `cost = 1−w`로 구성. 할당 불가(임계 미달)는 휴지 값으로 처리.
   * 결과: **중복 없는** `mapping_result.json` 도출.
4. SIFT fallback은 유지(옵션). ORB 파라미터는 CPU 환경 고려로 `nfeatures` 700 전후 유지.&#x20;

---

## 6) 평가 체계 고도화 (라벨 수가 적어도 타당하게)

### 수정 파일: `5_A_evaluate_mapping.py`

**이유**: 지금은 “예측된 단일 쌍을 라벨로 긍/부 표기” 방식이라 TN=0이 빈번. 평가를 **랭킹 기반**으로 확장하고, **to_label.csv**도 “허들 주변 샘플” 위주로 제안합니다.&#x20;

**핵심 변경점**

1. **랭킹 지표 추가**

   * `Precision@1 / Recall@K / mAP`(K=5,10)
   * `ROC-AUC / PR-AUC`(가능 시)
   * 트랙별/채널별 분할 리포트.
2. **라벨링 제안 개선**

   * 현재도 임계 주변(`thresholds.json`) 중심으로 `to_label.csv`를 뽑지만, 여기에 **전역 매칭에서 탈락한 근접 후보**(가중치 차이가 근소했던 2~3위)도 포함 → **능동 라벨링**으로 오판 교정이 빠릅니다.
3. **리포트**

   * 기존 MD 출력 유지 + `error_pairs.html`(TP/FP/FN 시각 검토) 이미 지원 중.&#x20;

---

## 7) 결과 확인/적용 도구 정리

* `5_check_map_result.py`: 중복 할당/누락 로그 강화(“같은 추출 재사용” 경고는 이미 존재). 필요시 **원본·추출 썸네일 타일링** 옵션 추가.&#x20;
