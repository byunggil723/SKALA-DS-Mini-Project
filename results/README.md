# Results

이 폴더는 피처 엔지니어링, 모델 선택, 최종 평가에서 생성된 결과를 저장한다.

## 최상위 결과

### `model_performance.csv`

프로젝트의 핵심 성능표다. 최종 선택 모델의 Train, Valid, Test MAPE와 구간 사이의 성능 격차를 저장한다.

| `index` | 의미 |
|---|---|
| `Train (Batch 1 CV)` | Batch 1 개발 데이터의 protocol-grouped nested CV 평균 MAPE |
| `Valid (Batch 1 Hold-out)` | 학습에서 보지 않은 Batch 1 충전 protocol에 대한 MAPE |
| `Test (Batch 2)` | Batch 1로 학습한 최종 모델의 Batch 2 외부 평가 MAPE |
| `Gap (Train-Valid)` | Valid MAPE - Train MAPE. 내부 과적합 신호 |
| `Gap (Valid-Test)` | Test MAPE - Valid MAPE. 배치 일반화 저하 신호 |
| `Gap (Target-Test)` | Test MAPE - 논문 목표 9.1% |

현재 결과는 Train 7.62%, Valid 7.96%, Test 37.46% MAPE다. Train과 Valid는 비슷하지만 Batch 2에서 오차가 크게 증가했으므로, 내부 과적합보다 배치 분포 이동이 더 큰 문제로 해석한다.

### `feature_table.csv`

원본 MAT에서 계산한 배터리 셀별 요약표다. 한 행이 하나의 배터리 셀을 나타낸다. 실제 수명과 함께 초기 100사이클의 용량·내부저항·온도·충전시간·ΔQ(V)에서 계산한 **기술통계와 파생변수**가 들어 있다.

- 식별·그룹: `cell_key`, `cell_index`, `batch`, `batch_date`
- Target: `cycle_life`
- 충전 조건: `policy_readable`, `first_stage_c_rate`, `max_c_rate`
- 용량: `initial_QDischarge`, `QDischarge_mean_100`, `QDischarge_cycle100_minus_cycle10`
- 상태·운전: `IR_mean_100`, `temperature_slope_100`, `chargetime_mean_100`
- ΔQ(V): `dq_mean`, `dq_std`, `dq_min`, `dq_range`, `dq_kurtosis`

입력 변수는 cycle 2~100에서만 계산했다. 최종 모델이 사용하는 변수는 `dq_std`, `dq_kurtosis`, `initial_QDischarge`, `IR_mean_100`, `temperature_slope_100`, `chargetime_mean_100`, `max_c_rate`다.

### `final_model.joblib`

저장된 최종 scikit-learn 모델이다. Batch 1 전체로 재학습한 `log(cycle_life)` ElasticNet과 결측값 대치·표준화 파이프라인을 포함한다.

```python
import joblib

model = joblib.load("results/final_model.joblib")
prediction = model.predict(new_features)
```

`new_features`는 위의 최종 7개 피처를 같은 이름과 순서의 DataFrame으로 제공해야 한다.

## `tables/` 세부 결과

### `model_comparison.csv`

Linear, Ridge, ElasticNet, RandomForest, HistGradientBoosting, Dummy 모델을 원본 target과 log target에서 비교한 표다.

- `cv_mape_percent`: nested CV 평균 MAPE
- `cv_std_percent`: CV fold MAPE의 표준편차
- `holdout_mape_percent`: Batch 1 protocol hold-out MAPE
- `fold_mapes`: 각 outer fold의 MAPE
- `best_params`: 내부 탐색에서 선택된 하이퍼파라미터

모델 선택은 hold-out이나 Batch 2 성능이 아닌 Batch 1 nested CV MAPE를 주 기준으로 수행했다.

### `feature_ablation.csv`

선택된 모델에 여러 피처 조합을 적용한 결과다. 특정 피처를 제거하거나 `dq_min`을 `dq_std`, `dq_mean`, `dq_range`로 대체했을 때의 성능을 비교한다. `dq_std_substitution`이 최종 피처 조합으로 선택됐다.

### `split_manifest.csv`

Batch 1의 각 셀이 `Development train`과 `Batch 1 hold-out` 중 어디에 배정됐는지 기록한다. `policy_readable`도 함께 저장해 두 구간 사이에 충전 protocol 중복이 없는지 검사할 수 있다.

### `predictions_and_errors.csv`

검증·테스트 셀별 예측 결과다.

- `cycle_life`: 실제 수명
- `prediction`: 예측 수명
- `signed_error`: `prediction - cycle_life`. 양수면 과대예측, 음수면 과소예측
- `absolute_percentage_error`: 셀별 절대 백분율 오차
- `evaluation_set`: 해당 셀이 속한 평가 구간

오차가 큰 셀을 찾거나 특정 protocol에서 오차가 반복되는지 분석할 때 사용한다.

### `error_analysis_summary.csv`

평가 구간별 오차를 요약한다.

- `mape_percent`: 평균 절대 백분율 오차
- `median_ape_percent`: 셀별 APE의 중앙값
- `mean_signed_error_cycles`: 평균 예측 편향
- `overprediction_rate_percent`: 실제보다 높게 예측한 셀의 비율

Batch 2는 MAPE 37.46%, 평균 signed error +179.1 cycles, 과대예측 비율 94.87%다.

### `batch2_error_by_life_group.csv`

Batch 2 셀을 Short(`<500`), Middle(`500~1000`), Long(`>1000`) 수명 그룹으로 나눠 오차를 요약한다. 어떤 수명 구간에서 모델이 실패하는지 확인할 수 있다. Short 28개 셀의 MAPE는 40.39%이며 모두 과대예측됐다.

### `batch_distribution_shift.csv`

Batch 1/2/3의 표본 수, 단수명 비율, target과 최종 피처의 평균·중앙값을 비교한다. Batch 1의 500사이클 미만 셀 비율은 0%인 반면 Batch 2는 71.79%로, 외부 평가 성능 저하의 주요 근거다.

### `supplementary_metrics.csv`

MAPE 외의 보조 평가 지표를 저장한다.

- `mae_cycles`: 평균 절대 오차를 cycle 단위로 표현
- `rmse_cycles`: 큰 오차에 더 큰 패널티를 주는 cycle 단위 지표
- `r2`: target 분산 설명력. 1에 가까울수록 좋고 0 이하면 단순 평균 예측보다도 나쁜 수 있다.

Batch 3 전체 labeled 셀과 논문 기준 품질 제외 후 결과도 보조적으로 포함한다.

### `run_metadata.json`

해당 실행의 재현성 메타데이터다.

- random seed
- 최종 모델과 target 변환
- 선택된 피처 세트와 컬럼
- 최적 하이퍼파라미터
- Batch별 표본 수
- 학습/hold-out protocol 중복 여부
- 논문 비교 목표 MAPE

## 중간 산출물

`processed/` 폴더는 `python src/preprocess.py`가 만드는 EDA용 중간 CSV를 담는다.

- `cell_data.csv`: 셀당 1행의 메타데이터와 `cycle_life`
- `cycle_summary.csv`: 셀·사이클당 1행의 용량, 내부저항, 온도, 충전시간

이 파일은 원본 MAT에서 다시 만들 수 있고 용량이 크므로 Git에서 제외된다.

## EDA 그림과 보고서

`figures/`에는 cycle life 분포, Batch별 비교, 방전 용량 열화, ΔQ(V), knee point, C-rate, 피처 상관관계 그림이 저장된다. 노트북에서 `SAVE_FIGURES = True`로 다시 생성할 수 있으며, README와 보고서에서 표시할 수 있도록 Git에 포함한다.

`reports/`에는 EDA 결과를 정리한 Markdown 보고서와 프로젝트 설계/발표 PDF가 있다. 이 파일은 분석 결과의 문서화 산출물이므로 Git에 포함한다.

## 결과 재생성

```bash
# 기존 feature_table.csv를 사용해 모델만 재실행
python src/train.py

# MAT에서 피처 추출부터 전체 재실행
python src/train.py --rebuild-features
```

재실행하면 동일한 이름의 결과 파일을 갱신한다.
