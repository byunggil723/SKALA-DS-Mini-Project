# ESS 배터리 수명 예측

## 목적

이 프로젝트는 MIT-Stanford Battery Dataset에서 각 배터리 셀의 초기 100사이클 데이터를 이용해 최종 수명(`cycle_life`)을 예측하는 것을 목적으로 한다. 모델은 방전 용량, 내부저항, 온도, 충전시간, 충전 조건, ΔQ(V) 통계처럼 수명 초기에 확인할 수 있는 정보만 사용한다.

단순히 높은 예측 성능을 얻는 데 그치지 않고, 초기 사이클에서 관측되는 어떤 신호가 배터리 수명과 관련되는지 분석한다. 또한 Batch 1에서 개발한 모델이 다른 시점에 수집된 Batch 2에서도 성능을 유지하는지 확인하고, 성능이 떨어진다면 그 원인을 셀별 오차와 배치별 데이터 분포 차이에서 찾는다.

최종적으로는 이 모델이 ESS(BESS) 운영에서 초기 위험 셀 선별, 추가 진단 대상 선정, 보증 등급 분류, 예비 부품 및 교체 계획 수립에 어느 정도 활용될 수 있는지 살펴본다. 동시에 새로운 제조 lot나 운전 조건에서 수명을 체계적으로 높게 예측할 수 있다는 한계도 함께 확인한다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (`2017-05-12`), 수명값이 있는 셀 46개
- 내부 검증: Batch 1 development 33개와 충전 protocol hold-out 13개
- 외부 평가 데이터: Batch 2 (`2018-02-20`), 수명값이 있는 셀 39개
- 보조 평가 데이터: Batch 3 (`2018-04-12`), 수명값이 있는 셀 44개
- 태스크: **Regression(회귀)** — 연속형 값인 최종 수명, 즉 사용 가능한 사이클 수를 예측한다.

이 프로젝트에서는 장수명과 단수명을 두 범주로만 나누는 분류 대신 회귀를 선택했다. 실제 `cycle_life`는 약 392~1,935사이클 범위의 연속형 값이다. 회귀를 사용하면 셀이 어느 범주에 속하는지만 판단하는 것이 아니라, 실제 수명보다 몇 사이클 높거나 낮게 예측했는지까지 확인할 수 있다.

### 전체 진행 흐름

```text
1. 프로젝트 실행 환경 준비
   └─ Python 가상환경(.venv)을 만들고 requirements.txt의 패키지를 설치한다.

2. 원본 데이터 다운로드
   └─ data/download_data.py로 Batch 1·2·3 MAT 파일 약 8.3GB를 data/에 받는다.

3. EDA용 데이터 변환
   └─ src/preprocess.py가 HDF5 참조 구조를 읽어 셀 단위 cell_data.csv와
      사이클 단위 cycle_summary.csv를 results/processed/에 만든다.

4. 탐색적 데이터 분석
   └─ 01_EDA.ipynb에서 수명 분포, 방전 용량 열화, ΔQ(V), 충전 protocol,
      Batch별 차이, 결측값과 다중공선성을 확인한다.

5. 셀별 초기 100사이클 요약 통계·파생변수 계산
   └─ cycle 2~100만 사용해 용량·내부저항·온도·충전시간·ΔQ(V)의 평균, 표준편차,
      기울기, 변화량 등을 계산하고 셀당 한 행인 results/feature_table.csv로 저장한다.

6. Batch 1 모델 개발
   └─ Batch 1을 development와 protocol hold-out으로 나눈다. development nested CV에서
      모델, target 변환, 하이퍼파라미터, 피처 조합을 선택한다.

7. 보지 않은 protocol 내부 검증
   └─ 선택한 구성을 Batch 1 hold-out에 적용해 새로운 충전 protocol에도 잘 작동하는지 본다.

8. Batch 2 외부 평가
   └─ Batch 1 전체로 최종 모델을 재학습한 뒤 Batch 2에서 재튜닝 없이 성능을 측정한다.

9. 실패 원인 해석
   └─ 셀별 예측 오차, 수명 그룹별 오차, 과대예측 비율, Batch 1·2·3의 target과
      피처 분포를 비교해 Batch 2에서 성능이 떨어진 이유를 찾는다.
```

수명 예측에 사용할 변수는 cycle 2~100의 데이터만 사용해 계산한다. cycle 1은 일부 배치에서 측정값이 일괄적으로 0으로 기록되어 있어 집계 대상에서 제외했다. knee point나 마지막 100사이클의 기울기처럼 수명 후반부를 알아야 계산할 수 있는 정보는 EDA에서 열화 특성을 설명하는 용도로만 활용한다. 실제 모델에는 포함하지 않아 초기 정보만으로 최종 수명을 예측하도록 구성했다.

## 파일 구조

```text
.
├── data/
│   ├── README.md                    # 데이터 설명과 다운로드 안내
│   ├── download_data.py             # MAT 파일 다운로드 및 이어받기
│   └── *.mat                        # 원본 데이터, Git 제외
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # 전처리 확인과 탐색적 데이터 분석
│   ├── 02_feature_engineering.ipynb # 셀별 요약 통계와 파생변수 확인
│   └── 03_modeling.ipynb            # 모델 비교, 오차 분석, 외부 평가 해석
│
├── src/
│   ├── __init__.py                  # src를 Python 패키지로 인식시키는 파일
│   ├── preprocess.py                # 원본 MAT를 EDA용 셀/사이클 CSV로 변환
│   ├── features.py                  # 셀별 초기 100사이클 기술통계·파생변수 계산
│   └── train.py                     # 계산된 변수와 실제 수명으로 모델 학습·평가
│
├── results/
│   ├── README.md                    # 결과 파일과 컬럼 설명
│   ├── feature_table.csv            # 셀별 실제 수명, 기술통계, 파생변수
│   ├── final_model.joblib           # 전처리까지 포함한 최종 모델
│   ├── model_performance.csv        # Train/Valid/Test 핵심 성능표
│   ├── processed/                   # EDA용 중간 CSV, Git 제외
│   │   ├── cell_data.csv
│   │   └── cycle_summary.csv
│   ├── figures/                     # README·보고서에 표시할 EDA 그림
│   ├── reports/                     # EDA 보고서와 발표 PDF
│   └── tables/                      # 모델 비교, 오차, 실행 메타데이터
│
├── requirements.txt
├── .gitignore
└── README.md
```

용량이 큰 원본 데이터와 다시 만들 수 있는 중간 산출물은 Git에 올리지 않는다. `.gitignore`는 `data/*.mat`, `results/processed/`, Python cache인 `__pycache__/`와 `*.pyc`, 가상환경을 제외한다. 따라서 Python을 실행할 때 생기는 파이캐시 폴더도 push 대상에서 빠진다. 반면 셀별 특성표, 성능표, 세부 평가표, 보고서와 문서에서 사용하는 그림은 결과를 바로 확인할 수 있도록 Git에 포함한다.

### `src/` 파일의 역할

`__init__.py`는 실제 분석을 수행하는 파일이 아니다. Python이 `src` 폴더를 패키지로 인식하게 해 `from src.features import ...`처럼 모듈을 불러올 수 있게 한다. 현재는 패키지 설명 외에 다른 로직은 넣지 않았다.

`preprocess.py`는 EDA를 위한 전처리를 담당한다. MATLAB v7.3/HDF5 reference를 따라가며 셀 메타데이터와 사이클 요약값을 읽고, `cell_data.csv`와 `cycle_summary.csv`를 만든다. EDA 노트북이 필요한 특정 셀의 시계열, `Vdlin`, ΔQ(V)만 선택적으로 읽는 함수도 이 파일에 있다.

`features.py`는 각 배터리 셀의 초기 100사이클을 요약한다. 단순히 평균과 표준편차만 내는 것은 아니다. Batch 1·2·3 MAT 파일에서 cycle 2~100 구간을 읽고 방전 용량·내부저항·충전시간의 초기값과 평균을 계산한다. 온도의 cycle별 기울기, cycle 10과 100 사이의 방전 용량 변화량, protocol에서 추출한 C-rate, ΔQ(V)의 평균·표준편차·최솟값·범위·첨도도 만든다. 이 결과는 한 행이 하나의 셀을 나타내는 `feature_table.csv`에 저장하며, 표의 숫자들이 수명 예측 모델의 입력 변수가 된다.

`train.py`는 `feature_table.csv`의 셀별 숫자와 실제 수명을 읽어 예측 모델을 만든다. Batch 1을 development와 hold-out으로 나누고, 여러 회귀 모델과 target 변환을 nested CV로 비교하며, 변수를 빼거나 대체하는 ablation 실험을 거쳐 최종 구성을 고른다. 이후 Batch 1 전체로 모델을 다시 학습하고 Batch 2·3을 평가해 성능표, 예측 오차표, 메타데이터, 최종 모델을 저장한다. `--rebuild-features`를 붙이면 학습 전에 `features.py`를 먼저 실행해 셀별 특성을 다시 계산한다.

## 환경 설정

### 1. 프로젝트 루트로 이동

저장소를 내려받았다면 먼저 `README.md`가 있는 프로젝트 루트로 이동한다. 아래 값은 예시이므로 실제 GitHub 주소와 폴더명에 맞게 바꾼다.

```bash
git clone <프로젝트-GitHub-URL>
cd <프로젝트-폴더>
```

이미 프로젝트 폴더를 가지고 있다면 다음처럼 바로 이동하면 된다.

```bash
cd "/path/to/practice_materials"
```

이 문서의 모든 명령은 프로젝트 루트에서 실행한다. 다른 폴더에서 실행해도 일부 스크립트는 파일 위치를 기준으로 경로를 찾지만, 노트북과 상대경로를 일관되게 유지하려면 루트에서 시작하는 편이 안전하다.

### 2. Python 가상환경 생성

현재 프로젝트는 Python 3.11에서 검증했다. 다른 프로젝트와 패키지 버전이 섞이지 않도록 `.venv`를 만든다.

macOS와 Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

가상환경이 켜지면 터미널 앞에 `(.venv)`가 표시된다. 새 터미널을 열었다면 학습이나 노트북을 실행하기 전에 가상환경을 다시 활성화해야 한다.

### 3. 필요한 패키지 설치

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`에는 데이터 처리용 `numpy`, `pandas`, `scipy`, MAT v7.3 로딩용 `h5py`, 모델링용 `scikit-learn`, 시각화용 `matplotlib`·`seaborn`, 진단용 `statsmodels`, 모델 저장용 `joblib`, 노트북 환경을 위한 `jupyterlab`·`ipykernel`이 들어 있다.

설치가 끝나면 아래 명령으로 핵심 패키지를 모두 불러올 수 있는지 검사한다.

```bash
python -c "import numpy, pandas, scipy, sklearn, h5py, joblib; print('environment: OK')"
```

`environment: OK`가 출력되면 데이터를 받을 준비가 된 것이다.

### 4. 원본 MAT 파일 다운로드

프로젝트에서 사용하는 원본 파일은 용량이 크므로 Git에 포함하지 않는다. 저장소를 처음 받았다면 다음 명령으로 데이터를 받는다.

```bash
python data/download_data.py
```

| 파일                                                   | 역할                           | 예상 크기 |
| ------------------------------------------------------ | ------------------------------ | --------: |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1, 모델 개발과 최종 학습 |  약 3.0GB |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2, 최종 외부 평가        |  약 2.0GB |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3, 보조 평가             |  약 3.2GB |

세 파일은 `data/`에 저장되며 합치면 약 8.3GB다. 다운로드하기 전에 디스크 여유 공간과 네트워크 상태를 확인하는 편이 좋다.

다운로드 스크립트는 파일의 실제 byte 크기를 기대값과 비교한다.

- 파일이 없으면 처음부터 다운로드한다.
- 예상한 크기의 파일이 이미 있으면 `skip ... already complete`를 출력하고 넘어간다.
- 파일을 중간까지 받은 상태라면 서버가 HTTP Range를 지원할 때 이어받기를 시도한다.
- 완료 후 파일 크기가 다르면 손상되었거나 덜 받은 것으로 보고 오류를 낸다.

현재 폴더에 `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`도 있다면 EDA 전처리가 그 배치까지 포함한다. 이 파일은 선택 데이터라서 없어도 위 세 배치로 EDA와 모델링을 실행하는 데는 문제가 없다. 데이터만 다룬 설명은 `data/README.md`에서 볼 수 있다.

### 5. EDA용 CSV 생성

원본 데이터는 MATLAB v7.3/HDF5 형식이다. 셀 메타데이터와 사이클 시계열이 HDF5 reference로 연결돼 있어 CSV처럼 바로 읽을 수는 없다. `src/preprocess.py`는 참조를 따라가 값을 읽고 EDA에 필요한 두 테이블로 풀어낸다.

```bash
python src/preprocess.py
```

완료되면 `results/processed/`에 두 파일이 생긴다.

| 파일                | 한 행이 나타내는 대상 | 들어 있는 정보                                      |
| ------------------- | --------------------- | --------------------------------------------------- |
| `cell_data.csv`     | 배터리 셀 하나        | 배치, barcode, channel, 충전 protocol, `cycle_life` |
| `cycle_summary.csv` | 특정 셀의 한 사이클   | 방전·충전 용량, 내부저항, 온도, 충전시간            |

두 CSV는 `01_EDA.ipynb`가 읽는 중간 데이터다. 원본 MAT에서 다시 만들 수 있고 `cycle_summary.csv`의 용량이 크기 때문에 Git에는 올리지 않는다. 저장 위치를 다르게 지정해야 한다면 다음 옵션을 쓴다.

```bash
python src/preprocess.py \
  --data-dir /path/to/data \
  --output-dir /path/to/processed
```

### 6. JupyterLab 실행

```bash
jupyter lab
```

노트북은 파일명에 붙은 번호 순서대로 실행한다.

1. `notebooks/01_EDA.ipynb`: 데이터 품질과 주요 분포를 확인하고 탐색적 분석을 수행한다.
2. `notebooks/02_feature_engineering.ipynb`: 셀별 기술통계와 파생변수의 생성 결과 및 결측값을 점검한다.
3. `notebooks/03_modeling.ipynb`: 모델 비교 결과, 외부 평가 결과, 셀별 오차와 배치 차이를 해석한다.

## EDA

EDA는 처음부터 이상치를 지우거나 특정 배치를 제외하지 않고 데이터에 어떤 문제가 있는지 확인하는 단계부터 시작한다. 전체 데이터에는 141개 셀과 117,775개의 사이클 행이 있으며, 이 가운데 129개 셀에 `cycle_life`가 있다. 수명값이 없는 12개 셀은 분포 확인에는 사용할 수 있지만 지도학습 표본에서는 제외한다. 셀 식별자나 `(cell, cycle)` 조합의 중복은 확인되지 않았다.

### Cycle Life 분포

`cycle_life`가 있는 129개 셀의 수명은 392~1,935사이클에 걸쳐 있다. EDA에서는 Short(500사이클 미만) 28개, Middle(500~999사이클) 65개, Long(1,000사이클 이상) 36개로 나눠 비교했다.

전체로는 중수명 셀이 가장 많지만 배치별 구성은 크게 다르다. Batch 2에서는 셀의 71.79%가 단수명인 반면 Batch 3에서는 52.27%가 장수명이다.

**핵심 발견:** 배치마다 수명 범위와 장·단수명 비율이 크게 달라서 무작위 분할 점수만으로는 새로운 제조 배치에 대한 성능을 판단하기 어렵다. 따라서 Batch 1은 모델 개발에 사용하고 Batch 2는 끝까지 분리한 외부 평가 데이터로 사용했다.

### 열화 곡선 분석

cycle이 증가할 때 방전 용량이 어떻게 감소하는지 수명 그룹별로 비교했다. 초기 cycle 2~100 구간에서는 세 그룹 모두 평균 기울기가 표시 정밀도상 0에 가까워 초기 용량 곡선만 눈으로 보고 수명 차이를 구분하기가 쉽지 않았다. 반면 마지막 100사이클의 평균 기울기는 Short -0.0020, Middle -0.0010, Long -0.0007 Ah/cycle로 나타나 단수명 셀일수록 수명 후반의 용량 감소가 더 가팔랐다.

용량 저하가 눈에 띄게 가속되는 knee point의 중앙값은 Short 347.5, Middle 632.0, Long 885.5사이클이었다. 세 그룹 모두 knee point는 대체로 최종 수명의 77~79% 지점에서 나타났다.

**핵심 발견:** 장수명 셀은 knee point가 더 늦고 수명 후반의 열화 속도도 완만하다. 다만 knee point와 마지막 100사이클 기울기는 실제 수명 후반을 봐야 계산할 수 있으므로 모델 입력에는 넣지 않고 EDA 해석에만 사용했다.

### ΔQ(V) 곡선 분석

ΔQ(V)는 동일한 전압 구간에서 cycle 100의 방전 용량 곡선과 cycle 10의 곡선이 얼마나 달라졌는지를 나타낸다.

```text
ΔQ(V) = Q100(V) - Q10(V)
```

전체 141개 셀에서 ΔQ(V)를 정상적으로 계산했다. 대표 셀을 비교했을 때 단수명 셀은 약 3.0V 부근에서 평균 -0.051Ah, 장수명 셀은 약 2.9V 부근에서 평균 -0.024Ah 정도의 차이를 보였다. 이 값은 각 그룹의 대표 셀 3개를 비교한 근사치이므로 전체 모집단의 평균으로 해석하지 않는다.

ΔQ(V) 요약값과 수명 사이의 상관계수 절댓값은 `dq_min` 0.8273, `dq_std` 0.8235, `dq_range` 0.7999, `dq_mean` 0.7974로 높았다. 이 통계량끼리도 약 0.98 이상의 매우 강한 상관을 보였다.

**핵심 발견:** cycle 10과 100 사이에서 방전 곡선이 변한 정도는 초기 수명 예측에 유용하지만, 비슷한 정보를 담는 ΔQ(V) 통계를 모두 모델에 넣으면 변수가 중복된다. 따라서 대표 통계 하나를 중심으로 사용하고 대체 조합을 따로 비교했다.

### 충전 속도(C-rate)와 수명의 관계

충전 protocol 문자열에서 1단계 C-rate와 최대 C-rate를 추출해 수명과 비교했다. 수명과의 단순 상관계수는 1단계 C-rate -0.0783, 최대 C-rate -0.2595였다. 빠른 충전일수록 수명이 짧아지는 단순한 단조 관계는 확인되지 않았다.

| 충전 protocol    | 표준 구조 평균 수명 | newstructure 평균 수명 |
| ---------------- | ------------------: | ---------------------: |
| `4.8C(80%)-4.8C` | 560.6사이클 (`n=7`) | 1,333.3사이클 (`n=11`) |
| `5.2C(58%)-4C`   | 477.8사이클 (`n=4`) |    961.7사이클 (`n=3`) |
| `5.6C(26%)-4.5C` | 448.5사이클 (`n=6`) |    991.3사이클 (`n=3`) |

**핵심 발견:** 명목상 같은 protocol도 셀 구조에 따라 평균 수명이 크게 달랐다. C-rate만으로 수명을 설명할 수 없으며 이 관찰 자료만으로 “급속 충전이 수명을 단축한다”는 인과관계를 단정하지 않는다.

### 추가로 확인한 내용

- cycle 1에서 여러 측정값이 동시에 0인 셀은 46개로 전체의 32.62%였으며 모두 Batch 1에 속했다. 실제 물리 현상보다는 저장 과정의 placeholder일 가능성이 있어 원본은 삭제하지 않되 피처 집계는 cycle 2부터 시작한다.
- 초기 방전 용량과 초기 100사이클 평균 방전 용량의 절대 상관은 0.9667로 높았고 VIF도 크게 나타났다. 서로 비슷한 변수는 대표값만 남기거나 규제가 있는 모델을 사용해야 한다.
- 1단계 C-rate와 최대 C-rate의 절대 상관도 0.8827이었다. 두 값을 무조건 함께 넣기보다 대체 조합을 비교했다.
- `01_EDA.ipynb`의 `SAVE_FIGURES = True` 설정으로 그림을 `results/figures/`에 저장할 수 있다. README와 보고서에서 그림이 보이도록 이 폴더는 Git에 포함한다.

`01_EDA.ipynb`에서는 배치별 셀 수와 결측값, cycle 1의 0값, 수명 분포, 방전 용량 열화, ΔQ(V), C-rate, 피처 상관관계와 VIF를 직접 확인할 수 있다. `results/reports/`에는 Day 1 EDA를 정리한 Markdown 보고서와 프로젝트 설계·발표 자료 PDF가 있다.

## Modeling

### 피처 엔지니어링 전략

모델 입력 변수는 수명 초기에 실제로 알 수 있는 cycle 2~100 정보로 제한했다. `results/feature_table.csv`는 셀 하나를 한 행으로 두고 실제 수명, protocol, 초기 사이클의 기술통계와 파생변수를 함께 저장한다. `features.py`가 이 표를 생성하고, `02_feature_engineering.ipynb`에서 표본 수, 실제 수명 가용성, 결측값과 변수 조합을 점검한다.

| 구분      | 대표 피처                                                                        | 의미                                 |
| --------- | -------------------------------------------------------------------------------- | ------------------------------------ |
| ΔQ(V)     | `dq_mean`, `dq_std`, `dq_min`, `dq_range`, `dq_kurtosis`                         | cycle 10과 100의 방전 용량 곡선 차이 |
| 방전 용량 | `initial_QDischarge`, `QDischarge_mean_100`, `QDischarge_cycle100_minus_cycle10` | 초기 용량 수준과 변화량              |
| 상태·운전 | `IR_mean_100`, `temperature_slope_100`, `chargetime_mean_100`                    | 내부저항, 온도 변화, 충전시간        |
| 충전 조건 | `first_stage_c_rate`, `max_c_rate`                                               | protocol 문자열에서 추출한 C-rate    |

EDA에서 ΔQ(V) 통계끼리, 초기 방전 용량 변수끼리, C-rate 변수끼리 강한 상관이 확인됐다. 모든 변수를 한꺼번에 넣는 대신 대표 변수를 선택하고, 특정 변수를 제거하거나 다른 통계로 바꾸는 ablation 실험으로 조합의 안정성을 비교했다. 최종 조합은 `dq_std`, `dq_kurtosis`, `initial_QDischarge`, `IR_mean_100`, `temperature_slope_100`, `chargetime_mean_100`, `max_c_rate`의 7개 변수다.

기존 셀별 특성표로 모델만 다시 학습하려면 다음 명령을 사용한다.

```bash
python src/train.py
```

피처 계산 코드를 바꾸었거나 원본 MAT부터 전체 과정을 다시 실행하려면 `--rebuild-features`를 붙인다.

```bash
python src/train.py --rebuild-features
```

이 경우 세 MAT 파일에서 cycle 2~100 구간과 cycle 10/100의 `Qdlin`을 읽고 `results/feature_table.csv`를 덮어쓴 뒤 모델 학습과 평가를 이어간다. 원본 파일이 8GB를 넘으므로 기존 셀별 특성표를 쓰는 방법보다 시간이 더 걸린다.

기본 폴더가 아닌 다른 위치의 데이터를 쓰거나 결과를 따로 저장하려면 경로 옵션을 지정한다.

```bash
python src/train.py \
  --data-dir /path/to/data \
  --results-dir /path/to/results \
  --rebuild-features
```

### 모델 선택 및 근거

- 후보 모델: Dummy, Linear Regression, Ridge, ElasticNet, RandomForest, HistGradientBoosting
- 최종 모델: `log(cycle_life)`를 예측하는 ElasticNet
- 선택 이유: 학습 표본이 작고 초기 피처 사이의 상관이 강한 상황에서 계수를 규제할 수 있으며, 후보 가운데 Batch 1 nested CV MAPE가 가장 낮았다.

모델과 피처를 선택하는 순서는 다음과 같다.

1. `cycle_life`가 있는 셀만 지도학습 표본으로 남긴다.
2. Batch 1을 development 33개와 hold-out 13개로 나눈다. 충전 protocol을 그룹으로 묶어 두 구간에 같은 protocol이 섞이지 않게 한다.
3. development 안에서 protocol-grouped nested CV를 수행한다. inner CV는 하이퍼파라미터를 고르고 outer CV는 그 선택 과정까지 포함한 성능을 잰다.
4. 평균 CV MAPE가 낮은 순으로 모델을 비교한다. 점수가 비슷하면 fold 사이의 편차와 모델의 단순성을 함께 본다.
5. 선택된 모델로 피처 ablation을 진행한다. 특정 피처를 빼거나 ΔQ(V) 대표 통계를 다른 값으로 바꾸어 어떤 조합이 가장 안정적인지 비교한다.
6. 선택을 마친 구성을 Batch 1 hold-out에 적용해 보지 않은 protocol으로의 일반화 정도를 확인한다.
7. 최종 모델은 Batch 1 전체로 다시 학습한다. 그런 다음에야 Batch 2에서 재튜닝 없이 평가한다.
8. Batch 3 점수는 최종 결론을 바꾸기 위한 선택 기준이 아니라 다른 배치에서의 경향을 보는 보조 결과로 남긴다.

Dummy를 제외한 모델은 원본 `cycle_life`와 `log(cycle_life)` 두 가지 target 형태를 비교한다. 결측값 대치, 표준화, target 변환, 하이퍼파라미터 탐색은 모두 학습 fold 안에서 수행한다.

| 항목        | 선택값                                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| 모델        | ElasticNet                                                                                                                 |
| Target 변환 | `log(cycle_life)`                                                                                                          |
| `alpha`     | `0.01`                                                                                                                     |
| `l1_ratio`  | `0.9`                                                                                                                      |
| 피처 세트   | `dq_std_substitution`                                                                                                      |
| 최종 피처   | `dq_std`, `dq_kurtosis`, `initial_QDischarge`, `IR_mean_100`, `temperature_slope_100`, `chargetime_mean_100`, `max_c_rate` |

## 성능 결과

| 평가 구간                        | MAPE (%) | 이 점수가 나타내는 것                                               |
| -------------------------------- | -------: | ------------------------------------------------------------------- |
| Train: Batch 1 nested CV         |     7.62 | development 구간에서 protocol을 그룹으로 묶은 nested CV 평균이다.   |
| Valid: Batch 1 protocol hold-out |     7.96 | development에 없던 충전 protocol에 대한 내부 검증 결과다.           |
| Test: Batch 2 external batch     |    37.46 | Batch 1로 최종 학습한 뒤 Batch 2에서 재튜닝하지 않고 측정한 오차다. |
| Train → Valid gap                |  +0.35%p | Batch 1 안에서는 과적합 징후가 크지 않다.                           |
| Valid → Test gap                 | +29.50%p | 새로운 배치로 건너갈 때 성능이 크게 떨어졌다.                       |

Batch 3의 보조 평가 MAPE는 14.50%였다. Batch 2의 추가 지표는 MAE 194.24사이클, RMSE 212.72사이클, R² 0.059였다.

MAPE는 실제 수명을 기준으로 예측이 평균 몇 % 정도 어긋났는지를 보여준다. 예를 들어 Valid MAPE 7.96%는 hold-out 셀의 절대 백분율 오차가 평균 7.96%였다는 뜻이다. 실제 수명이 200사이클인 셀에 이 비율을 단순 적용하면 약 ±16사이클 규모지만, 이것은 개별 예측의 보장 범위나 신뢰구간이 아니다. 셀마다 오차가 다르고 Batch 2처럼 데이터 분포가 바뀌면 평균 오차도 크게 달라진다.

MAPE 하나만으로는 오차가 몇 사이클인지, 모델이 지속적으로 높게 예측하는지를 알 수 없다. 따라서 `supplementary_metrics.csv`와 `error_analysis_summary.csv`에 MAE, RMSE, R², signed error, 과대예측 비율을 함께 저장했다.

### 결과 파일 확인 순서

`results/` 루트에는 핵심 산출물이 있고 `results/tables/`에는 비교와 오차 분석에 필요한 세부 표가 있다. 각 컬럼의 뜻과 활용 예시는 `results/README.md`에 더 자세히 적었다.

1. `model_performance.csv`에서 Train, Valid, Test MAPE를 비교한다.
2. `model_comparison.csv`로 왜 ElasticNet이 선택됐는지 확인한다.
3. `feature_ablation.csv`에서 `dq_std_substitution`이 다른 피처 조합보다 낮은 CV MAPE를 낸 과정을 본다.
4. `predictions_and_errors.csv`로 오차가 큰 셀을 찾는다.
5. `batch_distribution_shift.csv`에서 Batch 1과 Batch 2의 target·피처 분포를 비교한다.

| 파일                                            | 파일을 열어보는 이유                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------- |
| `results/model_performance.csv`                 | Train–Valid–Test MAPE와 각 구간 사이의 성능 gap을 볼 수 있다.                   |
| `results/feature_table.csv`                     | 모델이 실제로 받은 target, protocol, 초기 피처를 셀 단위로 확인한다.            |
| `results/final_model.joblib`                    | 결측값 대치와 표준화까지 포함한 최종 scikit-learn 모델이다.                     |
| `results/tables/model_comparison.csv`           | 모델·target 변환·기본 피처 조합별 CV와 hold-out 점수가 들어 있다.               |
| `results/tables/feature_ablation.csv`           | 피처를 빼거나 대체했을 때 성능이 얼마나 바뀌는지 보여준다.                      |
| `results/tables/split_manifest.csv`             | Batch 1의 각 셀이 development와 hold-out 중 어디에 배정됐는지 기록한다.         |
| `results/tables/predictions_and_errors.csv`     | 셀별 실제 수명, 예측 수명, signed error, APE를 포함한다.                        |
| `results/tables/error_analysis_summary.csv`     | 평가 구간별 MAPE, 중앙 APE, 평균 편향, 과대예측 비율을 요약한다.                |
| `results/tables/batch2_error_by_life_group.csv` | Batch 2의 단수명·중수명·장수명 구간별로 예측 실패가 어디에 집중되는지 보여준다. |
| `results/tables/batch_distribution_shift.csv`   | Batch 1/2/3의 target과 최종 피처 평균·중앙값을 나란히 비교한다.                 |
| `results/tables/supplementary_metrics.csv`      | MAPE 외에 MAE, RMSE, R²를 함께 보여준다.                                        |
| `results/tables/run_metadata.json`              | seed, 선택 모델, 피처, 하이퍼파라미터, 표본 수를 남긴 실행 기록이다.            |

`results/figures/`에는 cycle life 분포, Batch별 수명 비교, 방전 용량 열화, ΔQ(V), knee point, C-rate, 피처 상관관계 그림이 있다. 노트북 설정을 바꾸면 원본 데이터에서 다시 생성할 수 있다. 노트북은 분석 과정을 재현하는 용도이고, `results/reports/`의 보고서는 결과와 판단 근거를 빠르게 확인하는 용도다.

## 오류 분석

Batch 1 nested CV와 protocol hold-out의 MAPE는 각각 7.62%와 7.96%로 가깝다. 이 결과만 보면 Batch 1 안에서 모델이 특정 표본을 과도하게 외웠다고 보기는 어렵다. 그런데 Batch 2에서는 MAPE가 37.46%로 뛰었다. 같은 모델이 새로운 배치에서 제대로 일반화되지 않았다는 뜻이다.

Batch 2 예측의 94.87%가 실제보다 높았고 평균 signed error는 +179.1사이클이었다. 특히 500사이클 미만인 Short 그룹 28개는 모두 과대예측됐으며 이 그룹의 MAPE는 40.39%였다.

실제 셀 단위 예시는 다음과 같다.

| 실제 수명 |   예측 수명 | 차이 | 절대 백분율 오차 |
| --------: | ----------: | ---: | ---------------: |
| 392사이클 |   548사이클 | +156 |            39.7% |
| 393사이클 |   691사이클 | +298 |            75.8% |
| 396사이클 |   609사이클 | +213 |            53.7% |
| 499사이클 |   701사이클 | +202 |            40.5% |
| 997사이클 | 1,129사이클 | +132 |            13.3% |

### 모델이 가장 크게 틀린 셀의 공통점

- Batch 1 학습 데이터에는 없던 500사이클 미만의 단수명 셀이 많다.
- Batch 2에서는 전체의 71.79%가 500사이클 미만이다.
- Batch 2의 충전 protocol 대부분은 Batch 1에서 보지 않은 protocol이다.
- target뿐 아니라 온도 기울기, 충전시간, ΔQ(V) 통계의 분포도 Batch마다 다르다.
- 오차가 큰 셀은 대체로 모델이 실제보다 수명을 길게 예측한 경우다.

### 원인 가설 및 개선 방향

주요 원인은 Batch 1과 Batch 2 사이의 분포 차이로 보인다. Batch 1에는 500사이클 미만 셀이 하나도 없기 때문에 모델은 매우 짧은 수명 구간의 패턴을 학습하지 못했다. 여기에 새로운 충전 protocol, 제조 lot와 셀 구조 차이가 함께 작용했을 가능성이 있다.

개선하려면 여러 제조 lot와 운전 조건에서 단수명 셀을 추가로 확보하고 예측값과 함께 불확실성 구간을 제공해야 한다. 새로운 입력이 학습 분포에서 벗어났는지 탐지하는 OOD 검사, 배치별 calibration, protocol 정보를 고려한 domain adaptation 또는 계층형 모델도 필요하다. 운영 중에는 센서 품질을 점검하고 시간에 따라 데이터를 추가해 주기적으로 재학습해야 한다.

## ESS 도메인 해석

### 실제 BESS에서 활용할 수 있는 의사결정

이 모델은 초기 100사이클만으로 상대적으로 수명이 짧을 가능성이 큰 셀을 선별하는 보조 도구로 활용할 수 있다.

- 입고 또는 초기 운전 단계에서 추가 진단이 필요한 셀을 우선 선별한다.
- 예상 수명에 따라 보증 등급이나 사용 목적을 구분하는 참고 지표로 쓴다.
- 위험도가 높은 셀에는 운전 강도를 낮추거나 모니터링 주기를 짧게 설정한다.
- 예비 셀 수량, 점검 일정, 교체 시기와 유지보수 예산을 계획하는 데 활용한다.
- 여러 신호가 함께 악화된 셀을 정밀 검사 대상으로 보내 운영자의 판단을 보조한다.

### 한계와 실배포 전에 필요한 사항

현재 모델은 Batch 1 내부에서는 약 8% MAPE를 보였지만 Batch 2에서는 37.46%로 성능이 크게 떨어졌고, 단수명 셀을 주로 과대예측했다. 따라서 안전 한계, 보증 승인, 교체 시점을 이 모델 하나로 결정해서는 안 된다.

실제 배포 전에는 다음 조건을 보완해야 한다.

- 다양한 제조사, 셀 화학계, 제조 lot, 온도와 운전 profile을 포함한 학습 데이터
- 단수명과 이상 셀을 충분히 포함한 표본 구성
- 점 예측뿐 아니라 예측 구간과 신뢰도 제공
- 학습 범위를 벗어난 셀을 감지하고 사람의 검토로 넘기는 절차
- 현장 센서의 결측, 편향, 시간 동기화 문제를 확인하는 데이터 품질 관리
- 새로운 운영 데이터를 반영하는 성능 모니터링과 정기 재학습 체계
- 실제 BESS의 안전 진단, 열관리, 전기적 보호 로직과 결합한 다중 판단 구조

현재 결과는 초기 신호로 상대적인 위험을 선별하거나 추가 진단 대상을 고르는 데는 의미가 있다. 그러나 새로운 배치에서 수명을 체계적으로 높게 예측할 수 있으므로 외부 배치 검증과 불확실성 관리가 실사용의 필수 조건이다.

## 재현성과 데이터 누수 방지

분석 결과를 우연에 맡기지 않고 평가 데이터의 정보가 학습에 섞이지 않도록 다음 원칙을 적용했다.

- 무작위성이 들어가는 분할과 모델에는 `random_state=42`를 사용한다.
- 선택된 모델, 피처, 하이퍼파라미터, 표본 수는 `results/tables/run_metadata.json`에 남긴다.
- 모델 입력은 cycle 2~100 정보로만 만든다.
- cycle 1의 일괄 0값은 삭제하지 않고 원본에 남겨 두되 초기 피처 집계에서는 제외한다.
- knee point, 마지막 100사이클 기울기, 최종 수명을 분모로 쓰는 비율은 EDA에만 남긴다.
- 결측값 대치, 표준화, target 변환, 하이퍼파라미터 탐색은 각 training fold 안에서 다시 적합한다.
- development와 hold-out은 `policy_readable`을 그룹으로 나눈다. 같은 충전 protocol이 두 구간에 동시에 들어가지 않는다.
- Batch 2는 모델, 피처, 하이퍼파라미터를 고르는 데 사용하지 않는다. 선택이 모두 끝난 후에 한 번 평가한다.

새 환경에서 원본 데이터부터 모든 결과를 다시 만들려면 아래 명령을 순서대로 실행한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py
python src/preprocess.py
python src/train.py --rebuild-features
jupyter lab
```

## 참고문헌

- Severson, K. A., Attia, P. M., et al. (2019). _Data-driven prediction of battery cycle life before capacity degradation_. Nature Energy, 4, 383–391. <https://doi.org/10.1038/s41560-019-0356-8>
- MIT-Stanford Battery Dataset: <https://data.matr.io/1/>
- 원 데이터 구조와 공개 로더: <https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation>
