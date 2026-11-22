# GAE+ (v3) 안내

공저자용 문서입니다. 이 브랜치(`gae+v3`)가 무엇을 하는지, 어떤 옵션을 바꿀 수 있는지를 정리했습니다.

## 한눈에 보기

`gae+v3`는 `main`의 코드를 그대로 두고, 그 위에 두 가지만 얹은 브랜치입니다.

1. **Step 1 수축**: 표본이 적을 때 흔들리는 OOD 부분공간 추정을 안정화합니다.
2. **GAE+ 선택**: Fixed, Step 1, Step 2를 섞은 후보 6개를 만들고, utility score로 하나를 고릅니다.

이전 버전(`gae+v2`)에 있던 단계별 선별, 승급 규칙, preset, bootstrap은 모두 없앴습니다.

| 파일 | main 대비 변경 |
|---|---|
| `gae.py` | 옵션 세 개 추가 (Step 1 수축, Step 2 고정 bias, Step 2 λ 자동 결정). 모두 기본은 꺼짐 |
| `config.py` | `GAE_AC_LAMBDA_ID` 기본값을 0.05에서 0으로 |
| `gae_plus.py` | 새 파일. GAE+ 전체 (후보 생성, 선택, 평가) |
| `tests/test_gae_plus.py` | 새 파일. 자체 점검 |
| `scripts/gae_plus/` | 새 폴더. 실험 하나를 재현하는 bash 파일 |
| `run_experiment.py`, `ood_utils/` | 변경 없음 |

GAE 알고리즘과 평가 점수는 main의 함수를 그대로 불러 씁니다.
(`gae_torch`, `fit_affine_constrained_decoder`, `compute_feature_mask_scores` 등)

## 왜 필요한가

적응에 쓸 OOD 표본이 적으면(예: 128개) 두 가지 문제가 생깁니다.

- **OOD 부분공간 추정이 흔들립니다.** MedQuAD에서 128개로 추정한 부분공간은 3,431개로 추정한 것과 절반 정도만 겹쳤습니다.
- **적응된 decoder가 통하지 않는 입력이 늘어납니다.** Fixed는 평가 1,000개 중 770개가 유효한데, GAE를 128개로 적합하면 616개로 줄었습니다.

GAE+는 이 상황에서 별도 validation set 없이, 적응에 쓴 표본 S만으로 "얼마나 적응할지"를 정합니다.

## 핵심 세 가지

### 1. Step 1 수축

Step 1은 OOD activation의 covariance에서 상위 r개 방향을 뽑아 부분공간으로 씁니다.
수축을 켜면 이 covariance를 기준 행렬 쪽으로 섞은 뒤에 부분공간을 뽑습니다.

```
M = (1 − ρ) × (OOD 표본의 covariance) + ρ × (기준 행렬)
ρ = (OOD covariance 추정값의 흔들림) ÷ (OOD covariance와 기준 행렬 사이의 거리)
```

- ρ는 데이터에서 자동으로 정해집니다. 사람이 정하는 값이 없습니다.
- 표본이 적으면 ρ가 커지고, 많으면 0에 가까워져서 기존 GAE와 같아집니다. MedQuAD에서 N=64일 때 0.72, 128일 때 0.57, 2,048일 때 0.08이었습니다.
- 기준 행렬은 두 가지 중에서 고릅니다.

| 기준 | 내용 | 필요한 것 |
|---|---|---|
| `decoder` (기본) | 원래 decoder로 만든 행렬 (a × W₀W₀ᵀ) | 체크포인트만 |
| `id` | ID activation의 covariance | ID 데이터 일부 (예: OpenWebText 2,000개) |

Ledoit–Wolf 추정량에서 "섞는 비율을 구하는 식"을 빌려 온 것입니다. 기준 행렬이 다르고 부분공간만 뽑아 쓰기 때문에, 원래 추정량을 그대로 적용한 것은 아닙니다.

### 2. η 후보와 utility score

decoder 세 개를 섞어 후보를 만듭니다.

| 기호 | 뜻 |
|---|---|
| A₀ | Fixed (원래 decoder) |
| A₁ | Step 1 결과 (정렬된 dictionary, bias는 원래 값) |
| A₂ | Step 2 결과 (재적합) |

```
A(η) = (1 − η₁ − η₂)·A₀ + η₁·A₁ + η₂·A₂
```

후보는 여섯 개입니다.

| (η₁, η₂) | 뜻 |
|---|---|
| (0, 0) | Fixed |
| (0, 0.5) | Fixed와 GAE의 중간 |
| (0, 1) | GAE 전체 적용 |
| (0.5, 0) | Fixed와 Step 1의 중간 |
| (0.5, 0.5) | Step 1과 GAE의 중간 |
| (1, 0) | Step 1만 |

후보마다 S에서 deletion 점수를 재고, utility score가 가장 높은 것을 고릅니다. utility score는 네 가지로 이루어집니다.

- **입력 단위 fallback**: 후보의 margin이 0.1 이하인 입력은 Fixed 결과로 바꿉니다. margin은 feature를 전부 켠 복원과 전부 끈 복원의 target logit 차이입니다.
- **분모 통일**: 모든 후보를 Fixed의 margin으로 나눕니다. 자기 margin으로 나누면 margin이 줄어든 후보가 유리해 보이는 왜곡이 생깁니다.
- **clip**: 입력별 값을 [−1, 1]로 자른 점수와 자르지 않은 점수를 둘 다 구합니다.
- **Fixed 대비 이득**: 두 점수 각각에서 Fixed 대비 상대 이득을 구하고, 더 나쁜 쪽을 그 후보의 utility로 삼습니다. Fixed의 utility는 0입니다.

어떤 후보도 Fixed를 넘지 못하면 Fixed가 선택됩니다. 동점이면 덜 바꾸는 쪽을 고릅니다.

### 3. 평가 때의 gate

고른 decoder로 평가할 때도 같은 fallback을 적용할 수 있습니다(`--gate on`).
결과 파일에는 gate를 적용한 값과 적용하지 않은 값이 항상 둘 다 저장됩니다.

## 실행 방법

```bash
python gae_plus.py \
  --model gpt2 --explainer transcoder \
  --checkpoint <ID로 학습한 explainer 체크포인트> \
  --adaptation <적응용 프롬프트 파일> --evaluation <평가용 프롬프트 파일> \
  --n-fit 128 --rank 32 \
  --output results/<이름>
```

입력 파일은 `.jsonl` 또는 `.json`이고, 한 줄(한 항목)에 토큰 id 목록이 들어 있으면 됩니다.

```json
{"record_id": "medquad:...", "tokens": [50256, 1858, 423, ...]}
```

`tokens` 대신 main의 `context_ids`도 읽습니다. `text`만 있으면 BOS를 붙여 128토큰까지 토큰화합니다.
적응용 파일에서는 앞에서부터 `--n-fit`개를 S로 씁니다. 평가용 파일은 S와 겹치지 않아야 합니다.

## 옵션

### GAE 옵션 (main과 같은 뜻)

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--rank` | OOD 부분공간의 rank | 64 |
| `--lam-gae` | Step 2에서 Step 1 결과 쪽으로 당기는 세기 | 0.2 |
| `--lam-geom` | Step 2에서 부분공간 밖 성분을 줄이는 세기 | 0.1 |
| `--lam-id` | Step 2에서 원래 decoder 쪽으로 당기는 세기 | 0 |

λ는 main 코드와 같이 표본 합 기준입니다. 논문 수식의 평균 기준 값과는 N배 차이가 납니다.
rank와 λ는 utility score로 고르지 않습니다. 여러 값을 보려면 bash 파일에서 반복문으로 돌립니다.

### 새로 추가된 옵션

| 옵션 | 값 | 기본값 | 설명 |
|---|---|---|---|
| `--shrink` | `decoder`, `id`, `none` | `decoder` | Step 1 수축의 기준 행렬. `none`이면 수축하지 않음 |
| `--id-data` | 파일 경로 | 없음 | `--shrink id`일 때 쓰는 ID 프롬프트 파일 |
| `--lam-auto` | 켜기/끄기 | 꺼짐 | Step 2의 `--lam-gae`를 Ledoit–Wolf 식으로 자동 결정 |
| `--bias` | `free`, `fixed`, `both` | `free` | Step 2의 bias. `free`는 새로 적합(main 방식), `fixed`는 원래 값 유지, `both`는 두 계열을 모두 후보에 넣음 |
| `--gate` | `on`, `off` | `on` | 평가 점수를 보고할 때 fallback을 적용할지 |
| `--n-fit` | 정수 | 128 | S의 크기 |
| `--n-eval` | 정수 | 전체 | 평가에 쓸 프롬프트 수 |

## 출력

`--output` 폴더에 두 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `result.json` | 설정, 수축 강도 ρ, 후보별 utility, 선택된 η, 평가 점수 |
| `rows.jsonl` | 후보별, 입력별 원본 점수 (s0, s_empty, s_del). 다른 규칙으로 다시 집계할 때 씁니다 |

`result.json`의 평가 부분은 이렇게 생겼습니다.

```json
"evaluation": {
  "fixed":    {"n_aopc": 0.8742, "n_comp": 1.0720, "n_valid": 770, "n_eval": 1000},
  "gae":      {"eta": [0.0, 1.0], "with_gate": {...}, "without_gate": {...}, "fallback": 384},
  "gae_plus": {"eta": [0.0, 1.0], "with_gate": {...}, "without_gate": {...}, "fallback": 384}
}
```

- `gae`는 η = (0, 1), 즉 GAE를 전체 적용한 decoder입니다.
- `gae_plus`는 utility score가 고른 decoder입니다.
- nAOPC와 nComp는 main의 평가 함수로 계산하고, 유효 입력(margin > 0.1)에서만 평균을 냅니다.

## 확인된 결과

GPT-2 Small Transcoder × MedQuAD, N=128, rank 32, λ 0.2. 평가 1,000개. 값은 nAOPC / nComp입니다.
재현: `bash scripts/gae_plus/medquad_gpt2_tc.sh`

| 설정 | ρ | 선택된 η | gate 없이 (유효 입력) | gate 적용 (유효 입력) |
|---|---|---|---|---|
| Fixed | – | – | 0.8742 / 1.0720 (770) | – |
| 수축 없음 | – | (0, 1) | 0.8692 / 1.1363 (616) | 0.8870 / 1.1868 (830) |
| decoder 기준 수축 | 0.527 | (0, 1) | 0.8843 / 1.1458 (621) | 0.9004 / 1.1995 (831) |
| ID 기준 수축 | 0.574 | (0, 1) | 0.8905 / 1.1762 (634) | 0.9022 / 1.2218 (831) |

- Fixed는 논문 표의 값(0.8742 / 1.0720)과 같습니다.
- 세 설정 모두 utility score가 GAE 전체 적용을 골랐습니다. Step 1만 쓰는 후보는 utility가 음수라 걸러졌습니다.
- 수축을 넣으면 gate 없이도 Fixed의 nAOPC를 넘습니다.

## 알아 둘 점

- **검증한 조건이 하나뿐입니다.** 위 표는 한 칸, 표본 묶음 하나의 결과입니다. 다른 데이터셋과 다른 표본 묶음에서는 아직 확인하지 않았습니다.
- **`--lam-id` 기본값이 main과 다릅니다.** main은 0.05(원래 decoder를 20% 섞음)이고, 여기서는 0입니다. 원래 decoder와 섞는 일은 η가 맡습니다. `run_experiment.py`의 GAE baseline도 이 기본값을 따르므로, main과 같은 결과가 필요하면 `GAE_AC_LAMBDA_ID=0.05`로 실행하세요.
- **gate는 GAE+에만 있는 장치입니다.** 다른 baseline과 비교할 때는 gate를 적용하지 않은 값도 같이 보는 것이 안전합니다.
- **`--lam-auto`는 지금 구조에서 점수를 거의 바꾸지 않았습니다.** MedQuAD에서 λ가 0.2일 때와 자동값(약 1.0)일 때의 차이가 0.001 이하였습니다.
- **`--shrink id`에는 ID 데이터가 필요합니다.** explainer를 학습한 데이터의 일부를 모델에 한 번 통과시켜 covariance를 구합니다.

## 자체 점검

```bash
python tests/test_gae_plus.py
```

수축을 끄면 main과 같은 결과가 나오는지, 수축 강도가 0과 1 사이인지, utility와 fallback이 의도대로 작동하는지 확인합니다.

## 코드 위치

| 내용 | 위치 |
|---|---|
| Step 1 수축 | `gae.py`의 `shrink_covariance`, `gae_torch(shrink=...)` |
| Step 2 λ 자동 결정 | `gae.py`의 `ledoit_wolf_ridge`, `fit_affine_constrained_decoder(lam_auto=...)` |
| Step 2 고정 bias | `gae.py`의 `fit_affine_constrained_decoder(fixed_bias=...)` |
| η 후보 | `gae_plus.py`의 `ETA_GRID`, `mix` |
| fallback, utility, 선택 | `gae_plus.py`의 `gate`, `surrogate`, `utility`, `select` |
| 평가 집계 | `gae_plus.py`의 `aggregate` (main의 `compute_aopc_and_normalized`, `compute_comp_suff` 사용) |
