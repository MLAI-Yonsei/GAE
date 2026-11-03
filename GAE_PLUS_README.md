# GAE+ 사용법 (브랜치 `gae+`)

GAE+는 GAE의 변화량을 **얼마나 반영할지**(η₁, η₂)를 정하는 방법입니다.
목표는 "높은 확률(1 − δ)로 원래 decoder보다 나빠지지 않는 것"입니다.

```
A = [W, b]                                    # decoder와 bias, shape [d, k+1]
A₊ = A0 + η₁ (A1 − A0) + η₂ (A2 − A0),   η₁ ≥ 0, η₂ ≥ 0, η₁ + η₂ ≤ 1
   = (1 − η₁ − η₂)·A0 + η₁·A1 + η₂·A2       # 세 decoder를 섞는 비율
A0 = 원래 decoder,  A1 = Step 1 이후,  A2 = Step 2 이후 = GAE (λ_id = 0)
```

| η = (η₁, η₂) | 섞는 비율 A0 : A1 : A2 | 의미 |
|---|---|---|
| (0, 0) | 1 : 0 : 0 | 원래 decoder 그대로 |
| (1, 0) | 0 : 1 : 0 | Step 1만 적용 |
| (0, 1) | 0 : 0 : 1 | GAE 그대로 |
| (0, 0.3) | 0.7 : 0 : 0.3 | 원래 decoder에서 GAE 쪽으로 30%만 이동 |
| (0.3, 0.2) | 0.5 : 0.3 : 0.2 | 세 개를 모두 섞음 |

알고리즘은 [`gae_plus.py`](gae_plus.py)에 있고, 진입점은 `gae_plus_select`입니다.

---

## 1. 기본값으로 실행하기

```python
from gae_plus import gae_plus_select
from gae_plus_probe import F64, build_candidates, encode

# OOD 표본을 둘로 나눕니다: idx_a(절반)는 A1·A2를 만드는 데, idx_b(나머지 절반)는 η를 고르는 데 씁니다.
A0, A1, A2, _ = build_candidates(explainer, model, layer, id_in, id_out,
                                 ood_in[idx_a], ood_out[idx_a], r=32)   # idx_a: Step 1, 2
Z, _ = encode(explainer, ood_in[idx_b])                                 # idx_b: feature [n, k]
Y = ood_out[idx_b].to(**F64)                                            # idx_b: target  [n, d]
eta1, eta2, A_plus = gae_plus_select(Z, Y, A0, A1, A2)                  # η 결정
W_plus, b_plus = A_plus[:, :-1], A_plus[:, -1]
```

기본값은 `mode="elementwise"`, `bound="best"`, `basis="gae"`, `delta=0.05`입니다.

---

## 2. 옵션 정리

### 2-1. `gae_plus_select`의 인자

```python
gae_plus_select(Z, Y, A0, A1, A2, mode="elementwise", bound="best", delta=0.05, basis="gae")
```

| 인자 | 하는 일 | 고를 수 있는 값 | **기본값 (현재)** | **원래 계획서** |
|---|---|---|---|---|
| `Z` | 각 OOD 표본의 feature 값 (원래 encoder가 낸 값), `[n, k]` float64 | (필수) | — | — |
| `Y` | decoder가 맞춰야 하는 정답 activation, `[n, d]` float64 | (필수) | — | — |
| `A0, A1, A2` | 서로 비교할 decoder 세 개 (원래 / Step 1 이후 / GAE). 각각 decoder 가중치 W `[d, k]` 옆에 bias b를 한 열로 붙인 행렬 `[d, k+1]`, float64 | (필수) | — | — |
| `mode` | 상한을 어디에 몇 개 붙일지 | `"elementwise"`, `"matrix"`, `"nosplit"` | `"elementwise"` | `"nosplit"` |
| `bound` | 상한에 쓸 부등식 | `"best"`, `"bernstein"`, `"chebyshev"` | `"best"` | `"chebyshev"` |
| `delta` | 전체 실패 확률. 작을수록 상한이 커지고 η가 0 쪽으로 감 | 0 < δ < 1 | `0.05` | `0.05` |
| `basis` | 상한을 붙일 계수 기준 (`mode="elementwise"`일 때만 적용) | `"gae"`, `"step"` | `"gae"` | 해당 없음 |

```python
# 기본값
gae_plus_select(Z, Y, A0, A1, A2)                # Z, Y는 idx_b, A0~A2는 idx_a로 생성 (§1)

# 원래 계획서 (표본 분할 없음)
gae_plus_select(Z, Y, A0, A1, A2, mode="nosplit", bound="chebyshev")   # Z, Y, A0~A2 모두 OOD 표본 전체
```

A1·A2까지 원래 계획서대로(논문 Eqs. 5–6) 만들려면, 스크립트 맨 위 설정을 아래처럼 바꾸세요.
GAE+ 스크립트는 코드 기본값(`centered`, `1`)을 씁니다.

```bash
SUBSPACE_MATRIX=second_moment   # 논문 Eq. 5 (2차 모멘트 행렬)
STEP1_MATCH_NORMS=0             # Step 1 회전 후 열 크기 맞춤 끔
```

`Z, Y, A0, A1, A2`는 어떤 mode든 똑같이 넣습니다. 달라지는 것은 **어떤 표본을 넣느냐**뿐입니다.
`"elementwise"`, `"matrix"`는 A1·A2를 만들 때 쓰지 않은 표본을 Z, Y로 넣어야 하고 (§1처럼 반으로 나눔), `"nosplit"`은 같은 표본을 넣어도 됩니다.

돌려주는 값은 `(eta1, eta2, A_plus)`입니다. `A_plus[:, :-1]`이 decoder, `A_plus[:, -1]`이 bias입니다.

### 2-2. 각 값의 뜻

η를 고르는 방식은 모든 값에서 같습니다.
η를 고르는 표본에서 "A₊로 바꾸면 재구성 오차가 얼마나 변하는지"를 계산하고, 표본 수가 유한해서 생기는 추정 오차만큼 **벌점**을 더한 뒤, 합이 가장 작은 η를 고릅니다.
추정 오차의 최댓값을 **상한**이라고 부르고, 아래 옵션들은 이 상한을 어떻게 구할지 정합니다.

오차 변화는 `yᵀHy − 2gᵀy`로 계산합니다.
H(2×2)와 g(숫자 2개)는 decoder 변화 방향 두 개(D₁, D₂)에 대해 표본마다 계산한 값의 평균이고, y는 그 방향의 계수입니다.

#### `mode`: 상한을 어디에 몇 개 붙일지

| 값 | 뜻 | 상한 개수 | 표본 분할 |
|---|---|---|---|
| `"elementwise"` | H의 숫자 3개(H₁₁, H₁₂, H₂₂)와 g의 숫자 2개(g₁, g₂)에 상한을 **하나씩 따로** 붙입니다. 실패 확률 δ를 5개로 나눠 하나당 δ/5씩 씁니다 | 5개 | 필요 |
| `"matrix"` | H 전체에 하나(‖Ĥ − H‖), g 전체에 하나(‖ĝ − g‖) 붙입니다. 하나당 δ/2씩 씁니다. 계수는 항상 `basis="step"` 기준입니다 | 2개 | 필요 |
| `"nosplit"` | 원래 계획서 방식입니다. decoder와 상관없는 데이터 통계 M = mean z̄z̄ᵀ, T = mean e₀z̄ᵀ에 상한(a_N, b_N)을 붙이므로, decoder가 어떻게 바뀌든 성립합니다. 벌점은 a_N‖A₊−A0‖²_F + 2b_N‖A₊−A0‖_F | 2개 | 필요 없음 |

- `"elementwise"`와 `"matrix"`의 차이: `"matrix"`는 2×2 행렬에서 가장 큰 숫자에 맞춰 모든 방향에 같은 크기의 벌점을 줍니다. `"elementwise"`는 작은 숫자에는 작은 벌점을 줍니다.
- `"nosplit"`은 표본을 나누지 않아도 되지만, GAE의 decoder 변화는 ‖A₊ − A0‖_F가 매우 커서 벌점도 매우 커집니다. HaluEval 실험에서는 η = (0, 0)만 나왔습니다.
- `"elementwise"`, `"matrix"`의 상한은 이미 만들어진 A1, A2에 대해서만 성립하므로, A1·A2를 만들 때 쓰지 않은 표본을 넣어야 합니다.

#### `bound`: 상한을 계산하는 부등식

아래에서 s, L은 표본에서 관측한 최댓값(범위)이고, δ'는 상한 하나에 배정된 실패 확률(δ/5 또는 δ/2)입니다.

| 값 | 상한(반지름) | 필요한 값 |
|---|---|---|
| `"chebyshev"` | s / √(n·δ') | 범위 s |
| `"bernstein"` | Lℓ/3 + √((Lℓ/3)² + 2vℓ), ℓ = log(차원 / δ') | 범위 L, 분산 v (범위로부터 계산) |
| `"best"` | 상한마다 `"chebyshev"`와 `"bernstein"` 중 작은 값 | — |

`"bernstein"`이 mode마다 실제로 쓰는 정리:

| mode | H 쪽 상한 (또는 a_N) | g 쪽 상한 (또는 b_N) |
|---|---|---|
| `"elementwise"` | 숫자 하나에 대한 Bernstein | 숫자 하나에 대한 Bernstein |
| `"matrix"` | Tropp 행렬 Bernstein (2015, Thm 1.6.2), 2×2 | Pinelis 차원 무관 Bernstein (1994, Thm 3.4) |
| `"nosplit"` | Tropp 행렬 Bernstein, M 크기 그대로 | Pinelis 차원 무관 Bernstein |

표본이 많으면 보통 `"bernstein"`이 더 작고, 표본이 아주 적으면 `"chebyshev"`가 더 작을 수 있습니다. `"best"`는 항상 둘 중 작은 쪽을 씁니다.

#### `delta`: 전체 실패 확률

- δ = 0.05이면 "95% 확률로, 진짜 오차 변화가 계산한 오차 변화 + 벌점보다 크지 않다"는 뜻입니다.
- δ를 줄이면 상한이 커지고, η는 0(원래 decoder) 쪽으로 갑니다.
- 상한이 여러 개면 δ를 나눠 씁니다. 예: δ = 0.05, `"elementwise"`이면 상한 하나당 0.01입니다.
- 범위 s, L을 관측된 최댓값으로 정하므로, 이 확률 보장은 엄밀하지 않습니다.

#### `basis`: 상한을 붙일 계수 기준 (`"elementwise"`에서만 사용)

같은 A₊를 두 가지 방향 쌍으로 쓸 수 있습니다. 예를 들어 η = (0.3, 0.2)이면:

| 값 | 방향 쌍 | 계수 | 예시 |
|---|---|---|---|
| `"gae"` | (A1 − A0, A2 − A0) | (η₁, η₂) | A₊ = A0 + 0.3(A1 − A0) + 0.2(A2 − A0) |
| `"step"` | (A1 − A0, A2 − A1) | (η₁ + η₂, η₂) | A₊ = A0 + 0.5(A1 − A0) + 0.2(A2 − A1) |

- 두 식의 A₊는 같고, 함수가 돌려주는 η도 같습니다. 달라지는 것은 상한을 어느 계수에 붙이느냐뿐입니다.
- 기본값이 `"gae"`인 이유: Step 1만 적용한 A1은 오차가 매우 큽니다. 그래서 `"step"` 기준에서는 H의 숫자가 763, −734, 743처럼 크고 서로 상쇄되어, 숫자마다 상한을 붙이면 벌점이 지나치게 커집니다.

## 3. 환경과 데이터 준비

스크립트마다 맨 위에 설정이 있고, 경로는 환경변수(`DATA_ROOT`, `CACHE_ROOT`)로 바꿀 수 있습니다.
각 스크립트는 환경을 직접 만들고(`uv venv --python 3.10 && uv pip sync requirements.lock`), 결과 폴더 `results/gae_plus/<run>/`에 `run_info.txt`, `uncommitted.diff`, 스크립트 사본을 남깁니다.

```bash
bash scripts/data/download_owt_subset.sh              # OpenWebText shard 0-1(학습), 79(ID 평가), revision 고정 + sha256 확인
bash scripts/train_id/train_gpt2_transcoder.sh [GPU]  # GPT-2 Small transcoder 학습, 논문 설정 (A6000에서 약 1시간)
bash scripts/gae_plus/build_edgar_prompts.sh          # Edgar prompt 생성, revision 고정
```

## 4. 실험 스크립트 (`scripts/gae_plus/`, 실행: `bash <스크립트> [GPU]`)

| 스크립트 | 하는 일 |
|---|---|
| `bounds_grid.sh` | 표본 분할 off/on × 상한 없음/Chebyshev/best (상한 2개), 그리고 elementwise (step/gae 기준) |
| `probe_bounds.sh` | 원래 계획서의 a_N, b_N (method A, B)과 그때 고른 η |
| `split_check.sh` | A0, A1, A2의 학습 표본 오차 vs 새 표본 오차 |
| `reproduce_gpt2_halueval.sh` | 논문 Table 2 재현 (HaluEval) |
| `reproduce_gpt2_edgar.sh` | 논문 Table 2 재현 (Edgar, 코드 기본값) |
| `reproduce_gpt2_edgar_paper.sh` | Edgar, 논문 설정 (`second_moment`, λ_id = 0) |
| `reproduce_gpt2_edgar_paper_nonorm.sh` | 위와 같고, Step 1의 열 크기 맞춤만 끔 |
