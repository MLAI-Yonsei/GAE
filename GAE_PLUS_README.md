# GAE+ 사용법 (브랜치 `gae+`)

GAE+는 GAE의 변화량을 **얼마나 반영할지**(η₁, η₂)를 정하는 방법입니다.
목표는 "높은 확률(1 − δ)로 원래 decoder보다 나빠지지 않는 것"입니다.

```
A = [W, b]                                    # decoder와 bias, shape [d, k+1]
A₊ = A0 + η₁ (A1 − A0) + η₂ (A2 − A1),   0 ≤ η₂ ≤ η₁ ≤ 1
A0 = 원래 decoder,  A1 = Step 1 이후,  A2 = Step 2 이후 (λ_id = 0)
```

- η = (0, 0): 원래 decoder를 그대로 사용
- η = (1, 1): GAE를 그대로 사용
- η = (0.3, 0.3): 원래 decoder에서 GAE 쪽으로 30%만 이동

알고리즘은 [`gae_plus.py`](gae_plus.py)에 있고, 진입점은 `gae_plus_select`입니다.

---

## 1. 기본값으로 실행하기

```python
from gae_plus import gae_plus_select
from gae_plus_probe import F64, build_candidates, encode

# OOD 표본을 절반 A(idx_a), 절반 B(idx_b)로 나눕니다.
A0, A1, A2, _ = build_candidates(explainer, model, layer, id_in, id_out,
                                 ood_in[idx_a], ood_out[idx_a], r=32)   # 절반 A: Step 1, 2
Z, _ = encode(explainer, ood_in[idx_b])                                 # 절반 B: feature [n, k]
Y = ood_out[idx_b].to(**F64)                                            # 절반 B: target  [n, d]
eta1, eta2, A_plus = gae_plus_select(Z, Y, A0, A1, A2)                  # η 결정
W_plus, b_plus = A_plus[:, :-1], A_plus[:, -1]
```

기본값은 `mode="elementwise"`, `bound="best"`, `basis="gae"`, `delta=0.05`입니다.

---

## 2. 변경점 한눈에 보기

아래 4가지가 원래 계획서에서 바뀐 점입니다. 모두 옵션으로 원래대로 되돌릴 수 있습니다.

| # | 항목 | 기존 (원래 계획서) | 변경 (현재 기본값) | 기존으로 되돌리는 법 |
|---|---|---|---|---|
| 1 | 표본 사용 | 같은 표본으로 Step 1·2를 만들고 η도 결정 | 절반 A로 Step 1·2, 나머지 절반 B로 η 결정 | 원래 계획서 경로 사용 (§3-1) |
| 2 | 상한 부등식 | Chebyshev | Chebyshev와 Bernstein 중 작은 값 | `bound="chebyshev"` |
| 3 | 계수 기준 | Step 1, Step 2 각각의 변화량 `(A1−A0, A2−A1)` | Step 1, GAE 전체의 변화량 `(A1−A0, A2−A0)` | `basis="step"` |
| 4 | 상한 개수 | 2개: ‖Ĥ − H‖ 하나(a_N), ‖ĝ − g‖ 하나(b_N) | 5개: H의 숫자 3개 + g의 숫자 2개에 각각 하나씩 (δ/5씩) | `mode="matrix"` |
| + | η 묶음 (추가 옵션) | η₁, η₂ 따로 결정 | η₁ = η₂로 묶어서 하나만 결정 | `mode="tie"`로 켬 |

### 옵션끼리의 관계

- `basis`는 `mode="elementwise"`일 때만 적용됩니다.
- `mode="matrix"`는 항상 `(A1−A0, A2−A1)` 기준이므로, 4번을 기존으로 돌리면 3번도 기존이 됩니다.
- `mode="tie"`는 `A2−A0` 방향 하나만 쓰므로 `basis`와 무관합니다.
- 1번을 기존으로 돌리면 원래 계획서의 함수들을 사용하고, 여기서는 2번(부등식)만 고를 수 있습니다.

---

## 3. 항목별 설명과 코드

### 3-1. 표본 분할 (1번)

**왜 바꿨나:** 같은 표본으로 A2를 만들고 평가하면 A2가 실제보다 훨씬 좋아 보입니다.
HaluEval 실험에서 A2의 오차는 학습에 쓴 표본에서 0.024였지만, 새 표본에서는 26.12였습니다 (원래 decoder A0는 20.11).

**기존(원래 계획서)으로 실행하기:** 전체 표본으로 A0, A1, A2를 만들고, 같은 표본으로 η를 고릅니다.

```python
from gae_plus import apply_eta, compute_bounds, eta_objective_terms, select_eta

A0, A1, A2, _ = build_candidates(explainer, model, layer, id_in, id_out,
                                 ood_in, ood_out, r=32)                 # 전체 표본
Z, _ = encode(explainer, ood_in)
Y = ood_out.to(**F64)
H, g, K, _ = eta_objective_terms(Z, Y, A0, A1, A2)                      # K = <D_j, D_l>_F
E0 = Y - Z @ A0[:, :-1].T - A0[:, -1]                                   # 원래 decoder의 오차
a_N, b_N = compute_bounds(Z, E0, delta=0.05, method="chebyshev")        # 2번도 기존 (Chebyshev)
eta1, eta2, _ = select_eta(H, g, K, a_N, b_N)
A_plus = apply_eta(A0, A1, A2, eta1, eta2)
```

`compute_bounds`의 `method`로 고를 수 있는 값:

| method | a_N, b_N을 구하는 부등식 |
|---|---|
| `"chebyshev"` | Chebyshev, c = √(2 / (Nδ)) — **원래 계획서** |
| `"A"` | Tropp 행렬 Bernstein (2015, Thm 1.6.2) |
| `"B"` | Wang–Ramdas 경험적 Bernstein (분산 기반) |
| `"bernstein"` | a_N: Tropp, b_N: Pinelis 차원 무관 Bernstein (1994, Thm 3.4) |
| `"best"` (기본값) | `"chebyshev"`와 `"bernstein"` 중 작은 값 |

명령줄에서는 `gae_plus_bounds_grid.py --split off`가 이 경로를 실행합니다.

### 3-2. 상한 부등식 (2번)

**왜 바꿨나:** 상황에 따라 더 작은(덜 보수적인) 상한이 나오는 부등식이 다르기 때문입니다.
`best`는 Chebyshev와 Bernstein으로 각각 상한을 구한 뒤 작은 값을 씁니다. 상한이 작을수록 η를 더 과감하게 고를 수 있습니다.

| 이름 | 반지름(상한) | 필요한 값 |
|---|---|---|
| Chebyshev | s / √(nδ) | 범위 s |
| Bernstein | t* = Lℓ/3 + √((Lℓ/3)² + 2vℓ),  ℓ = log(차원/δ) | 범위 L, 분산 v |

예시 (η 묶음, 표본 분할): Chebyshev는 a_N = 33.69, b_N = 5.64, best는 a_N = 17.55, b_N = 2.72로 약 절반입니다.

```python
gae_plus_select(Z, Y, A0, A1, A2, bound="chebyshev")   # 기존
gae_plus_select(Z, Y, A0, A1, A2, bound="bernstein")   # Bernstein만
gae_plus_select(Z, Y, A0, A1, A2)                      # best (기본값)
```

### 3-3. 계수 기준 (3번)

같은 A₊를 두 가지 방식으로 쓸 수 있습니다. 예를 들어 η = (0.5, 0.2)이면:

```
기존 (step): A₊ = A0 + 0.5 (A1 − A0) + 0.2 (A2 − A1)
변경 (gae) : A₊ = A0 + 0.3 (A1 − A0) + 0.2 (A2 − A0)     # 계수 = (η₁ − η₂, η₂)
```

두 식이 만드는 A₊는 같고, 함수가 돌려주는 η도 같은 정의(η₁, η₂)입니다.
달라지는 것은 **상한을 어느 기준으로 계산하느냐**뿐입니다.

**왜 바꿨나:** Step 1만 적용한 A1은 오차가 매우 큽니다(HaluEval 실험에서 원래 20에 비해 약 760).
그래서 step 기준에서는 H의 숫자들이 763, −734, 743처럼 크고, 서로 상쇄되어 실제 곡률은 39 정도밖에 되지 않습니다.
숫자마다 상한을 따로 붙이면 이 큰 숫자들에 상한이 붙어 지나치게 보수적이 됩니다. gae 기준에서는 이런 상쇄가 없습니다.

```python
gae_plus_select(Z, Y, A0, A1, A2, basis="step")   # 기존
gae_plus_select(Z, Y, A0, A1, A2)                 # basis="gae" (기본값)
```

### 3-4. 상한 개수 (4번)

| | 기존 `mode="matrix"` | 변경 `mode="elementwise"` |
|---|---|---|
| H (2×2) | ‖Ĥ − H‖ 전체에 상한 1개 (δ/2) | H₁₁, H₁₂, H₂₂ 각각 상한 1개 (δ/5씩) |
| g (2) | ‖ĝ − g‖ 전체에 상한 1개 (δ/2) | g₁, g₂ 각각 상한 1개 (δ/5씩) |

**왜 바꿨나:** 전체 행렬에 상한 하나를 붙이면 가장 큰 숫자에 맞춰 모든 방향이 똑같이 벌점을 받습니다.
숫자마다 따로 붙이면 작은 숫자에는 작은 상한이 붙습니다. δ를 5로 나눠도(Bonferroni) 전체 실패 확률은 δ 이하로 유지됩니다.

```python
gae_plus_select(Z, Y, A0, A1, A2, mode="matrix")   # 기존 (basis는 자동으로 step)
gae_plus_select(Z, Y, A0, A1, A2)                  # elementwise (기본값)
```

### 3-5. η 묶음 (추가 옵션)

η₁ = η₂ = t로 묶으면 `A₊ = A0 + t (A2 − A0)`입니다. 원래 decoder와 GAE 사이의 한 직선 위에서만 고르게 됩니다.

**왜 만들었나:** 위험한 A1 방향을 아예 쓰지 않고 한 숫자만 추정하므로 상한이 작아집니다.

```python
gae_plus_select(Z, Y, A0, A1, A2, mode="tie")
```

---

## 4. 조합별 예시 코드

| 원하는 설정 | 코드 |
|---|---|
| 현재 기본값 (1~4 모두 변경) | `gae_plus_select(Z, Y, A0, A1, A2)` |
| 2번만 기존 | `gae_plus_select(..., bound="chebyshev")` |
| 3번만 기존 | `gae_plus_select(..., basis="step")` |
| 3·4번 기존 | `gae_plus_select(..., mode="matrix")` |
| 2·3·4번 기존 (1번만 변경) | `gae_plus_select(..., mode="matrix", bound="chebyshev")` |
| 1~4번 모두 기존 (원래 계획서) | §3-1 코드 |

### 참고: 실제 결과 (GPT-2 transcoder, HaluEval)

A0·A1·A2 생성과 η 선택에 OOD 표본 2,000개를 쓰고(분할 시 1,000개씩), 따로 떼어 둔 1,000개에서 재구성 오차를 쟀습니다 (낮을수록 좋음).
같은 1,000개에서 원래 decoder(A0)는 19.94입니다.

| 설정 | η | 재구성 오차 |
|---|---|---:|
| 1~4 모두 기존 (Chebyshev) | (0, 0) | 19.94 |
| 1번 기존 + best | (0, 0) | 19.94 |
| `mode="matrix", bound="chebyshev"` | (0.01, 0.01) | 19.61 |
| `mode="matrix"` (best) | (0.02, 0.02) | 19.28 |
| `basis="step", bound="chebyshev"` | (0, 0) | 19.94 |
| `basis="step"` (best) | (0.02, 0.02) | 19.28 |
| `bound="chebyshev"` (elementwise, gae) | (0.09, 0.09) | 17.20 |
| **기본값** (elementwise, gae, best) | (0.24, 0.24) | **14.06** |
| `mode="tie", bound="chebyshev"` | (0.15, 0.15) | 15.73 |
| `mode="tie"` (best) | (0.25, 0.25) | 13.91 |
| 상한 없음 (참고용, 보장 없음) | (0.43, 0.43) | 12.63 |

범위 s, L은 관측된 최댓값으로 정했으므로 엄밀한 보장은 아닙니다.

---

## 5. 환경과 데이터 준비

스크립트마다 맨 위에 설정이 있고, 경로는 환경변수(`DATA_ROOT`, `CACHE_ROOT`)로 바꿀 수 있습니다.
각 스크립트는 환경을 직접 만들고(`uv venv --python 3.10 && uv pip sync requirements.lock`), 결과 폴더 `results/gae_plus/<run>/`에 `run_info.txt`, `uncommitted.diff`, 스크립트 사본을 남깁니다.

```bash
bash scripts/data/download_owt_subset.sh              # OpenWebText shard 0-1(학습), 79(ID 평가), revision 고정 + sha256 확인
bash scripts/train_id/train_gpt2_transcoder.sh [GPU]  # GPT-2 Small transcoder 학습, 논문 설정 (A6000에서 약 1시간)
bash scripts/gae_plus/build_edgar_prompts.sh          # Edgar prompt 생성, revision 고정
```

## 6. 실험 스크립트 (`scripts/gae_plus/`, 실행: `bash <스크립트> [GPU]`)

| 스크립트 | 하는 일 |
|---|---|
| `bounds_grid.sh` | §4 결과표를 만드는 실험. 표본 분할 off/on × η 따로/묶음 × 상한 없음/Chebyshev/best, 그리고 elementwise |
| `probe_bounds.sh` | 원래 계획서의 a_N, b_N (method A, B)과 그때 고른 η |
| `split_check.sh` | A0, A1, A2의 학습 표본 오차 vs 새 표본 오차 |
| `reproduce_gpt2_halueval.sh` | 논문 Table 2 재현 (HaluEval) |
| `reproduce_gpt2_edgar.sh` | 논문 Table 2 재현 (Edgar, 코드 기본값) |
| `reproduce_gpt2_edgar_paper.sh` | Edgar, 논문 설정 (`second_moment`, λ_id = 0) |
| `reproduce_gpt2_edgar_paper_nonorm.sh` | 위와 같고, Step 1의 열 크기 맞춤만 끔 |

## 7. 새로 추가한 환경변수 (기본값 = 기존 코드 동작)

| 변수 | 기본값 | 바꾸면 |
|---|---|---|
| `GAE_SUBSPACE_MATRIX` | `centered` (공분산) | `second_moment`: 논문 Eq. 5의 2차 모멘트 |
| `GAE_STEP1_MATCH_NORMS` | `1` (회전 후 열 크기를 원래대로 맞춤) | `0`: 맞추지 않음 |
| `GAE_AC_LAMBDA_ID` | `5e-2` | `0`: GAE+ 스크립트는 모두 0 사용 |
| `DOMAIN_EVAL_DISJOINT` | `0` | `1`: 평가 prompt를 GAE 학습 prompt와 겹치지 않게 뽑음 |
| `GAE_ID_DATA_FILES` | 없음 (Hub에서 다운로드) | 로컬 parquet 파일 경로 (쉼표로 구분) |

## 8. 테스트

```bash
python tests/test_gae_plus.py      # 27개: 목적함수, 상한이 실제로 덮는지, η 선택, 옵션
python tests/test_gae_options.py   # GAE_SUBSPACE_MATRIX 옵션
```

## 9. 알려진 문제 (고치지 않음)

- HaluEval 평가 prompt가 175개만 만들어집니다 (split 필터링 전에 1.2×n개만 뽑음).
- HaluEval의 `dialogue_history` 필드를 읽지 않아 QA 표본만 사용됩니다.
