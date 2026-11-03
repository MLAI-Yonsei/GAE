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
| 1 | 표본 사용 | 같은 표본으로 Step 1·2를 만들고 η도 결정 | 절반 A로 Step 1·2, 나머지 절반 B로 η 결정 | `mode="nosplit"` (Z, Y에 A1·A2를 만든 표본을 그대로 넣음) |
| 2 | 상한 부등식 | Chebyshev | Chebyshev와 Bernstein 중 작은 값 | `bound="chebyshev"` |
| 3 | 계수 기준 | Step 1, Step 2 각각의 변화량 `(A1−A0, A2−A1)`의 계수 | Step 1, GAE 전체의 변화량 `(A1−A0, A2−A0)`의 계수 (= η) | `basis="step"` (상한만 step 기준으로 계산) |
| 4 | 상한 개수 | 2개: ‖Ĥ − H‖ 하나(a_N), ‖ĝ − g‖ 하나(b_N) | 5개: H의 숫자 3개 + g의 숫자 2개에 각각 하나씩 (δ/5씩) | `mode="matrix"` |

η₁ = η₂로 묶는 옵션(`tie`)은 새 정의에서는 쓰지 않기로 해서 없앴습니다.

### 옵션끼리의 관계

- `basis`는 3번 "계수 기준"의 코드 인자 이름이며, `mode="elementwise"`일 때만 적용됩니다.
- `mode="matrix"`는 항상 `(A1−A0, A2−A1)` 기준이므로, 4번을 기존으로 돌리면 3번도 기존이 됩니다.
- `mode="nosplit"`(1번 기존)은 원래 계획서의 상한을 쓰므로 `basis`는 영향이 없고, `bound`(2번)만 고를 수 있습니다.
- 1번과 4번은 둘 다 `mode`로 고르므로 함께 지정할 수 없습니다. `"nosplit"`은 1·3·4번이 모두 원래 계획서 방식이고, `"matrix"`는 표본 분할을 쓰면서 3·4번만 기존입니다.
- 어떤 옵션을 쓰든 함수가 돌려주는 η는 항상 위의 새 정의(섞는 비율)입니다.

---

## 3. 옵션 정리

### 3-1. `gae_plus_select`의 인자

```python
gae_plus_select(Z, Y, A0, A1, A2, mode="elementwise", bound="best", delta=0.05, basis="gae")
```

| 인자 | 하는 일 | 고를 수 있는 값 | **기본값 (현재)** | **원래 계획서** |
|---|---|---|---|---|
| `Z`, `Y` | η를 정할 때 쓰는 feature `[n, k]` / target `[n, d]` (float64) | — | 절반 B | 전체 표본 (A1·A2를 만든 표본과 같음) |
| `A0, A1, A2` | 원래 / Step 1 이후 / GAE decoder `[d, k+1]` (float64) | — | 절반 A로 생성 | 전체 표본으로 생성 |
| `mode` | 상한을 어디에 몇 개 붙일지 | `"elementwise"`, `"matrix"`, `"nosplit"` | `"elementwise"` (H, g의 숫자 5개에 하나씩) | `"nosplit"` (데이터 통계 M = mean z̄z̄ᵀ, T = mean e₀z̄ᵀ에 2개 → 벌점 a_N‖A₊−A0‖²_F + 2b_N‖A₊−A0‖_F) |
| `bound` | 상한에 쓸 부등식 | `"best"`, `"bernstein"`, `"chebyshev"` | `"best"` | `"chebyshev"` |
| `delta` | 전체 실패 확률. 작을수록 상한이 커지고 η가 0 쪽으로 감 | 0 < δ < 1 | `0.05` | `0.05` |
| `basis` | 상한을 붙일 계수 기준 (`mode="elementwise"`일 때만 적용) | `"gae"`, `"step"` | `"gae"` | 영향 없음 (`"nosplit"`은 ‖A₊−A0‖_F 기준) |

```python
# 기본값
gae_plus_select(Z_B, Y_B, A0, A1, A2)          # A0~A2는 절반 A로 생성

# 원래 계획서 (표본 분할 없음)
gae_plus_select(Z, Y, A0, A1, A2, mode="nosplit", bound="chebyshev")   # Z, Y, A0~A2 모두 전체 표본
```

돌려주는 값은 `(eta1, eta2, A_plus)`입니다. `A_plus[:, :-1]`이 decoder, `A_plus[:, -1]`이 bias입니다.

## 4. 환경과 데이터 준비

스크립트마다 맨 위에 설정이 있고, 경로는 환경변수(`DATA_ROOT`, `CACHE_ROOT`)로 바꿀 수 있습니다.
각 스크립트는 환경을 직접 만들고(`uv venv --python 3.10 && uv pip sync requirements.lock`), 결과 폴더 `results/gae_plus/<run>/`에 `run_info.txt`, `uncommitted.diff`, 스크립트 사본을 남깁니다.

```bash
bash scripts/data/download_owt_subset.sh              # OpenWebText shard 0-1(학습), 79(ID 평가), revision 고정 + sha256 확인
bash scripts/train_id/train_gpt2_transcoder.sh [GPU]  # GPT-2 Small transcoder 학습, 논문 설정 (A6000에서 약 1시간)
bash scripts/gae_plus/build_edgar_prompts.sh          # Edgar prompt 생성, revision 고정
```

## 5. 실험 스크립트 (`scripts/gae_plus/`, 실행: `bash <스크립트> [GPU]`)

| 스크립트 | 하는 일 |
|---|---|
| `bounds_grid.sh` | 표본 분할 off/on × 상한 없음/Chebyshev/best (상한 2개), 그리고 elementwise (step/gae 기준) |
| `probe_bounds.sh` | 원래 계획서의 a_N, b_N (method A, B)과 그때 고른 η |
| `split_check.sh` | A0, A1, A2의 학습 표본 오차 vs 새 표본 오차 |
| `reproduce_gpt2_halueval.sh` | 논문 Table 2 재현 (HaluEval) |
| `reproduce_gpt2_edgar.sh` | 논문 Table 2 재현 (Edgar, 코드 기본값) |
| `reproduce_gpt2_edgar_paper.sh` | Edgar, 논문 설정 (`second_moment`, λ_id = 0) |
| `reproduce_gpt2_edgar_paper_nonorm.sh` | 위와 같고, Step 1의 열 크기 맞춤만 끔 |
