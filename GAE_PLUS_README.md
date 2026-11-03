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

`Z, Y, A0, A1, A2`는 어떤 mode든 똑같이 넣습니다. 달라지는 것은 **어떤 표본을 넣느냐**뿐입니다.
`"elementwise"`, `"matrix"`는 A1·A2를 만들 때 쓰지 않은 표본을 Z, Y로 넣어야 하고 (§1처럼 반으로 나눔), `"nosplit"`은 같은 표본을 넣어도 됩니다.

돌려주는 값은 `(eta1, eta2, A_plus)`입니다. `A_plus[:, :-1]`이 decoder, `A_plus[:, -1]`이 bias입니다.

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
