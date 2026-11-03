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

기존 GAE 함수(`gae.py`)로 decoder 세 개를 만들고, `gae_plus_select`로 η를 고릅니다. SAE와 transcoder 모두 같은 방식입니다.

| 준비물 | 모양 | 뜻 |
|---|---|---|
| `explainer` | — | 학습된 SAE / transcoder. `explainer(h, return_z=True)`가 `(복원값, feature)`를 돌려줘야 함 |
| `B_id`, `b0` | `[d, k]`, `[d]` | 원래 decoder 가중치와 bias. transcoder: `W_dec.T`, `b_dec_out` / SAE: `W_dec.T`, `b_dec` |
| `H_in` | `[N, d_in]` | OOD 입력 activation |
| `H_out` | `[N, d]` | decoder가 맞춰야 하는 정답 activation. transcoder면 MLP 출력, SAE면 `H_in`과 같음 |

모든 텐서는 같은 장치(GPU/CPU)에 있어야 합니다.

```python
import torch
from gae import gae_torch, fit_affine_constrained_decoder
from gae_plus import gae_plus_select

N = H_in.shape[0]
a, b = slice(0, N // 2), slice(N // 2, N)   # 앞 절반: A1·A2 생성, 뒤 절반: η 선택

# Step 1: OOD 부분공간에 맞춰 회전 + 열 크기를 원래대로 맞춤
B1, geo = gae_torch(B_id, H_out[a], r=32)
B1 = B1 * (B_id.norm(dim=0) / (B1.norm(dim=0) + 1e-8))

# Step 2: decoder와 bias를 다시 맞춤 (λ_id = 0)
W2, b2, _ = fit_affine_constrained_decoder(
    explainer=explainer, activations_in=H_in[a], activations_target=H_out[a],
    B_original=B_id, B_gae=B1, U_ood=geo["U_ood"],
    lam_geom=0.1, lam_id=0.0, lam_gae=0.2, fit_samples=2048)

# decoder 세 개: 가중치 옆에 bias를 한 열로 붙임 [d, k+1]
A0 = torch.cat([B_id, b0[:, None]], 1).double()   # 원래
A1 = torch.cat([B1,   b0[:, None]], 1).double()   # Step 1 이후
A2 = torch.cat([W2,   b2[:, None]], 1).double()   # GAE

# η 결정: A1·A2를 만들 때 쓰지 않은 뒤 절반으로
with torch.no_grad():
    _, Z = explainer(H_in[b], return_z=True)       # feature [N/2, k]
eta1, eta2, A_plus = gae_plus_select(Z.double(), H_out[b].double(), A0, A1, A2)
W_plus, b_plus = A_plus[:, :-1], A_plus[:, -1]     # 새 decoder와 bias
```

- `gae_plus_select`의 기본값은 `mode="elementwise"`, `bound="best"`, `basis="gae"`, `delta=0.05`입니다.
- 위 코드는 실험 스크립트(`gae_plus_probe.build_candidates`)와 같은 A0·A1·A2를 만듭니다. GPT-2 transcoder, HaluEval 1,000개로 확인했고, A0·A1·A2 차이는 0, 세 mode에서 고른 η도 같았습니다 (`scripts/gae_plus/direct_check.sh`).
- 논문 방식(Eqs. 5–6)으로 Step 1을 하려면 `gae_torch(..., subspace_matrix="second_moment")`로 바꾸고 열 크기 맞춤 줄을 지우세요.

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
gae_plus_select(Z, Y, A0, A1, A2)                # Z, Y는 뒤 절반, A0~A2는 앞 절반으로 생성 (§1)

# 원래 계획서 (표본 분할 없음)
gae_plus_select(Z, Y, A0, A1, A2, mode="nosplit", bound="chebyshev")   # Z, Y, A0~A2 모두 OOD 표본 전체
```

A1·A2까지 원래 계획서대로(논문 Eqs. 5–6) 만들려면, §1 코드에서는 마지막 항목처럼 바꾸고, 실험 스크립트에서는 맨 위 설정을 아래처럼 바꾸세요.
GAE+ 스크립트는 코드 기본값(`centered`, `1`)을 씁니다.

```bash
SUBSPACE_MATRIX=second_moment   # 논문 Eq. 5 (2차 모멘트 행렬)
STEP1_MATCH_NORMS=0             # Step 1 회전 후 열 크기 맞춤 끔
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
