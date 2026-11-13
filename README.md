# GAE+ V2

GAE+ V2는 학습된 SAE 또는 Transcoder의 encoder를 고정하고 decoder만 gradient update 없이 적응시킵니다.
이 브랜치는 V2 적용·평가용 최소 코드이며 ID-trained explainer와 backbone checkpoint를 입력으로 사용합니다.
원 논문과 원형 구현은 [GAE](https://github.com/MLAI-Yonsei/GAE)에서 확인할 수 있습니다.

## 설치

Python 3.10 이상에서 프로젝트 전용 가상환경을 사용합니다.
검증 환경은 Python 3.10.18, PyTorch 2.8.0, NumPy 1.26.4, TransformerLens 2.16.1, Transformers 4.55.2입니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_gae_plus_v2.py --help
```

## 방법

1. 고정 encoder로 adaptation 표본 집합 S의 feature Z와 복원 목표 Y를 계산합니다.
2. Geometry 정렬과 자유 bias 적합으로 HP 후보를 구성하고 서로 다른 rank의 후보 두 개를 선택합니다.
3. 선택된 geometry에서 고정 bias decoder를 다시 적합하고 원본·Step1·재적합 decoder를 simplex 비율로 혼합합니다.
4. 같은 S에서 Fixed 분모 surrogate와 bootstrap q20으로 후보를 단계적으로 선택한 뒤 predictor를 동결합니다.

고정 bias 적합은 다음 합 손실을 최소화합니다.

```text
||Y - Z Wᵀ - b₀||²_F + λ ||W - W₁||²_F + γ ||(I - UUᵀ)W||²_F
W_mix = (1 - η₁ - η₂)W₀ + η₁W₁ + η₂W_fixed
η₁ ≥ 0, η₂ ≥ 0, η₁ + η₂ ≤ 1
```

λ와 γ는 평균 손실로 나누지 않은 sum-loss 단위입니다.
Geometry의 `centered`는 FP32 centered covariance 기반 의료 구현이고 `second-moment`는 FP64 비중심 SVD 기반 구현입니다.
`--reference-solver svd`는 원본 decoder의 SVD basis를, `eigh`는 FP64 W₀W₀ᵀ의 내림차순 eigenbasis를 사용합니다.
Pythia의 standard profile은 기본적으로 `eigh`를 사용합니다.
의료와 standard의 geometry·rank·dtype·N·HP를 원래 실행과 맞춰 지정해야 합니다.

현재 V2의 선택 후보군에는 고정 bias 혼합 후보와 original 자유 bias 후보가 함께 들어갑니다.
`selection.json`의 `metadata.family`는 실제 선택된 `fixed`, `fixed_bias`, `original`을 구분합니다.
각 입력은 feature 삭제 전 후보의 target margin이 0.1보다 클 때 후보를 사용하고, 그렇지 않으면 Fixed의 전체 삭제 궤적을 사용합니다.
이 Fixed fallback은 평가 결과를 보지 않는 사전 정의된 입력별 inference 정책입니다.
Bootstrap q20은 경험적 선택 점수이며 같은 S를 fitting과 선택에 사용한 편향을 보정하는 population 안전 보장이 아닙니다.

## 입력

`--adaptation`과 `--evaluation`에는 역할을 정한 JSONL 또는 JSON 배열 파일을 전달합니다.
FineWeb·EDGAR·HaluEval·PubMedQA·PMC-Patients·MedQuAD를 아래 공통 형식으로 준비하며 원래 실험의 split과 tokenizer를 보존합니다.
이 CLI는 입력 cohort를 새로 분할하지 않습니다.

```json
{"record_id":"adapt-001","leakage_group":"document-001","tokens":[50256,15496,995,13],"role":"adaptation","kind":"qa"}
```

`tokens` 대신 `text` 또는 `prompt`를 주면 backbone의 tokenizer로 BOS를 붙여 최대256개 token으로 변환합니다.
이미 준비한 `tokens`는 자르거나 다시 tokenization하지 않습니다.
가능하면 기존 `text_sha256`, `token_hash`, `source_id`, `group_id`도 보존합니다.
Standard profile의 `kind`는 `qa`, `dialogue`, `fineweb`, `edgar` 중 하나이며 bootstrap은 kind별로 표본을 추출합니다.
의료 profile은 원래 의료 selector의 일반 bootstrap을 사용합니다.
선택과 최종 평가는 문서 group·text·token 기준으로 겹치지 않아야 합니다.

Explainer checkpoint는 `cfg`와 `state_dict`를 포함하는 PyTorch 파일입니다.
적합과 평가는 원래 실행처럼 FP32 explainer checkpoint를 사용합니다.
`state_dict`에는 `W_enc`, `W_dec`, `b_enc`, `b_dec`가 필요하고 Transcoder에는 `b_dec_out`도 필요합니다.
`cfg.activation_fn`은 `relu` 또는 `topk`이며 TopK에는 `topk_k`가 필요합니다.
원본 `sae_training.config.LanguageModelSAERunnerConfig` 형식은 학습 package 없이 data-only 설정으로 읽습니다.

## 실행

의료 profile의 예제입니다.
모델 revision에는 사용한 Hugging Face commit SHA를 지정합니다.

```bash
python run_gae_plus_v2.py fit \
  --model gpt2 --model-revision <commit-sha> \
  --checkpoint <id-trained-sae.pt> --kind sae --layer 8 \
  --adaptation <adaptation.jsonl> --evaluation <evaluation.jsonl> \
  --profile medical --dataset pubmedqa --n-fit 128 \
  --geometry centered --device cuda:0 --output results/medical-v2
```

Standard profile은 해당 실행의 geometry를 명시해서 사용합니다.
GPT-2 SAE의 비중심 geometry 예제입니다.

```bash
python run_gae_plus_v2.py fit \
  --model gpt2 --model-revision <commit-sha> \
  --checkpoint <id-trained-sae.pt> --kind sae --layer 8 \
  --adaptation <fineweb-adaptation.jsonl> --evaluation <fineweb-evaluation.jsonl> \
  --profile standard --dataset fineweb --n-fit 128 \
  --geometry second-moment --device cuda:0 --output results/fineweb-v2
```

`--ranks`, `--lambdas`, `--geom`, `--warm-eta`로 후보군을 지정할 수 있습니다.
기본 grid는 rank={8,32,64,96}, λ={0.001,0.005,0.05,0.2}, γ=0.1입니다.
Reference preset은 기본 rank32·λ0.2·γ0.1이며 후보군에 포함해야 합니다.
작은 차원의 검증에서는 `--preset-rank`와 rank grid를 함께 줄일 수 있습니다.
`--model-dir`는 해당 revision의 로컬 Hugging Face snapshot을 사용하고 `--cache-dir`는 다운로드 cache 위치를 지정합니다.
동결된 decoder를 평가할 때는 실제 로드한 backbone의 설정·weight fingerprint도 일치해야 합니다.
사전에 저장한 TransformerLens `cfg`와 `state_dict` 파일은 `--model-state`로 불러올 수 있습니다.
`--device cpu`로 CPU에서도 실행할 수 있습니다.

`--evaluation`을 생략하면 선택 모델만 저장합니다.
이후 동결된 모델을 같은 backbone·explainer·profile로 평가합니다.

```bash
python run_gae_plus_v2.py evaluate \
  --model gpt2 --model-revision <commit-sha> \
  --checkpoint <id-trained-sae.pt> --kind sae --layer 8 \
  --decoder results/medical-v2/selected_decoder.pt \
  --evaluation <evaluation.jsonl> --profile medical \
  --device cuda:0 --output results/medical-v2-evaluation
```

출력은 선택 decoder·선택 기록과 평가 시 원시 score·집계입니다.
기존 출력 디렉터리는 덮어쓰지 않습니다.
평가기는 원 공개 코드의 scalar full-context ablation을 사용하며 빠른 batch 구현보다 오래 걸릴 수 있습니다.

## 지표와 결과

nAOPC는 공개 구현처럼 각 입력의 normalized AOPC를 [-1,1]로 clipping한 뒤 유효 입력에서 평균합니다.
nComp는 같은 budget={1,2,4,8,16,32,64,128}에서 계산한 normalized comprehensiveness의 평균이며32 한 점의 값과 구분합니다.
정규화에 쓰는 원본 target margin이0.1보다 큰 입력을 사용하고, 이를 후보 전체의 일정 비율 조건으로 바꾸지 않습니다.
ΔCE는 선택·현재 평가에서 사용하지 않습니다.

결과표는 [의료 실험](docs/medical_results.md)과 [원 benchmark 비교](docs/orig_results.md)에 있습니다.
표의 실험 run ID는 원시 결과 보관 자료의 식별자입니다.
표 재현에는 해당 checkpoint·cohort·N·HP·geometry·환경을 사용해야 하며 서로 다른 조건의 보고값은 각 문서의 주석을 따릅니다.
