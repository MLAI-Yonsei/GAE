#!/bin/bash
set -euo pipefail

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
# Prevent memory-fragmentation OOM during SAEBoost residual training on pythia-1.4b.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Redirect torch.compile/inductor cache to avoid quota exhaustion.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${DATA_ROOT}/torch_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DATA_ROOT}/triton_cache}"
# wandb offline mode — prevent sync-loop disk writes on /home.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${DATA_ROOT}/wandb}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"

# Dolma local data dir (download a subset of URLs into this folder)
export DATA_DIR="${DATA_ROOT}/dataset_cache/dolma"
# Avoid picking up partial/backup files like *.json.gz.1.
export DOLMA_FILE_GLOB="${DOLMA_FILE_GLOB:-${DATA_DIR}/*.json.gz}"

cd ${REPO_ROOT}

DEVICE=${1:-0}
MODEL=${2:-gpt2}       # gpt2 pythia-410m pythia-1.4b
OOD_SET=${3:-fineweb}  # fineweb dolma_web
SAEBOOST_COEF=${4:-1.0}
RESIDUAL_CKPT_ARG=${5:-}

die() { echo "$*" >&2; exit 1; }
is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

SAEBOOST_RESID_BATCH_SIZE="${SAEBOOST_RESID_BATCH_SIZE:-8192}"
STORE_BATCH_SIZE="${STORE_BATCH_SIZE:-64}"
N_BATCHES_IN_BUFFER="${N_BATCHES_IN_BUFFER:-16}"

case "$MODEL" in
    gpt2)
        CHECKPOINT="${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt"
        BATCH_SIZE=256
        ID_LR=2e-5
        ID_LAMBDA=6e-5
        ;;
    pythia-410m)
        CHECKPOINT="${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt"
        BATCH_SIZE=256
        ID_LR=2e-5
        ID_LAMBDA=5.5e-5
        ;;
    pythia-1.4b)
        CHECKPOINT="${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt"
        BATCH_SIZE=64
        ID_LR=2e-5
        ID_LAMBDA=2e-5
        # For 1.4B, significantly smaller defaults to avoid OOM.
        SAEBOOST_RESID_BATCH_SIZE="${SAEBOOST_RESID_BATCH_SIZE:-32}"
        STORE_BATCH_SIZE="${STORE_BATCH_SIZE:-8}"
        N_BATCHES_IN_BUFFER="${N_BATCHES_IN_BUFFER:-2}"
        SAEBOOST_RESID_CONTEXT_SIZE="${SAEBOOST_RESID_CONTEXT_SIZE:-512}"
        ;;
    *)
        die "Unsupported MODEL: $MODEL"
        ;;
esac

if [ -n "$RESIDUAL_CKPT_ARG" ]; then
    RESIDUAL_CHECKPOINT="$RESIDUAL_CKPT_ARG"
else
    # SAEBoost checkpoint naming:
    # checkpoints/{task}/{model}_{explainer}_{ood_set}_saeboost_resid_{coef}.pt
    # Strip trailing ".0" so coef=1.0 → filename suffix "1"
    COEF_SUFFIX="${SAEBOOST_COEF%.0}"
    RESIDUAL_CHECKPOINT="${REPO_DATA}/checkpoints/timeshift_ood/${MODEL}_transcoder_${OOD_SET}_saeboost_resid_${COEF_SUFFIX}.pt"
fi



SAEBOOST_TRAIN_RESIDUAL=${SAEBOOST_TRAIN_RESIDUAL:-1}

# SAEBoost-main residual defaults (config_resid.yaml):
# lr=8e-4, dict_size=1024, num_tokens=1e9, top_k=5, batchtopk.
SAEBOOST_RESID_LR=${SAEBOOST_RESID_LR:-"$ID_LR"}
SAEBOOST_RESID_DICT_SIZE=${SAEBOOST_RESID_DICT_SIZE:-1024}
SAEBOOST_RESID_TOTAL_TOKENS=${SAEBOOST_RESID_TOTAL_TOKENS:-1000000000}
SAEBOOST_RESID_LAMBDA=${SAEBOOST_RESID_LAMBDA:-0}
SAEBOOST_RESID_CONTEXT_SIZE=${SAEBOOST_RESID_CONTEXT_SIZE:-1024}
SAEBOOST_RESID_OBJECTIVE=${SAEBOOST_RESID_OBJECTIVE:-ERM}
SAEBOOST_RESID_SPARSE_METHOD=${SAEBOOST_RESID_SPARSE_METHOD:-batchtopk}
SAEBOOST_RESID_TOP_K=${SAEBOOST_RESID_TOP_K:-5}
# Leave empty by default so trainer uses actual residual d_sae as start_top_k (dense warmup).
SAEBOOST_RESID_START_TOP_K=${SAEBOOST_RESID_START_TOP_K:-}
SAEBOOST_RESID_TOPK_WARMUP_FRAC=${SAEBOOST_RESID_TOPK_WARMUP_FRAC:-0.1}
# SAEBoost-main style inference:
SAEBOOST_RESID_JUMPRELU_THRESHOLD=${SAEBOOST_RESID_JUMPRELU_THRESHOLD:-0.03}
if [ -n "$SAEBOOST_RESID_JUMPRELU_THRESHOLD" ]; then
    _SAEBOOST_RESID_INFER_SPARSE_DEFAULT="jumprelu"
else
    _SAEBOOST_RESID_INFER_SPARSE_DEFAULT="batchtopk"
fi
SAEBOOST_RESID_INFER_SPARSE_METHOD=${SAEBOOST_RESID_INFER_SPARSE_METHOD:-$_SAEBOOST_RESID_INFER_SPARSE_DEFAULT}
SAEBOOST_RESID_INFER_TOP_K=${SAEBOOST_RESID_INFER_TOP_K:-$SAEBOOST_RESID_TOP_K}



# Optional: base-side sparse inference control (normally keep auto).
SAEBOOST_BASE_INFER_SPARSE_METHOD=${SAEBOOST_BASE_INFER_SPARSE_METHOD:-auto}
SAEBOOST_BASE_INFER_TOP_K=${SAEBOOST_BASE_INFER_TOP_K:-}
SAEBOOST_BASE_JUMPRELU_THRESHOLD=${SAEBOOST_BASE_JUMPRELU_THRESHOLD:-}

# Auto-cap residual token budget when local OOD data is finite/small.
# Default OFF for dolma_web: full token scan can be very slow on long documents.
SAEBOOST_AUTO_CAP_TOTAL_TOKENS=${SAEBOOST_AUTO_CAP_TOTAL_TOKENS:-0}
SAEBOOST_TOKEN_SCAN_MAX_EXAMPLES=${SAEBOOST_TOKEN_SCAN_MAX_EXAMPLES:-2000}

if is_true "$SAEBOOST_AUTO_CAP_TOTAL_TOKENS"; then
    echo "[SAEBoost] Auto-capping residual total tokens (requested=${SAEBOOST_RESID_TOTAL_TOKENS})..."
    CAPPED_TOKENS=$(MODEL="$MODEL" OOD_SET="$OOD_SET" REQUESTED_TOKENS="$SAEBOOST_RESID_TOTAL_TOKENS" TOKEN_SCAN_MAX_EXAMPLES="$SAEBOOST_TOKEN_SCAN_MAX_EXAMPLES" python3 - <<'PY'
import json
import os

from transformers import AutoTokenizer

from ood_utils.data_timeshift import load_ood_dataset
from ood_utils.data_utils import _extract_text

model = os.environ["MODEL"]
ood_set = os.environ["OOD_SET"]
requested = int(float(os.environ["REQUESTED_TOKENS"]))
scan_limit = int(os.environ.get("TOKEN_SCAN_MAX_EXAMPLES", "0"))

model_to_tokenizer = {
    "gpt2": "gpt2",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}

result = {"effective_tokens": requested, "reason": "requested", "scanned_examples": 0}

if ood_set != "dolma_web":
    result["reason"] = "skip_nonlocal_streaming_dataset"
    print(json.dumps(result))
    raise SystemExit

tokenizer_name = model_to_tokenizer.get(model, model)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

dataset, text_field = load_ood_dataset(ood_set, split="train")

total_tokens = 0
n_examples = 0
truncated_scan = False

for ex in dataset:
    txt = _extract_text(ex, text_field)
    if not isinstance(txt, str) or len(txt) == 0:
        continue
    n_tok = len(tokenizer.encode(txt, add_special_tokens=False))
    # Approximate BOS boundary handling in streaming packer.
    total_tokens += (n_tok + 1)
    n_examples += 1
    if scan_limit > 0 and n_examples >= scan_limit:
        truncated_scan = True
        break

result["scanned_examples"] = n_examples

if n_examples == 0:
    result["reason"] = "no_examples_scanned_keep_requested"
    print(json.dumps(result))
    raise SystemExit

if truncated_scan:
    # Conservative: if scan is truncated, keep requested budget.
    result["reason"] = "scan_truncated_keep_requested"
    print(json.dumps(result))
    raise SystemExit

effective = min(requested, int(total_tokens))
result["effective_tokens"] = max(1, effective)
result["estimated_total_tokens"] = int(total_tokens)
result["reason"] = "capped_to_available_tokens" if effective < requested else "requested_within_available_tokens"
print(json.dumps(result))
PY
)
    if [ $? -eq 0 ] && [ -n "$CAPPED_TOKENS" ]; then
        CAPPED_JSON=$(echo "$CAPPED_TOKENS" | tail -n 1)
        CAPPED_VALUE=$(echo "$CAPPED_JSON" | python3 -c 'import sys,json; print(int(json.loads(sys.stdin.read())["effective_tokens"]))')
        CAP_REASON=$(echo "$CAPPED_JSON" | python3 -c 'import sys,json; x=json.loads(sys.stdin.read()); print(x.get("reason",""))')
        CAP_SCANNED=$(echo "$CAPPED_JSON" | python3 -c 'import sys,json; x=json.loads(sys.stdin.read()); print(int(x.get("scanned_examples",0)))')
        if [ "$CAPPED_VALUE" -lt "$SAEBOOST_RESID_TOTAL_TOKENS" ]; then
            echo "[SAEBoost] Auto-capped SAEBOOST_RESID_TOTAL_TOKENS: ${SAEBOOST_RESID_TOTAL_TOKENS} -> ${CAPPED_VALUE} (reason=${CAP_REASON}, scanned_examples=${CAP_SCANNED})"
            SAEBOOST_RESID_TOTAL_TOKENS="$CAPPED_VALUE"
        else
            echo "[SAEBoost] Auto-cap check: keeping SAEBOOST_RESID_TOTAL_TOKENS=${SAEBOOST_RESID_TOTAL_TOKENS} (reason=${CAP_REASON}, scanned_examples=${CAP_SCANNED})"
        fi
    else
        echo "[SAEBoost] Auto-cap failed; keeping SAEBOOST_RESID_TOTAL_TOKENS=${SAEBOOST_RESID_TOTAL_TOKENS}"
    fi
fi


if [ ! -f "$CHECKPOINT" ]; then
    die "Base checkpoint not found: $CHECKPOINT"
fi

if ! is_true "$SAEBOOST_TRAIN_RESIDUAL"; then
    if [ ! -f "$RESIDUAL_CHECKPOINT" ]; then
        die "$(cat <<EOF
Residual checkpoint not found: $RESIDUAL_CHECKPOINT
Either provide an existing residual checkpoint or enable training:
  SAEBOOST_TRAIN_RESIDUAL=1 bash scripts/timeshift_ood/saeboost.sh $DEVICE $MODEL $OOD_SET
EOF
)"
    fi
fi

EXTRA_ARGS=(
    --saeboost_residual_checkpoint "$RESIDUAL_CHECKPOINT"
)

if is_true "$SAEBOOST_TRAIN_RESIDUAL"; then
    EXTRA_ARGS+=(
        --saeboost_train_residual
        --saeboost_residual_total_tokens "$SAEBOOST_RESID_TOTAL_TOKENS"
        --saeboost_residual_lr "$SAEBOOST_RESID_LR"
        --saeboost_residual_lambda_sparse "$SAEBOOST_RESID_LAMBDA"
        --saeboost_residual_batch_size "$SAEBOOST_RESID_BATCH_SIZE"
        --saeboost_residual_context_size "$SAEBOOST_RESID_CONTEXT_SIZE"
        --saeboost_residual_dict_size "$SAEBOOST_RESID_DICT_SIZE"
        --saeboost_residual_objective "$SAEBOOST_RESID_OBJECTIVE"
        --saeboost_residual_sparse_method "$SAEBOOST_RESID_SPARSE_METHOD"
        --saeboost_residual_batch_top_k "$SAEBOOST_RESID_TOP_K"
        --saeboost_residual_batch_top_k_warmup_fraction "$SAEBOOST_RESID_TOPK_WARMUP_FRAC"
        --saelens_store_batch_size "$STORE_BATCH_SIZE"
        --saelens_n_batches_in_buffer "$N_BATCHES_IN_BUFFER"
    )
    if [ -n "$SAEBOOST_RESID_START_TOP_K" ]; then
        EXTRA_ARGS+=(--saeboost_residual_batch_top_k_start "$SAEBOOST_RESID_START_TOP_K")
    fi
fi

if [ -n "$SAEBOOST_RESID_JUMPRELU_THRESHOLD" ]; then
    EXTRA_ARGS+=(--saeboost_residual_jumprelu_threshold "$SAEBOOST_RESID_JUMPRELU_THRESHOLD")
fi
if [ -n "$SAEBOOST_BASE_JUMPRELU_THRESHOLD" ]; then
    EXTRA_ARGS+=(--saeboost_base_jumprelu_threshold "$SAEBOOST_BASE_JUMPRELU_THRESHOLD")
fi
if [ -n "$SAEBOOST_BASE_INFER_TOP_K" ]; then
    EXTRA_ARGS+=(--saeboost_base_infer_batch_top_k "$SAEBOOST_BASE_INFER_TOP_K")
fi

########################################################
## Run Time-shift OOD experiment (SAEBoost baseline)
########################################################
WANDB_NAME="${MODEL}_transcoder_${OOD_SET}_saeboost_${SAEBOOST_RESID_SPARSE_METHOD}_k${SAEBOOST_RESID_TOP_K}_coef${SAEBOOST_COEF}"

COMMON_ARGS=(
    timeshift
    --model "$MODEL"
    --explainer transcoder
    --ood_set "$OOD_SET"
    --checkpoint "$CHECKPOINT"
    --saeboost_coef "$SAEBOOST_COEF"
    --output_dir results
    --n_eval 1000
    --batch_size "$BATCH_SIZE"
    --baselines saeboost
    --saeboost_base_infer_sparse_method "$SAEBOOST_BASE_INFER_SPARSE_METHOD"
    --saeboost_residual_infer_sparse_method "$SAEBOOST_RESID_INFER_SPARSE_METHOD"
    --saeboost_residual_infer_batch_top_k "$SAEBOOST_RESID_INFER_TOP_K"
    --use_wandb
    --wandb_project GAE_OOD-Evaluation
    --wandb_name "$WANDB_NAME"
)

CUDA_VISIBLE_DEVICES="$DEVICE" python3 run_experiment.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
