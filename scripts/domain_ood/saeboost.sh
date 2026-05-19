#!/bin/bash
# Train + evaluate SAEBoost residual for Domain OOD.
# Adapted from scripts/timeshift_ood/saeboost.sh (timeshift version).
set -euo pipefail

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
export DATA_DIR="${DATA_ROOT}/dataset_cache/dolma"
export DOLMA_FILE_GLOB="${DOLMA_FILE_GLOB:-${DATA_DIR}/*.json.gz}"
# Prevent memory-fragmentation OOM during SAEBoost residual training on pythia-1.4b.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Redirect torch.compile/inductor cache to avoid quota exhaustion.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${DATA_ROOT}/torch_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DATA_ROOT}/triton_cache}"
# wandb offline mode — prevent sync-loop disk writes on /home.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${DATA_ROOT}/wandb}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"

cd ${REPO_ROOT}

DEVICE=${1:-0}
MODEL=${2:-pythia-1.4b}          # gpt2 pythia-410m pythia-1.4b
OOD_SET=${3:-patents}            # patents edgar govreport
SAEBOOST_COEF=${4:-1.0}
RESIDUAL_CKPT_ARG=${5:-}

die() { echo "$*" >&2; exit 1; }
is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

DOMAIN_JSONL_DIR="${DOMAIN_OOD_JSONL_DIR:-${REPO_DATA}/domain_ood_prompts}"

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
    COEF_SUFFIX="${SAEBOOST_COEF%.0}"
    RESIDUAL_CHECKPOINT="${REPO_DATA}/checkpoints/domain_ood/${MODEL}_transcoder_${OOD_SET}_saeboost_resid_${COEF_SUFFIX}.pt"
fi

SAEBOOST_TRAIN_RESIDUAL=${SAEBOOST_TRAIN_RESIDUAL:-1}

# Match timeshift defaults (coef=1, batchtopk k=5, ERM, 1e9 tokens).
SAEBOOST_RESID_LR=${SAEBOOST_RESID_LR:-"$ID_LR"}
SAEBOOST_RESID_DICT_SIZE=${SAEBOOST_RESID_DICT_SIZE:-1024}
SAEBOOST_RESID_TOTAL_TOKENS=${SAEBOOST_RESID_TOTAL_TOKENS:-1000000000}
SAEBOOST_RESID_LAMBDA=${SAEBOOST_RESID_LAMBDA:-0}
SAEBOOST_RESID_CONTEXT_SIZE=${SAEBOOST_RESID_CONTEXT_SIZE:-1024}
SAEBOOST_RESID_OBJECTIVE=${SAEBOOST_RESID_OBJECTIVE:-ERM}
SAEBOOST_RESID_SPARSE_METHOD=${SAEBOOST_RESID_SPARSE_METHOD:-batchtopk}
SAEBOOST_RESID_TOP_K=${SAEBOOST_RESID_TOP_K:-5}
SAEBOOST_RESID_START_TOP_K=${SAEBOOST_RESID_START_TOP_K:-}
SAEBOOST_RESID_TOPK_WARMUP_FRAC=${SAEBOOST_RESID_TOPK_WARMUP_FRAC:-0.1}
SAEBOOST_RESID_JUMPRELU_THRESHOLD=${SAEBOOST_RESID_JUMPRELU_THRESHOLD:-0.03}
if [ -n "$SAEBOOST_RESID_JUMPRELU_THRESHOLD" ]; then
    _SAEBOOST_RESID_INFER_SPARSE_DEFAULT="jumprelu"
else
    _SAEBOOST_RESID_INFER_SPARSE_DEFAULT="batchtopk"
fi
SAEBOOST_RESID_INFER_SPARSE_METHOD=${SAEBOOST_RESID_INFER_SPARSE_METHOD:-$_SAEBOOST_RESID_INFER_SPARSE_DEFAULT}
SAEBOOST_RESID_INFER_TOP_K=${SAEBOOST_RESID_INFER_TOP_K:-$SAEBOOST_RESID_TOP_K}

SAEBOOST_BASE_INFER_SPARSE_METHOD=${SAEBOOST_BASE_INFER_SPARSE_METHOD:-auto}
SAEBOOST_BASE_INFER_TOP_K=${SAEBOOST_BASE_INFER_TOP_K:-}
SAEBOOST_BASE_JUMPRELU_THRESHOLD=${SAEBOOST_BASE_JUMPRELU_THRESHOLD:-}

if [ ! -f "$CHECKPOINT" ]; then
    die "Base checkpoint not found: $CHECKPOINT"
fi

if ! is_true "$SAEBOOST_TRAIN_RESIDUAL"; then
    if [ ! -f "$RESIDUAL_CHECKPOINT" ]; then
        die "Residual checkpoint not found: $RESIDUAL_CHECKPOINT (set SAEBOOST_TRAIN_RESIDUAL=1 to train)"
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

WANDB_NAME="${MODEL}_transcoder_${OOD_SET}_saeboost_${SAEBOOST_RESID_SPARSE_METHOD}_k${SAEBOOST_RESID_TOP_K}_coef${SAEBOOST_COEF}"

COMMON_ARGS=(
    domain
    --model "$MODEL"
    --explainer transcoder
    --ood_set "$OOD_SET"
    --checkpoint "$CHECKPOINT"
    --saeboost_coef "$SAEBOOST_COEF"
    --output_dir results
    --n_eval 1000
    --batch_size "$BATCH_SIZE"
    --baselines saeboost
    --domain_ood_jsonl_dir "$DOMAIN_JSONL_DIR"
    --saeboost_base_infer_sparse_method "$SAEBOOST_BASE_INFER_SPARSE_METHOD"
    --saeboost_residual_infer_sparse_method "$SAEBOOST_RESID_INFER_SPARSE_METHOD"
    --saeboost_residual_infer_batch_top_k "$SAEBOOST_RESID_INFER_TOP_K"
    --use_wandb
    --wandb_project GAE_Domain-OOD
    --wandb_name "$WANDB_NAME"
)

CUDA_VISIBLE_DEVICES="$DEVICE" python3 run_experiment.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
