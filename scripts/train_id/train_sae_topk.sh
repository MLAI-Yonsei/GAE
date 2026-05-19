#!/bin/bash
# Train TopK SAE for GPT-2, Pythia-410M, Pythia-1.4B
#
# Usage: bash scripts/train_id/train_sae_topk.sh <GPU_ID> <MODEL>
#   e.g., bash scripts/train_id/train_sae_topk.sh 2 gpt2
#         bash scripts/train_id/train_sae_topk.sh 4 pythia-410m
#         bash scripts/train_id/train_sae_topk.sh 5 pythia-1.4b

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

DEVICE=${1:?Usage: $0 <GPU_ID> <MODEL>}
MODEL=${2:?Usage: $0 <GPU_ID> <MODEL>}

# ── Common TopK hyperparameters (from Gao et al. 2024 / BatchTopK) ──
TOPK_K=32
LR=3e-4
TOTAL_TOKENS=1000000000   # 1B tokens
TRAIN_BATCH_SIZE=4096
STORE_BATCH_SIZE=32
N_BATCHES_IN_BUFFER=128
CONTEXT_SIZE=128
LR_SCHEDULER=cosineannealingwarmup
LR_WARMUP=5000
B_DEC_INIT=geometric_median
TOPK_AUX_COEFF=0.03125   # 1/32
ACTIVATION_FN=topk

# ── Per-model settings ──
# dataset_streaming=False: use local cache (no network dependency during training)
# All datasets use local cache (no network dependency during training)
# Pile: use pre-downloaded single shard (~5.9M examples, ~10B tokens)
if [ "$MODEL" = "gpt2" ]; then
    DICT_SIZE=24576
    DATASET_PATH="Skylion007/openwebtext"
    DATASET_SPLIT="train"
elif [ "$MODEL" = "pythia-410m" ]; then
    DICT_SIZE=32768
    DATASET_PATH="${DATA_ROOT}/datasets_cache/pile_shard/train/00.jsonl.zst"
    DATASET_SPLIT="train"
elif [ "$MODEL" = "pythia-1.4b" ]; then
    DICT_SIZE=65536
    DATASET_PATH="${DATA_ROOT}/datasets_cache/pile_shard/train/00.jsonl.zst"
    DATASET_SPLIT="train"
else
    echo "Unknown model: $MODEL (choose gpt2, pythia-410m, pythia-1.4b)"
    exit 1
fi

WANDB_NAME="${MODEL}_sae_topk_k${TOPK_K}_lr${LR}"

echo "============================================"
echo " TopK SAE Training (local cache mode)"
echo "  Model:       $MODEL"
echo "  Dict size:   $DICT_SIZE"
echo "  TopK k:      $TOPK_K"
echo "  LR:          $LR"
echo "  Tokens:      $TOTAL_TOKENS (1B)"
echo "  Batch size:  $TRAIN_BATCH_SIZE"
echo "  Dataset:     $DATASET_PATH ($DATASET_SPLIT)"
echo "  GPU:         $DEVICE"
echo "============================================"

CUDA_VISIBLE_DEVICES=$DEVICE python3 train_explainers.py \
    --model $MODEL \
    --explainer sae \
    --activation_fn $ACTIVATION_FN \
    --topk_k $TOPK_K \
    --topk_aux_coeff $TOPK_AUX_COEFF \
    --objective ERM \
    --dict_size_k $DICT_SIZE \
    --dataset_path $DATASET_PATH \
    --dataset_split "$DATASET_SPLIT" \
    --context_size $CONTEXT_SIZE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --store_batch_size $STORE_BATCH_SIZE \
    --n_batches_in_buffer $N_BATCHES_IN_BUFFER \
    --total_tokens $TOTAL_TOKENS \
    --lr $LR \
    --lr_scheduler_name $LR_SCHEDULER \
    --lr_warm_up_steps $LR_WARMUP \
    --b_dec_init_method $B_DEC_INIT \
    --checkpoint_path ${REPO_DATA}/checkpoints \
    --n_checkpoints 3 \
    --use_wandb \
    --wandb_project GAE_ID-training \
    --wandb_name $WANDB_NAME
