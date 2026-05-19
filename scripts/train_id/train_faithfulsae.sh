#!/bin/bash
# Train FaithfulSAE transcoders (transcoders trained on model's own synthetic data)
# Based on FaithfulSAE (arXiv:2506.17673) methodology.
# Synthetic data generated with: temperature=1.0, top_p=0.9, repetition_penalty=1.1
# Training tokens: 100M (paper default)
# Usage: bash scripts/train_id/train_faithfulsae.sh <DEVICE> <MODEL>

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

DEVICE=${1:-0}
MODEL=${2:-gpt2}

# NOTE: Use DIRECTORY path (not .jsonl file path) for ActivationsStore compatibility
SYNTHETIC_DIR="${REPO_DATA}/synthetic_data"

if [ "$MODEL" = "gpt2" ]; then
    DICT_SIZE=24576
    LR=2e-5
    L1=6e-5
    DATASET_DIR="${SYNTHETIC_DIR}/gpt2"
elif [ "$MODEL" = "pythia-410m" ]; then
    DICT_SIZE=32768
    LR=2e-5
    L1=5.5e-5
    DATASET_DIR="${SYNTHETIC_DIR}/pythia-410m"
elif [ "$MODEL" = "pythia-1.4b" ]; then
    DICT_SIZE=65536
    LR=2e-5
    L1=2e-5
    DATASET_DIR="${SYNTHETIC_DIR}/pythia-1.4b"
fi

CUDA_VISIBLE_DEVICES=$DEVICE python3 train_explainers.py \
    --model $MODEL \
    --explainer transcoder \
    --objective ERM \
    --dict_size_k $DICT_SIZE \
    --context_size 128 \
    --train_batch_size 256 \
    --store_batch_size 32 \
    --n_batches_in_buffer 128 \
    --total_tokens 100000000 \
    --lr $LR \
    --l1_coefficient $L1 \
    --lr_scheduler_name constantwithwarmup \
    --lr_warm_up_steps 5000 \
    --b_dec_init_method mean \
    --dataset_path ${DATASET_DIR} \
    --checkpoint_path ${REPO_DATA}/checkpoints \
    --use_wandb \
    --wandb_project GAE_ID-training \
    --wandb_name ${MODEL}_faithfulsae_transcoder_erm_lr${LR}_lambda${L1}
