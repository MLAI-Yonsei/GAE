#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

DEVICE=${1:-0}
MODEL=${2:-gpt2}     # gpt2 pythia-1.4b
OOD_SET=${3:-edgar}  # patents edgar

# Resolve checkpoint path per model
if [ "$MODEL" = "gpt2" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
    BATCH_SIZE=256
elif [ "$MODEL" = "pythia-1.4b" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt
    BATCH_SIZE=64
else
    echo "Unknown model: $MODEL"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain \
    --model $MODEL \
    --explainer transcoder \
    --ood_set $OOD_SET \
    --checkpoint $CHECKPOINT \
    --output_dir results \
    --n_eval 1000 \
    --baselines ood_finetuned \
    --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts \
    --ood_finetuned_total_tokens 5000000 \
    --ood_finetuned_batch_size $BATCH_SIZE \
    --use_wandb \
    --wandb_project GAE_OOD-Evaluation \
    --wandb_name ${MODEL}_transcoder_${OOD_SET}_ood_finetuned_kissane5M
