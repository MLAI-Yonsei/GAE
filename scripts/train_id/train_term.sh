#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"


cd ${REPO_ROOT}

DEVICE=$1


# ## GPT-2 Small + Transcoder
# for lr in 2e-5; do
# for lambda_sparse in 6e-5 8e-5 1e-4 2.5e-4;do
# CUDA_VISIBLE_DEVICES=$DEVICE python3 train_explainers.py \
# --model gpt2 \
# --explainer transcoder \
# --objective TERM \
# --term_t 0.001 \
# --dict_size_k 24576 \
# --context_size 128 \
# --train_batch_size 256 \
# --store_batch_size 32 \
# --n_batches_in_buffer 128 \
# --total_tokens 60000000 \
# --lr $lr \
# --l1_coefficient $lambda_sparse \
# --lr_scheduler_name constantwithwarmup \
# --lr_warm_up_steps 5000 \
# --b_dec_init_method mean \
# --checkpoint_path ${REPO_DATA}/checkpoints \
# --use_wandb \
# --wandb_project GAE_ID-training \
# --wandb_name gpt2_transcoder_term_t0.001_lr${lr}_lambda${lambda_sparse}
# done
# done


# ## Pythia-410M + Transcoder
# for lr in 2e-5; do
# for lambda_sparse in 5.5e-5;do 
# CUDA_VISIBLE_DEVICES=$DEVICE python3 train_explainers.py \
# --model pythia-410m \
# --explainer transcoder \
# --objective TERM \
# --term_t 0.001 \
# --dict_size_k 32768 \
# --context_size 128 \
# --train_batch_size 256 \
# --store_batch_size 32 \
# --n_batches_in_buffer 128 \
# --total_tokens 60000000 \
# --lr $lr \
# --l1_coefficient $lambda_sparse \
# --lr_scheduler_name constantwithwarmup \
# --lr_warm_up_steps 5000 \
# --b_dec_init_method mean \
# --checkpoint_path ${REPO_DATA}/checkpoints \
# --use_wandb \
# --wandb_project GAE_ID-training \
# --wandb_name pythia-410m_transcoder_term_t0.001_lr${lr}_lambda${lambda_sparse}
# done
# done


# ## Pythia-1.4B + Transcoder
# for lr in 2e-5; do
# for lambda_sparse in 2e-5;do
# CUDA_VISIBLE_DEVICES=$DEVICE python3 train_explainers.py \
# --model pythia-1.4b \
# --explainer transcoder \
# --objective TERM \
# --term_t 0.001 \
# --dict_size_k 65536 \
# --context_size 128 \
# --train_batch_size 256 \
# --store_batch_size 32 \
# --n_batches_in_buffer 128 \
# --total_tokens 60000000 \
# --lr $lr \
# --l1_coefficient $lambda_sparse \
# --lr_scheduler_name constantwithwarmup \
# --lr_warm_up_steps 5000 \
# --b_dec_init_method mean \
# --checkpoint_path ${REPO_DATA}/checkpoints \
# --use_wandb \
# --wandb_project GAE_ID-training \
# --wandb_name pythia-1.4b_transcoder_term_t0.001_lr${lr}_lambda${lambda_sparse}
# done
# done