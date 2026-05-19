#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

DEVICE=$1
MODEL=${2:-gpt2}       # gpt2 pythia-410m pythia-1.4b
# OOD_SET=${3:-patents} # patents edgar


for OOD_SET in edgar;do # patents edgar; do
    for GAE_R in 512 1024 2048;do # 32 64 128 256 512 1024 2048;do

        if [ "$MODEL" = "gpt2" ]; then
            CHECKPOINT=${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
            BATCH_SIZE=256
            GAE_DICT_METHOD="normal"
        elif [ "$MODEL" = "pythia-410m" ]; then
            CHECKPOINT=${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt
            BATCH_SIZE=256
            GAE_DICT_METHOD="normal"
        elif [ "$MODEL" = "pythia-1.4b" ]; then
            CHECKPOINT="${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt"
            BATCH_SIZE=64
            GAE_DICT_METHOD="row"
        fi  

        CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain \
                --model $MODEL \
                --explainer transcoder \
                --ood_set $OOD_SET \
                --checkpoint $CHECKPOINT \
                --output_dir results \
                --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts \
                --n_eval 1000 \
                --batch_size $BATCH_SIZE \
                --baselines gae \
                --gae_dict_method $GAE_DICT_METHOD \
                --gae_r $GAE_R \
                --use_wandb \
                --wandb_project GAE_OOD-Evaluation \
                --wandb_name ${MODEL}_transcoder_${OOD_SET}_gae_r${GAE_R}
    done

    # Additional GAE run with theoretically-grounded auto-rank selection
    for ENERGY in 0.99 0.95 0.90; do
    CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain \
    --model $MODEL \
    --explainer transcoder \
    --ood_set $OOD_SET \
    --checkpoint $CHECKPOINT \
    --output_dir results \
    --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts \
    --n_eval 1000 \
    --batch_size $BATCH_SIZE \
    --baselines gae \
    --gae_dict_method $GAE_DICT_METHOD \
    --gae_rank_mode energy \
    --gae_rank_energy $ENERGY \
    --gae_rank_delta_mode ood \
    --gae_rank_min 1 \
    --use_wandb \
    --wandb_project GAE_OOD-Evaluation \
    --wandb_name ${MODEL}_transcoder_${OOD_SET}_gae_e${ENERGY}
    done
done