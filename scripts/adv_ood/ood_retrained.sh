#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

# Dolma local data dir (download a subset of URLs into this folder)
export DATA_DIR="${DATA_ROOT}/dataset_cache/dolma"


cd ${REPO_ROOT}

DEVICE=$1
MODEL=${2:-gpt2}             # gpt2 pythia-410m pythia-1.4b

# Resolve checkpoint path per model
if [ "$MODEL" = "gpt2" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
elif [ "$MODEL" = "pythia-410m" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt
elif [ "$MODEL" = "pythia-1.4b" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt
fi

## $MODEL + Transcoder
for ood_set in halu_eval jailbreakbench;do
for lr in 1e-3 5e-4 1e-4; do
for lambda_sparse in 0.001 0.01 0.1; do
CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py adv \
--model $MODEL \
--explainer transcoder \
--ood_set $ood_set \
--checkpoint $CHECKPOINT \
--output_dir results \
--n_eval 1000 \
--baselines ood_retrained \
--ood_retrained_max_epochs 200 \
--ood_retrained_lr $lr \
--ood_retrained_lambda_sparse $lambda_sparse \
--use_wandb \
--wandb_project GAE_Adversarial-OOD \
--wandb_name ${MODEL}_transcoder_${ood_set}_ood_retrained_lr${lr}_lambda${lambda_sparse}
done
done
done
