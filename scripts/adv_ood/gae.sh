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

for ood_set in halu_eval jailbreakbench;do
    for gae_r in 1024; do
        CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py adv \
            --model $MODEL \
            --explainer transcoder \
            --ood_set $ood_set \
            --checkpoint $CHECKPOINT \
            --output_dir results \
            --n_eval 1000 \
            --baselines gae \
            --gae_r $gae_r \
            --use_wandb \
            --wandb_project GAE_Adversarial-OOD \
            --wandb_name ${MODEL}_transcoder_${ood_set}_gae_r${gae_r}
    done
done
