#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

# Dolma local data dir (download a subset of URLs into this folder)
export DATA_DIR="${DATA_ROOT}/dataset_cache/dolma"
# Avoid picking up partial/backup files like *.json.gz.1.
export DOLMA_FILE_GLOB="${DOLMA_FILE_GLOB:-${DATA_DIR}/*.json.gz}"

cd ${REPO_ROOT}

DEVICE=$1
# MODEL=${2:-gpt2}       # gpt2 pythia-410m pythia-1.4b
# OOD_SET=${3:-fineweb} # fineweb dolma_web



########################################################
########################################################
## Run Time-shift OOD experiment
########################################################
########################################################
for OOD_SET in dolma_web; do # fineweb dolma_web; do
    for MODEL in pythia-410m pythia-1.4b; do # gpt2 pythia-410m pythia-1.4b; do
    
    if [ "$MODEL" = "gpt2" ]; then
        CHECKPOINT=${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
        BATCH_SIZE=256
    elif [ "$MODEL" = "pythia-410m" ]; then
        CHECKPOINT=${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt
        BATCH_SIZE=256
    elif [ "$MODEL" = "pythia-1.4b" ]; then
        CHECKPOINT="${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt"
        BATCH_SIZE=64
    fi

    CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py timeshift \
        --model $MODEL \
        --explainer transcoder \
        --ood_set $OOD_SET \
        --checkpoint $CHECKPOINT \
        --output_dir results \
        --n_eval 1000 \
        --batch_size $BATCH_SIZE \
        --baselines fixed \
        --use_wandb \
        --wandb_project GAE_OOD-Evaluation \
        --wandb_name ${MODEL}_transcoder_${OOD_SET}_fixed
    done
done
