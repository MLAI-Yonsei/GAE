#!/bin/bash

export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

DEVICE=${1:-0}
MODEL=${2:-gpt2}       # gpt2 pythia-410m pythia-1.4b
OOD_SET=${3:-patents} # patents edgar

OOD_RETRAINED_WARM_START=${OOD_RETRAINED_WARM_START:-1}    # 1=on, 0=off
OOD_RETRAINED_REINIT_B_DEC=${OOD_RETRAINED_REINIT_B_DEC:-1} # 1=on, 0=off
OOD_RETRAINED_CONTEXT_SIZE=${OOD_RETRAINED_CONTEXT_SIZE:-128}

if [ "$MODEL" = "gpt2" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
    BATCH_SIZE=16384
elif [ "$MODEL" = "pythia-410m" ]; then
    CHECKPOINT=${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt
    BATCH_SIZE=16384
elif [ "$MODEL" = "pythia-1.4b" ]; then
    CHECKPOINT="${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt"
    BATCH_SIZE=8192
fi

for TOTAL_TOKENS in 1000000000; do
for LR in 1e-5; do
for SPARSITY in 5e-6; do # 5e-6; do # 2e-5 5e-6; do
EXTRA_ARGS=()
if [ "$OOD_RETRAINED_WARM_START" = "0" ] || [ "$OOD_RETRAINED_WARM_START" = "false" ]; then
    EXTRA_ARGS+=(--no_ood_retrained_warm_start)
fi
if [ "$OOD_RETRAINED_REINIT_B_DEC" = "0" ] || [ "$OOD_RETRAINED_REINIT_B_DEC" = "false" ]; then
    EXTRA_ARGS+=(--no_ood_retrained_reinit_b_dec)
fi
CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain \
--model $MODEL \
--explainer transcoder \
--ood_set $OOD_SET \
--checkpoint $CHECKPOINT \
--output_dir results \
--n_eval 1000 \
--baselines ood_retrained \
--domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts \
--ood_retrained_streaming \
--ood_retrained_context_size $OOD_RETRAINED_CONTEXT_SIZE \
--ood_retrained_total_tokens $TOTAL_TOKENS \
--ood_retrained_lr $LR \
--ood_retrained_lambda_sparse $SPARSITY \
--ood_retrained_batch_size $BATCH_SIZE \
${EXTRA_ARGS[@]} \
--use_wandb \
--wandb_project GAE_OOD-Evaluation \
--wandb_name ${MODEL}_transcoder_${OOD_SET}_ood_retrained_lr${LR}_lambda${SPARSITY}_total${TOTAL_TOKENS}
done
done
done


# GPT2 + Patents
CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model gpt2 --explainer transcoder --ood_set patents --checkpoint ${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 1000000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 16384 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name gpt2_transcoder_patents_ood_retrained_lr1e-5_lambda5e-6_total1000000000

# # Pythia-410m + Patents
# CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model pythia-410m --explainer transcoder --ood_set patents --checkpoint ${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 4000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 256 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name pythia-410m_transcoder_patents_ood_retrained_lr1e-5_lambda5e-6_total4000000

# # Pythia-1.4b + Patents
# CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model pythia-1.4b --explainer transcoder --ood_set patents --checkpoint ${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 4000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 64 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name pythia-1.4b_transcoder_patents_ood_retrained_lr1e-5_lambda5e-6_total4000000

# # GPT2 + Edgar
# CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model gpt2 --explainer transcoder --ood_set edgar --checkpoint ${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 4000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 256 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name gpt2_transcoder_edgar_ood_retrained_lr1e-5_lambda5e-6_total4000000

# # Pythia-410m + Edgar
# CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model pythia-410m --explainer transcoder --ood_set edgar --checkpoint ${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 4000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 256 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name pythia-410m_transcoder_edgar_ood_retrained_lr1e-5_lambda5e-6_total4000000

# # Pythia-1.4b + Edgar
# CUDA_VISIBLE_DEVICES=$DEVICE python3 run_experiment.py domain --model pythia-1.4b --explainer transcoder --ood_set edgar --checkpoint ${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt --output_dir results --n_eval 1000 --baselines ood_retrained --domain_ood_jsonl_dir ${REPO_DATA}/domain_ood_prompts --ood_retrained_streaming --ood_retrained_context_size 128 --ood_retrained_total_tokens 4000000 --ood_retrained_lr 1e-5 --ood_retrained_lambda_sparse 5e-6 --ood_retrained_batch_size 64 --use_wandb --wandb_project GAE_OOD-Evaluation --wandb_name pythia-1.4b_transcoder_edgar_ood_retrained_lr1e-5_lambda5e-6_total4000000
