#!/bin/bash
# Evaluate ID faithfulness for all models (GPT-2, Pythia-410M, Pythia-1.4B)
# Purpose: Diagnose whether Pythia-1.4B's low OOD performance stems from
#          checkpoint quality or genuine distribution shift effects.

export HF_HOME="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"

DEVICE=${1:-0}
N_EVAL=${2:-500}

for MODEL in gpt2 pythia-410m pythia-1.4b; do
    if [ "$MODEL" = "gpt2" ]; then
        CHECKPOINT="${REPO_DATA}/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt"
        BATCH_SIZE=256
    elif [ "$MODEL" = "pythia-410m" ]; then
        CHECKPOINT="${REPO_DATA}/checkpoints/pythia-410m_transcoder_erm_lr2e-5_lambda5.5e-5/final_sparse_autoencoder_pythia-410m_blocks.15.ln2.hook_normalized_32768.pt"
        BATCH_SIZE=256
    elif [ "$MODEL" = "pythia-1.4b" ]; then
        CHECKPOINT="${REPO_DATA}/checkpoints/pythia-1.4b_transcoder_erm_lr2e-5_lambda2e-5/final_sparse_autoencoder_pythia-1.4b_blocks.15.ln2.hook_normalized_65536.pt"
        BATCH_SIZE=64
    fi

    echo "=============================================="
    echo "Evaluating ID faithfulness: $MODEL"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=$DEVICE python3 scripts/train_id/eval_id_faithfulness.py \
        --model $MODEL \
        --explainer transcoder \
        --checkpoint $CHECKPOINT \
        --n_eval $N_EVAL \
        --batch_size $BATCH_SIZE \
        --output_dir results

done

echo ""
echo "Done. Compare results:"
echo "  results/id_faithfulness_gpt2_transcoder.json"
echo "  results/id_faithfulness_pythia-410m_transcoder.json"
echo "  results/id_faithfulness_pythia-1.4b_transcoder.json"
