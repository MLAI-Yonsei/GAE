#!/usr/bin/env bash
# GAE+ on GPT-2 Small Transcoder x MedQuAD with 128 adaptation prompts:
# no shrinkage / decoder-target shrinkage / ID-target shrinkage / decoder-target shrinkage with both bias families.
# Usage: bash scripts/gae_plus/medquad_gpt2_tc.sh          (env: PY=python, GPUS="0 1 2 3")
# Inputs: artifacts/checkpoints/... (ID transcoder), data/medical/gpt2/medquad/{adaptation_n128.json,evaluation.jsonl},
#         data/id/owt_heldout_2000.jsonl (OpenWebText held-out texts, only for the ID target).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}; GPUS=(${GPUS:-0 1 2 3})
D=data/medical/gpt2/medquad; OUT=results/gae_plus/medquad_gpt2_tc; mkdir -p $OUT
CKPT=artifacts/checkpoints/20261102T080514Z_gpt2_tc_l2mean/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
run() { name=$1; gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY gae_plus.py --model gpt2 --explainer transcoder --checkpoint $CKPT \
    --adaptation $D/adaptation_n128.json --evaluation $D/evaluation.jsonl --n-fit 128 \
    --rank 32 --lam-gae 0.2 --lam-geom 0.1 --lam-id 0 --output $OUT/$name "$@" > $OUT/$name.log 2>&1 & }
run plain               ${GPUS[0]} --shrink none
run shrink_dec          ${GPUS[1]} --shrink decoder
run shrink_id           ${GPUS[2]} --shrink id --id-data data/id/owt_heldout_2000.jsonl
run shrink_dec_bothbias ${GPUS[3]} --shrink decoder --bias both
wait
for n in plain shrink_dec shrink_id shrink_dec_bothbias; do echo "== $n"; grep -E "^(eta=|\[eval\])" $OUT/$n.log; done
