#!/bin/bash
# Retrain the GPT-2 Small ID transcoder with the paper setting
# (scripts/train_id/train_erm.sh), reading only the local OpenWebText shards
# from scripts/data/download_owt_subset.sh.
# Usage: bash scripts/train_id/train_gpt2_transcoder.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- settings ----
GPU=${1:-1}
SEED=2026
LR=2e-5
L1=6e-5
TOTAL_TOKENS=60000000
DICT_SIZE=24576
CONTEXT=128
TRAIN_BS=256
STORE_BS=32
N_BUFFER=128
WARMUP=5000
RUN_NAME=gpt2_transcoder_erm_lr${LR}_lambda${L1}

# ---- paths (override via env) ----
export DATA_ROOT=${DATA_ROOT:-/data1/heedong/gae_data}  # HF caches, data, checkpoints
CACHE_ROOT=${CACHE_ROOT:-/data1/heedong/.cache}
export UV_CACHE_DIR=$CACHE_ROOT/uv UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv-python
export TMPDIR=$CACHE_ROOT/tmp XDG_CACHE_HOME=$CACHE_ROOT
export PATH="$HOME/.local/bin:$PATH"
OWT=$DATA_ROOT/openwebtext/79d93d786212f7344586290adb811d4ae6a1762c
TRAIN_FILES=$OWT/train-00000-of-00080.parquet,$OWT/train-00001-of-00080.parquet
OUT=results/train_id/$RUN_NAME
mkdir -p "$TMPDIR" "$OUT"

# ---- data + environment (authors' env: Python 3.10 + requirements.lock) ----
bash scripts/data/download_owt_subset.sh
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements.lock

# ---- run info ----
{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "commit: $(git rev-parse HEAD)"
    echo "git status:"; git status --short
    .venv/bin/python -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__)"
    nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader
} > "$OUT/run_info.txt"
git diff > "$OUT/uncommitted.diff"
cp "$0" "$OUT/"

# ---- train ----
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python train_explainers.py \
    --model gpt2 \
    --explainer transcoder \
    --objective ERM \
    --dataset_path "$TRAIN_FILES" \
    --dict_size_k $DICT_SIZE \
    --context_size $CONTEXT \
    --train_batch_size $TRAIN_BS \
    --store_batch_size $STORE_BS \
    --n_batches_in_buffer $N_BUFFER \
    --total_tokens $TOTAL_TOKENS \
    --lr $LR \
    --l1_coefficient $L1 \
    --lr_scheduler_name constantwithwarmup \
    --lr_warm_up_steps $WARMUP \
    --b_dec_init_method mean \
    --checkpoint_path "$DATA_ROOT/checkpoints" \
    --wandb_name "$RUN_NAME" \
    --device cuda \
    --seed $SEED \
    2>&1 | tee "$OUT/train.log"
