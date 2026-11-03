#!/bin/bash
# GAE+ bounds grid on GPT-2 + Transcoder + HaluEval: a_N/b_N by Chebyshev or best inequality,
# sample split off and on (both run), one bound on H and one on g, and (split on) element-wise
# Bonferroni bounds in the step and full-GAE bases. Selected eta is scored
# on held-out test activations.
# Paper Table 9 setting, lambda_id = 0 for GAE+. Writes results/gae_plus/<run>/bounds_grid_split_{off,on}.json.
# Usage: bash scripts/gae_plus/bounds_grid.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- settings ----
GPU=${1:-1}
MODEL=gpt2
OOD_SET=halu_eval
R=32
LAM_GEOM=0.1
LAM_PRES=0.2          # = GAE_AC_LAMBDA_GAE (paper lambda_pres)
LAM_ID=0              # GAE+ path: prior = Step 1 decoder only
FIT_SAMPLES=2048
DELTA=0.05
N_FIT=2000
N_TEST=1000
SPLITS="off on"
SUBSPACE_MATRIX=centered
STEP1_MATCH_NORMS=1
RUN_NAME=bounds_grid_${MODEL}_${OOD_SET}_r${R}

# ---- paths (override via env) ----
export DATA_ROOT=${DATA_ROOT:-/data1/heedong/gae_data}
CACHE_ROOT=${CACHE_ROOT:-/data1/heedong/.cache}
export UV_CACHE_DIR=$CACHE_ROOT/uv UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv-python
export TMPDIR=$CACHE_ROOT/tmp XDG_CACHE_HOME=$CACHE_ROOT
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1   # live log through tee
OWT=$DATA_ROOT/openwebtext/79d93d786212f7344586290adb811d4ae6a1762c
export GAE_ID_DATA_FILES=$OWT/train-00079-of-00080.parquet   # held-out ID shard
CKPT=$DATA_ROOT/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
OUT=results/gae_plus/$RUN_NAME
mkdir -p "$TMPDIR" "$OUT"

# ---- GAE config (read by config.py at import) ----
export GAE_AC_LAMBDA_GEOM=$LAM_GEOM GAE_AC_LAMBDA_GAE=$LAM_PRES GAE_AC_LAMBDA_ID=$LAM_ID
export GAE_AC_FIT_SAMPLES=$FIT_SAMPLES GAE_AC_DECODER_MIX_PRIOR=0
export GAE_SUBSPACE_MATRIX=$SUBSPACE_MATRIX GAE_STEP1_MATCH_NORMS=$STEP1_MATCH_NORMS
export N_COV=$((N_FIT + N_TEST))   # OOD activations collected = fit + test

# ---- data + environment (authors' env: Python 3.10 + requirements.lock) ----
bash scripts/data/download_owt_subset.sh
[ -f "$CKPT" ] || { echo "missing checkpoint: run scripts/train_id/train_gpt2_transcoder.sh"; exit 1; }
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

# ---- run (split off and on) ----
for SPLIT in $SPLITS; do
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python gae_plus_bounds_grid.py \
        --model $MODEL \
        --checkpoint "$CKPT" \
        --ood_set $OOD_SET \
        --r $R \
        --delta $DELTA \
        --split $SPLIT \
        --n_fit $N_FIT \
        --n_test $N_TEST \
        --out "$OUT" \
        2>&1 | tee "$OUT/bounds_grid_split_${SPLIT}.log"
done
