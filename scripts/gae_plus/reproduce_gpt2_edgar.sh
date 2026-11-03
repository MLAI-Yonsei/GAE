#!/bin/bash
# Reproduce paper Table 2 (GPT-2 + Transcoder + Edgar): Fixed vs GAE with the
# Table 9 setting, using the stock run_experiment.py on the retrained checkpoint.
# Needs prompts from scripts/gae_plus/build_edgar_prompts.sh.
# Usage: bash scripts/gae_plus/reproduce_gpt2_edgar.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- settings (paper Table 9: r=4, lambda_geom=0.1, lambda_pres=0.2, alpha=0) ----
GPU=${1:-1}
MODEL=gpt2
OOD_SET=edgar
N_EVAL=1000
R=4
LAM_GEOM=0.1
LAM_PRES=0.2          # = GAE_AC_LAMBDA_GAE
LAM_ID=0.05           # original code default (the paper's Eq. 11 has no lambda_id term)
ALPHA=0               # = GAE_AC_DECODER_MIX_PRIOR
FIT_SAMPLES=2048
SUBSPACE_MATRIX=centered
STEP1_MATCH_NORMS=1
RUN_NAME=reproduce_${MODEL}_${OOD_SET}_r${R}_lamid${LAM_ID}

# ---- paths (override via env) ----
export DATA_ROOT=${DATA_ROOT:-/data1/heedong/gae_data}
export REPO_DATA=$DATA_ROOT
CACHE_ROOT=${CACHE_ROOT:-/data1/heedong/.cache}
export UV_CACHE_DIR=$CACHE_ROOT/uv UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv-python
export TMPDIR=$CACHE_ROOT/tmp XDG_CACHE_HOME=$CACHE_ROOT
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
OWT=$DATA_ROOT/openwebtext/79d93d786212f7344586290adb811d4ae6a1762c
export GAE_ID_DATA_FILES=$OWT/train-00079-of-00080.parquet   # held-out ID shard
PROMPTS_DIR=$DATA_ROOT/domain_ood_prompts
CKPT=$DATA_ROOT/checkpoints/gpt2_transcoder_erm_lr2e-5_lambda6e-5/final_sparse_autoencoder_gpt2_blocks.8.ln2.hook_normalized_24576.pt
OUT=results/gae_plus/$RUN_NAME
mkdir -p "$TMPDIR" "$OUT"

# ---- GAE config (read by config.py at import) ----
export GAE_AC_LAMBDA_GEOM=$LAM_GEOM GAE_AC_LAMBDA_GAE=$LAM_PRES GAE_AC_LAMBDA_ID=$LAM_ID
export GAE_AC_FIT_SAMPLES=$FIT_SAMPLES GAE_AC_DECODER_MIX_PRIOR=$ALPHA
export GAE_SUBSPACE_MATRIX=$SUBSPACE_MATRIX GAE_STEP1_MATCH_NORMS=$STEP1_MATCH_NORMS
export DOMAIN_EVAL_DISJOINT=1   # eval prompts disjoint from the GAE fit set (paper App. D.4)

# ---- data + environment (authors' env: Python 3.10 + requirements.lock) ----
bash scripts/data/download_owt_subset.sh
[ -f "$PROMPTS_DIR/${OOD_SET}_${MODEL}.jsonl" ] || { echo "missing prompts: run scripts/gae_plus/build_edgar_prompts.sh"; exit 1; }
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
    echo "prompts:"; sha256sum "$PROMPTS_DIR/${OOD_SET}_${MODEL}.jsonl"
} > "$OUT/run_info.txt"
git diff > "$OUT/uncommitted.diff"
cp "$0" "$OUT/"

# ---- run (fresh activations, no cache) ----
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_experiment.py domain \
    --model $MODEL \
    --explainer transcoder \
    --ood_set $OOD_SET \
    --checkpoint "$CKPT" \
    --output_dir "$OUT" \
    --domain_ood_jsonl_dir "$PROMPTS_DIR" \
    --n_eval $N_EVAL \
    --baselines fixed gae \
    --gae_r $R \
    --gae_decoder_mode affine \
    --force_recompute \
    2>&1 | tee "$OUT/run.log"
