#!/bin/bash
# Build the Edgar domain-OOD prompts used by `run_experiment.py domain`
# (repo flow: scripts/domain_ood/make_prompts.sh defaults), with the dataset revision pinned.
# Writes $DATA_ROOT/domain_ood_prompts/edgar_gpt2.jsonl (+ sha256).
# Usage: bash scripts/gae_plus/build_edgar_prompts.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- settings (make_prompts.sh defaults) ----
MODEL=gpt2
OOD_SET=edgar
SEED=2026
MAX_LEN=256
N_TOTAL=50000
EDGAR_REVISION=93e3b9fdb866c3cc1455b7a4a298648661b34bc5   # jlohding/sp500-edgar-10k

# ---- paths (override via env) ----
export DATA_ROOT=${DATA_ROOT:-/data1/heedong/gae_data}
CACHE_ROOT=${CACHE_ROOT:-/data1/heedong/.cache}
export HF_HOME=$DATA_ROOT/hf_cache HUGGINGFACE_HUB_CACHE=$DATA_ROOT/hf_cache HF_DATASETS_CACHE=$DATA_ROOT/datasets_cache
export UV_CACHE_DIR=$CACHE_ROOT/uv UV_PYTHON_INSTALL_DIR=$CACHE_ROOT/uv-python
export TMPDIR=$CACHE_ROOT/tmp XDG_CACHE_HOME=$CACHE_ROOT
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
OUT_DIR=$DATA_ROOT/domain_ood_prompts
mkdir -p "$TMPDIR" "$OUT_DIR"

# ---- environment (authors' env: Python 3.10 + requirements.lock) ----
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
uv pip sync --python .venv/bin/python requirements.lock

# ---- build ----
.venv/bin/python scripts/domain_ood/build_prompts.py \
    --out_dir "$OUT_DIR" \
    --seed $SEED \
    --max_len $MAX_LEN \
    --n_total $N_TOTAL \
    --model $MODEL \
    --ood_set $OOD_SET \
    --edgar_revision $EDGAR_REVISION \
    2>&1 | tee "$OUT_DIR/build_${OOD_SET}_${MODEL}.log"

sha256sum "$OUT_DIR/${OOD_SET}_${MODEL}.jsonl" | tee "$OUT_DIR/${OOD_SET}_${MODEL}.jsonl.sha256"
wc -l "$OUT_DIR/${OOD_SET}_${MODEL}.jsonl"
