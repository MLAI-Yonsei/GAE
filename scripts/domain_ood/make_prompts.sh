export HF_HOME="${DATA_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${DATA_ROOT}/datasets_cache"
mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"

cd ${REPO_ROOT}

OOD_SET=${1:-patents} # patents, edgar
MODEL=${2:-gpt2}
OUT_DIR=${3:-${REPO_DATA}/domain_ood_prompts}
SEED=${4:-2026}
MAX_LEN=${5:-256}
N_TOTAL=${6:-50000}



python3 scripts/domain_ood/build_prompts.py \
    --out_dir "$OUT_DIR" \
    --seed "$SEED" \
    --max_len "$MAX_LEN" \
    --n_total "$N_TOTAL" \
    --model "$MODEL" \
    --ood_set "$OOD_SET"
