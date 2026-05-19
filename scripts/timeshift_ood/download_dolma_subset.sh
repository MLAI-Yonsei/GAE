#!/bin/bash
set -euo pipefail

# Minimal Dolma download/repair helper for local subset.
# Usage (download mode, default):
#   DATA_DIR=/path/to/dolma \
#   DOLMA_VERSION=v1_7 \
#   DOLMA_N_URLS=50 \
#   PARALLEL_DOWNLOADS=8 \
#   ./download_dolma_subset.sh
#
# Usage (repair mode: re-download only corrupted *.json.gz files in DATA_DIR):
#   MODE=repair \
#   DATA_DIR=/path/to/dolma \
#   DOLMA_VERSION=v1_7 \
#   ./download_dolma_subset.sh

DATA_DIR="${DATA_DIR:-${DATA_ROOT}/dataset_cache/dolma}"
DOLMA_VERSION="${DOLMA_VERSION:-v1_7}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-8}"
DOLMA_N_URLS="${DOLMA_N_URLS:-50}"
DOLMA_REPO="${DOLMA_REPO:-${DATA_ROOT}/dolma}"
MODE="${MODE:-download}"  # "download" or "repair"

mkdir -p "${DATA_DIR}"

if [ ! -d "${DOLMA_REPO}" ]; then
  echo "Cloning dolma repo to ${DOLMA_REPO}..."
  git clone https://huggingface.co/datasets/allenai/dolma "${DOLMA_REPO}"
fi

URLS_FILE="${DOLMA_REPO}/urls/${DOLMA_VERSION}.txt"
if [ ! -f "${URLS_FILE}" ]; then
  echo "URLs file not found: ${URLS_FILE}"
  exit 1
fi

if [ "${MODE}" = "download" ]; then
  echo "Downloading ${DOLMA_N_URLS} files to ${DATA_DIR} (parallel=${PARALLEL_DOWNLOADS})..."
  echo "Reading URLs from ${URLS_FILE}..."
  # Let wget print per-file progress; parallel downloads will interleave logs.
  head -n "${DOLMA_N_URLS}" "${URLS_FILE}" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -P "${DATA_DIR}"

  echo "Download finished. Example files:"
  ls -lh "${DATA_DIR}" | head -n 5
elif [ "${MODE}" = "repair" ]; then
  echo "Scanning ${DATA_DIR} for corrupted *.json.gz files..."
  BAD_LIST="$(mktemp)"

  shopt -s nullglob
  for f in "${DATA_DIR}"/*.json.gz; do
    # gzip -t exits non-zero on corruption; we invert the check.
    if ! gzip -t "$f" >/dev/null 2>&1; then
      echo "$(basename "$f")" >> "${BAD_LIST}"
    fi
  done
  shopt -u nullglob

  if [ ! -s "${BAD_LIST}" ]; then
    echo "No corrupted gzip files detected in ${DATA_DIR}."
    rm -f "${BAD_LIST}"
    exit 0
  fi

  total_corrupted=$(wc -l < "${BAD_LIST}")
  echo "Corrupted files detected (${total_corrupted}):"
  cat "${BAD_LIST}"

  idx=0
  while read -r fname; do
    [ -z "${fname}" ] && continue
    idx=$((idx + 1))
    # Find matching URL in the Dolma URL list.
    url="$(grep "/${fname}\$" "${URLS_FILE}" | head -n 1 || true)"
    if [ -z "${url}" ]; then
      echo "Warning: URL not found for ${fname} in ${URLS_FILE}, skipping."
      continue
    fi
    echo "[${idx}/${total_corrupted}] Redownloading ${fname} from ${url}"
    wget -O "${DATA_DIR}/${fname}" "${url}"
  done < "${BAD_LIST}"

  rm -f "${BAD_LIST}"
  echo "Repair finished."
else
  echo "Unsupported MODE: ${MODE} (expected 'download' or 'repair')"
  exit 1
fi
