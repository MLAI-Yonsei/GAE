#!/bin/bash
# Download only the OpenWebText shards needed for GPT-2 ID data
# (pinned revision, sha256-checked). Re-running skips verified files.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data1/heedong/gae_data}
REVISION=79d93d786212f7344586290adb811d4ae6a1762c  # Skylion007/openwebtext
OUT_DIR=${DATA_ROOT}/openwebtext/${REVISION}

# name  sha256
# 00000-00001: transcoder training (read in order, same as a full download)
# 00079: held-out shard for ID activations in run_experiment.py
SHARDS="
train-00000-of-00080.parquet caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e
train-00001-of-00080.parquet 70307b7749e10cb0bfb895fde024dd82d96e6bb2710bca1886b952abe45bf3ea
train-00079-of-00080.parquet 80f15bb3dc64287827248d2a7a999ec986ecf15f07a79ed8c2e2b366e8046ae2
"

mkdir -p "$OUT_DIR"
echo "$SHARDS" | while read -r name sha; do
    [ -z "$name" ] && continue
    f="$OUT_DIR/$name"
    if [ -f "$f" ] && echo "$sha  $f" | sha256sum -c --quiet; then
        echo "ok (cached): $f"
        continue
    fi
    curl -fL --retry 4 -o "$f.part" \
        "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/$REVISION/plain_text/$name"
    mv "$f.part" "$f"
    echo "$sha  $f" | sha256sum -c
done
