#!/usr/bin/env bash
# fetch_ettin.sh — bootstrap the Ettin reranker reference.
#
# cross-encoder/ettin-reranker-17m-v1 is a sentence-transformers MODULE CHAIN, not a
# *ForSequenceClassification checkpoint: `architectures` is the bare ModernBertModel and
# the classification head lives in three separate safetensors files
# (2_Dense / 3_LayerNorm / 4_Dense). Fetching model.safetensors alone gives a trunk with
# no head, which looks exactly like a loader bug — hence the explicit file list below.
#
# Plain curl rather than huggingface_hub: the repo is public and ungated, and this way
# the fetch does not depend on the .venv's hub/transformers version pin.
#
# Run from repo root:  bash scripts/fetch_ettin.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPO="cross-encoder/ettin-reranker-17m-v1"
DEST="testdata/ettin-reranker-17m"
BASE="https://huggingface.co/${REPO}/resolve/main"

FILES=(
    config.json
    tokenizer.json
    tokenizer_config.json
    sentence_bert_config.json
    config_sentence_transformers.json
    modules.json
    model.safetensors
    1_Pooling/config.json
    2_Dense/config.json
    2_Dense/model.safetensors
    3_LayerNorm/config.json
    3_LayerNorm/model.safetensors
    4_Dense/config.json
    4_Dense/model.safetensors
)

echo "==> downloading ${REPO} -> ${DEST}"
for f in "${FILES[@]}"; do
    mkdir -p "${DEST}/$(dirname "$f")"
    echo "    $f"
    curl -sSfL "${BASE}/${f}" -o "${DEST}/${f}"
done

echo "==> done"
find "$DEST" -type f | sort | xargs ls -lh | awk '{print $5, $9}'
