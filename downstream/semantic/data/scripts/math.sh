#!/bin/bash
# Download DeepMind Mathematics Dataset v1.0
set -e

DATA_DIR="$(dirname "${BASH_SOURCE[0]}")/../datasets/deepmind_math"
mkdir -p "${DATA_DIR}" && cd "${DATA_DIR}"

[ -d "mathematics_dataset-v1.0" ] && echo "Dataset already exists" && exit 0

curl -LO https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz
tar -xzf mathematics_dataset-v1.0.tar.gz
rm mathematics_dataset-v1.0.tar.gz

echo "Done: ${DATA_DIR}/mathematics_dataset-v1.0"
