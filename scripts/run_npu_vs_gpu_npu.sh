#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/npu_vs_gpu.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

MODELS=($(jq -r '.models[]' "$CONFIG"))
BATCH_SIZES=($(jq -r '.batch_sizes[]' "$CONFIG"))

echo "========================================"
echo "Running NPU vs GPU Experiment (NPU Side)"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Models: ${MODELS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo ""

cd "$ROOT_DIR"

for model in "${MODELS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    echo "Processing: $model (bs=$batch_size)"
    echo "----------------------------------------"

    TEMP_CONFIG=$(mktemp)
    jq --arg model "$model" --argjson bs "$batch_size" \
      '.models = [$model] | .batch_sizes = [$bs]' "$CONFIG" > "$TEMP_CONFIG"

    python bench_npu.py experiment --config "$TEMP_CONFIG" --output-dir "$OUTPUT_DIR"

    rm "$TEMP_CONFIG"
    echo ""
  done
done

echo "========================================"
echo "NPU side complete!"
echo "Results saved to: $OUTPUT_DIR/npu_vs_gpu"
echo "========================================"
