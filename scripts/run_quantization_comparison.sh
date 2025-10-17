#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/quantization_comparison.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

MODELS=($(jq -r '.models[]' "$CONFIG"))

echo "========================================"
echo "Running Quantization Comparison"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Models: ${MODELS[*]}"
echo ""
echo "Comparing: FP16 vs FP8 vs W8A16"
echo ""

cd "$ROOT_DIR"

for model in "${MODELS[@]}"; do
  echo "Processing model: $model"
  echo "----------------------------------------"

  TEMP_CONFIG=$(mktemp)
  jq --arg model "$model" '.models = [$model]' "$CONFIG" > "$TEMP_CONFIG"

  python bench_npu.py experiment --config "$TEMP_CONFIG" --output-dir "$OUTPUT_DIR"

  rm "$TEMP_CONFIG"
  echo ""
done

echo "========================================"
echo "Quantization Comparison Complete!"
echo "Results saved to: $OUTPUT_DIR/quantization_comparison"
echo "========================================"
