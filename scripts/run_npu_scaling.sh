#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/npu_scaling.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

MODELS=($(jq -r '.models[]' "$CONFIG"))

echo "========================================"
echo "Running NPU Scaling Experiment"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Models: ${MODELS[*]}"
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
echo "NPU Scaling Experiment Complete!"
echo "Results saved to: $OUTPUT_DIR/npu_scaling"
echo "========================================"
