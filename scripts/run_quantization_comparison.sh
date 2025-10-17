#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/quantization_comparison.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

echo "========================================"
echo "Running Quantization Comparison"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Comparing: FP16 vs FP8 vs W8A16"
echo ""

cd "$ROOT_DIR"
python bench_npu.py experiment --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Quantization Comparison Complete!"
echo "Results saved to: $OUTPUT_DIR/quantization_comparison"
echo "========================================"
