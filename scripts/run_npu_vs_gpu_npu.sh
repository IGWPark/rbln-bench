#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/npu_vs_gpu.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

echo "========================================"
echo "Running NPU vs GPU Experiment (NPU Side)"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$ROOT_DIR"
python bench_npu.py experiment --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "NPU side complete!"
echo "Results saved to: $OUTPUT_DIR/npu_vs_gpu"
echo "========================================"
