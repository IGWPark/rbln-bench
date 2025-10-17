#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/configs/experiments/npu_vs_gpu.json"
OUTPUT_DIR="${1:-$ROOT_DIR/results}"

cd "$ROOT_DIR"

echo "========================================"
echo "Running NPU vs GPU Experiment (GPU Side)"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

echo "Run 1/2: Limited GPU memory (0.33 - parity with 16GB NPU)"
echo "========================================"
python bench_gpu.py experiment --config "$CONFIG" --output-dir "$OUTPUT_DIR" --gpu-memory-util 0.33

echo ""
echo "Run 2/2: Full GPU memory (0.9)"
echo "========================================"
python bench_gpu.py experiment --config "$CONFIG" --output-dir "$OUTPUT_DIR" --gpu-memory-util 0.9

echo ""
echo "========================================"
echo "GPU side complete!"
echo "Results saved to: $OUTPUT_DIR/npu_vs_gpu"
echo "  Files with _33.json suffix (limited)"
echo "  Files with _90.json suffix (full)"
echo "========================================"
