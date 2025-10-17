#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/compile/batch_variants.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

echo "Compiling batch variant models from: $CONFIG_PATH"
echo "================================================"
echo ""

jq -c '.[]' "$CONFIG_PATH" | while read -r job; do
    echo "Compiling: $job"
    python -m rbln_bench.compile "$job" || echo "⚠️  Compilation failed (possibly OOM), continuing..."
    echo ""
done

echo "================================================"
echo "Compilation complete!"
