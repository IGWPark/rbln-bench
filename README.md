# RBLN Benchmark Suite

Reproducible benchmarking framework for comparing RBLN NPU and GPU performance across different model sizes, batch configurations, and workloads.

## Overview

This benchmark suite enables:

1. **NPU Scaling Analysis**: Compare different model sizes on NPU with varying context lengths
2. **NPU vs GPU Comparison**: Direct performance comparison between NPU and GPU with different batch sizes

All results include comprehensive metrics:
- TTFT (Time To First Token)
- E2EL (End-to-End Latency)
- TPOT (Time Per Output Token)
- Throughput (tokens/s, requests/s)
- Hardware metrics (power, memory, utilization via rbln-mon)

## Repository Structure

```
rbln-bench/
├── rbln_bench/                      # Core modules
│   ├── benchmark.py                 # Benchmark engine
│   ├── compile.py                   # Model compilation
│   ├── models.py                    # Model registry
│   ├── monitor.py                   # Hardware monitoring
│   └── utils.py                     # Utilities
│
├── bench_npu.py                     # NPU benchmark script
├── bench_gpu.py                     # GPU benchmark script
│
├── configs/
│   ├── compile/
│   │   ├── batch1.json              # Batch size 1 compilation configs
│   │   └── batch_variants.json      # Multiple batch size configs
│   └── experiments/
│       ├── npu_scaling.json         # NPU model scaling experiment
│       └── npu_vs_gpu.json          # NPU vs GPU comparison
│
├── scripts/
│   ├── compile_batch_variants.sh    # Compile batch variants
│   ├── run_npu_scaling.sh           # Run NPU scaling experiment
│   ├── run_npu_vs_gpu_npu.sh        # Run NPU side of comparison
│   └── run_npu_vs_gpu_gpu.sh        # Run GPU side of comparison
│
├── compiled/                        # Compiled models
├── results/                         # Benchmark results
└── monitoring/                      # Hardware monitoring logs
```

## Installation

### Requirements

- Python 3.10+
- rebel-compiler 0.8.3 (for NPU compilation)
- RBLN 0.8.3 (vllm_rbln, optimum-rbln)
- rbln-mon (for hardware monitoring)

### Setup

```bash
# Clone repository
cd rbln-bench

# Install rebel-compiler (requires access to pypi.rbln.ai)
pip install -i https://pypi.rbln.ai/simple/ rebel-compiler==0.8.3

# Install core dependencies
pip install -r requirements.txt

# Install rbln-mon for hardware monitoring
pip install git+https://github.com/IGWPark/rbln-mon.git
```

## Quick Start

### 1. Compile Models

Compilation requires high RAM (128GB+ recommended). GPU is not required for compilation, only for GPU benchmarking.

```bash
# Compile all models with batch_size=1
./scripts/compile_batch_variants.sh configs/compile/batch1.json

# Compile batch size variants (4, 8, 16, 32) for 3 models
./scripts/compile_batch_variants.sh configs/compile/batch_variants.json
```

### 2. Transfer Models (Optional)

If compiling on a separate server, transfer models to the NPU server:

```bash
rsync -avz compiled/ npu-server:/path/to/rbln-bench/compiled/
```

### 3. Run Experiments

#### Experiment 1: NPU Scaling

```bash
./scripts/run_npu_scaling.sh
```

Tests 6 models (0.6B to 8B, including FP16/FP8/W8A16 quantization variants) with input lengths [1000, 2000, 4000, 8000] and 128 output tokens, batch_size=1.

Models: qwen3-0.6b, qwen3-1.7b, qwen3-4b, llama-3.1-8b, llama-3.1-8b-fp8, llama-3.1-8b-w8a16

Results → `results/npu_scaling/`

#### Experiment 2: NPU vs GPU

**NPU:**
```bash
./scripts/run_npu_vs_gpu_npu.sh
```

**GPU (requires NVIDIA GPU):**
```bash
./scripts/run_npu_vs_gpu_gpu.sh
```

Tests 3 models (Qwen3 0.6B/1.7B/4B) with batch sizes [1, 4, 8, 16, 32], input lengths [1000, 2000, 4000, 8000], 128 output tokens, and varying request loads to measure batching efficiency and throughput under different concurrency levels.

GPU benchmarks were conducted under two memory utilization settings to provide both an equal-capacity comparison with the NPU and a full-performance view:

- 0.33 utilization (~16 GB): Matches the NPU's memory capacity for an equal-capacity comparison.
  → Results stored with the `_33.json` suffix.

- 0.9 utilization (~43 GB): Represents the GPU's maximum practical performance.
  → Results stored with the `_90.json` suffix.

Results → `results/npu_vs_gpu/`

#### Experiment 3: Quantization (Optional)

```bash
./scripts/run_quantization_comparison.sh
```

Compares FP16 vs FP8 vs W8A16 quantization for Llama-3.1-8B with input lengths [1000, 2000, 4000, 8000] and 128 output tokens.

Results → `results/quantization_comparison/`

## Custom Benchmarks

Run individual benchmarks directly:

```bash
# Single benchmark (NPU)
python bench_npu.py single \
  --model qwen3-0.6b \
  --batch-size 1 \
  --input-len 1000 \
  --output-len 128 \
  --num-requests 10 \
  --output results/custom_test.json

# Single benchmark (GPU)
python bench_gpu.py single \
  --model qwen3-0.6b \
  --batch-size 4 \
  --input-len 1000 \
  --output-len 128 \
  --num-requests 10 \
  --output results/custom_test.json

# From config file
python bench_npu.py experiment \
  --config configs/experiments/npu_scaling.json \
  --output-dir results
```

## Available Models

### Models Used in Experiments

| Short Name | HuggingFace ID | Size | Used In |
|------------|---------------|------|---------|
| `qwen3-0.6b` | Qwen/Qwen3-0.6B | 0.6B | NPU Scaling, NPU vs GPU |
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | 1.7B | NPU Scaling, NPU vs GPU |
| `qwen3-4b` | Qwen/Qwen3-4B-Instruct-2507 | 4B | NPU Scaling, NPU vs GPU |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | NPU Scaling, Quantization |
| `llama-3.1-8b-fp8` | RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 | 8B (FP8) | NPU Scaling, Quantization |
| `llama-3.1-8b-w8a16` | RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 | 8B (W8A16) | NPU Scaling, Quantization |

### Additional Supported Models

Other models can be added to configs as needed:

| Short Name | HuggingFace ID | Size |
|------------|---------------|------|
| `qwen2.5-0.5b` | Qwen/Qwen2.5-0.5B-Instruct | 0.5B |
| `qwen2.5-1.5b` | Qwen/Qwen2.5-1.5B-Instruct | 1.5B |
| `gemma-2b` | google/gemma-2b-it | 2B |
| `phi-2` | microsoft/Phi-2 | 2.7B |
| `llama-3.2-3b` | meta-llama/Llama-3.2-3B-Instruct | 3B |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | 7B |
| `deepseek-r1-qwen3-8b` | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 8B |

## Results

Each benchmark produces a JSON file with:
- **Metrics**: TTFT, E2EL, TPOT with percentiles (p50/p90/p95/p99)
- **Throughput**: tokens/s, requests/s
- **Hardware**: utilization, memory, power, temperature (via rbln-mon)
- **Per-request data**: individual timings

Example: `results/npu_scaling/qwen3-0.6b/input_1000_output_128_bs_1_req_10.json`

## Configuration

Experiments are configured via JSON files in `configs/`:
- `compile/` - Model compilation settings
- `experiments/` - Benchmark workloads and parameters

Results are reproducible with the same configs and hardware.

## Notes

- All benchmarks use 128 output tokens for fair comparison
- Hardware monitoring via rbln-mon (disable with `--no-monitoring`)
- Automatically skips existing results - safe to re-run after interruptions
- Compilation may fail for large models/batch sizes (memory constraints)
