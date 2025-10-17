# 🔍 RBLN Benchmark Suite

A reproducible benchmarking framework to compare **RBLN NPU** and **GPU** performance across model sizes, batch configs, and workloads.

---

## 📈 Sample Results (NPU: RBLN-CA22)

### Peak Throughput by Context

| Model      |   Context |   Peak Throughput (tok/s) |   Peak Utilization (%) |   Power (W) |   Efficiency (tok/s/W) |
|:-----------|----------:|--------------------------:|-----------------------:|------------:|-----------------------:|
| Qwen3-0.6B |      1000 |                       117 |                   72.8 |        56   |                   2.08 |
| Qwen3-0.6B |      2000 |                        94 |                   77.2 |        57.6 |                   1.63 |
| Qwen3-0.6B |      4000 |                        64 |                   81.9 |        59   |                   1.08 |
| Qwen3-0.6B |      8000 |                        41 |                   85.7 |        60.9 |                   0.67 |
| Qwen3-1.7B |      1000 |                        88 |                   75.4 |        60.5 |                   1.45 |
| Qwen3-1.7B |      2000 |                        59 |                   80.1 |        61.6 |                   0.96 |
| Qwen3-1.7B |      4000 |                        41 |                   84.4 |        64.4 |                   0.63 |
| Qwen3-1.7B |      8000 |                        27 |                   86.9 |        64.8 |                   0.41 |
| Qwen3-4B   |      1000 |                        34 |                   76.4 |        61.7 |                   0.55 |
| Qwen3-4B   |      2000 |                        35 |                   86.6 |        64.6 |                   0.54 |
| Qwen3-4B   |      4000 |                        20 |                   91.2 |        65.7 |                   0.31 |
| Qwen3-4B   |      8000 |                        12 |                   90.2 |        65.7 |                   0.19 |

📊 Per-Batch Breakdown

| Model      |   Context |   Batch |   Throughput (tok/s) |   Utilization (%) |   Power (W) |   Efficiency (tok/s/W) |   Req/Batch |
|:-----------|----------:|--------:|---------------------:|------------------:|------------:|-----------------------:|------------:|
| Qwen3-0.6B |      1000 |       4 |                   86 |              38.5 |        51.8 |                   1.67 |           1 |
| Qwen3-0.6B |      1000 |       8 |                  100 |              66.4 |        42.2 |                   2.38 |           1 |
| Qwen3-0.6B |      1000 |      16 |                  117 |              72.8 |        56   |                   2.08 |           1 |
| Qwen3-0.6B |      1000 |      32 |                   82 |              74.1 |        55.1 |                   1.49 |           1 |
| Qwen3-0.6B |      2000 |       4 |                   70 |              70.4 |        58.2 |                   1.21 |           1 |
| Qwen3-0.6B |      2000 |       8 |                   83 |              76.7 |        57   |                   1.45 |           1 |
| Qwen3-0.6B |      2000 |      16 |                   94 |              77.2 |        57.6 |                   1.63 |           1 |
| Qwen3-0.6B |      2000 |      32 |                   72 |              74.6 |        56.4 |                   1.28 |           1 |
| Qwen3-0.6B |      4000 |       4 |                   49 |              82.8 |        58.5 |                   0.84 |           1 |
| Qwen3-0.6B |      4000 |       8 |                   56 |              78.1 |        59.1 |                   0.95 |           1 |
| Qwen3-0.6B |      4000 |      16 |                   64 |              81.9 |        59   |                   1.08 |           1 |
| Qwen3-0.6B |      4000 |      32 |                   47 |              78.9 |        58.9 |                   0.79 |           1 |
| Qwen3-0.6B |      8000 |       4 |                   32 |              86.5 |        59.2 |                   0.55 |           1 |
| Qwen3-0.6B |      8000 |       8 |                   36 |              85.9 |        60.2 |                   0.6  |           1 |
| Qwen3-0.6B |      8000 |      16 |                   41 |              85.7 |        60.9 |                   0.67 |           1 |
| Qwen3-0.6B |      8000 |      32 |                   33 |              87.3 |        60.8 |                   0.55 |           1 |
| Qwen3-1.7B |      1000 |       4 |                   50 |              52.2 |        52.3 |                   0.95 |           1 |
| Qwen3-1.7B |      1000 |       8 |                   70 |              79.4 |        59.8 |                   1.17 |           1 |
| Qwen3-1.7B |      1000 |      16 |                   88 |              75.4 |        60.5 |                   1.45 |           1 |
| Qwen3-1.7B |      2000 |       4 |                   42 |              79.1 |        62.4 |                   0.68 |           1 |
| Qwen3-1.7B |      2000 |       8 |                   57 |              77.6 |        62.6 |                   0.91 |           1 |
| Qwen3-1.7B |      2000 |      16 |                   59 |              80.1 |        61.6 |                   0.96 |           1 |
| Qwen3-1.7B |      4000 |       4 |                   30 |              79.1 |        63.5 |                   0.47 |           1 |
| Qwen3-1.7B |      4000 |       8 |                   41 |              84.4 |        64.4 |                   0.63 |           1 |
| Qwen3-1.7B |      4000 |      16 |                   38 |              83.5 |        64.1 |                   0.59 |           1 |
| Qwen3-1.7B |      8000 |       4 |                   22 |              84.2 |        63.9 |                   0.35 |           1 |
| Qwen3-1.7B |      8000 |       8 |                   27 |              86.9 |        64.8 |                   0.41 |           1 |
| Qwen3-1.7B |      8000 |      16 |                   24 |              87.6 |        64.2 |                   0.37 |           1 |
| Qwen3-4B   |      1000 |       4 |                   26 |              78.8 |        64.1 |                   0.41 |           1 |
| Qwen3-4B   |      1000 |       8 |                   34 |              76.4 |        61.7 |                   0.55 |           1 |
| Qwen3-4B   |      2000 |       4 |                   26 |              83.9 |        64.8 |                   0.4  |           1 |
| Qwen3-4B   |      2000 |       8 |                   35 |              86.6 |        64.6 |                   0.54 |           1 |
| Qwen3-4B   |      4000 |       4 |                   19 |              87.6 |        64.7 |                   0.3  |           1 |
| Qwen3-4B   |      4000 |       8 |                   20 |              91.2 |        65.7 |                   0.31 |           1 |
| Qwen3-4B   |      8000 |       4 |                   12 |              90.2 |        65.7 |                   0.19 |           1 |
| Qwen3-4B   |      8000 |       8 |                   12 |              93.9 |        65.7 |                   0.19 |           1 |

---

## 🏗️ Project Structure

```

rbln-bench/
├── rbln_bench/               # Core modules (benchmarking, compilation, monitoring)
├── bench_npu.py              # NPU benchmarking entry point
├── bench_gpu.py              # GPU benchmarking entry point
├── configs/
│   ├── compile/               # Compilation configurations
│   └── experiments/            # Workload definitions
├── scripts/                   # Shell wrappers for running experiments
├── compiled/                   # Generated model artifacts
├── results/                    # Benchmark outputs
└── monitoring/                 # Telemetry logs

````

---

## 🛠️ Installation & Getting Started

### Requirements

- Python 3.10+  
- `rebel-compiler` 0.8.3  
- `RBLN` 0.8.3 (including `vllm_rbln`, `optimum-rbln`)  
- `rbln-mon` for hardware telemetry  

```bash
git clone <repo-url>
cd rbln-bench

pip install -i https://pypi.rbln.ai/simple/ rebel-compiler==0.8.3  
pip install -r requirements.txt  
pip install git+https://github.com/IGWPark/rbln-mon.git  
````

### Run Benchmarks

1. **Compile models**

   ```bash
   ./scripts/compile_batch_variants.sh configs/compile/batch_variants.json
   ```

2. **Run experiments**

   * NPU scaling: `./scripts/run_npu_scaling.sh`
   * NPU vs GPU:

     ```bash
     ./scripts/run_npu_vs_gpu_npu.sh
     ./scripts/run_npu_vs_gpu_gpu.sh
     ```
   * Quantization comparison (optional):
     `./scripts/run_quantization_comparison.sh`

3. **Inspect results**
   Output JSONs and telemetry logs are in `results/` and `monitoring/`.

### Example: Single Run

```bash
python bench_npu.py single \
  --model qwen3-0.6b \
  --batch-size 1 \
  --input-len 1000 \
  --output-len 128 \
  --num-requests 10 \
  --output results/custom.json
```

---

## 📦 Supported Models (in experiments)

* `qwen3-0.6b`, `qwen3-1.7b`, `qwen3-4b`
* `llama-3.1-8b`, `llama-3.1-8b-fp8`, `llama-3.1-8b-w8a16`
* Easily extendable via JSON configs

---

## 📌 Notes & Constraints

* All benchmarks use **128 output tokens** for consistency
* Telemetry (power, temp, etc.) via `rbln-mon` — disable with `--no-monitoring` flag
* The runner avoids re-running already completed experiments
* Some model + batch combinations may fail compilation due to memory limits
