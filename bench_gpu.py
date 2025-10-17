#!/usr/bin/env python3
"""
GPU Benchmark CLI - Run benchmarks on GPU using vLLM directly.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from rbln_bench.benchmark import run_benchmark, save_benchmark_result
from rbln_bench.models import get_model_info
from rbln_bench.monitor import format_monitoring_metrics

try:
    from rbln_mon import track
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def get_memory_suffix(gpu_memory_util: float) -> str:
    return f"_{int(gpu_memory_util * 100)}"


def build_result_metadata(
    model_short_name: str,
    hf_model_id: str,
    batch_size: int,
    gpu_memory_util: float,
    experiment_name: str = None,
) -> Dict[str, Any]:
    """Build model and hardware metadata for result."""
    try:
        model_info = get_model_info(model_short_name)
    except ValueError:
        model_info = {"size": "unknown"}

    metadata = {
        "experiment": experiment_name or "single_run",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": {
            "name": model_short_name,
            "hf_id": hf_model_id,
            "size_params": model_info.get("size", "unknown"),
            "compilation": {
                "batch_size": batch_size,
                "backend": "gpu",
            }
        },
        "hardware": {
            "backend": "gpu",
            "device_name": "NVIDIA",
            "device_id": 0,
            "gpu_memory_utilization": gpu_memory_util,
        }
    }

    return metadata


async def run_single_benchmark(
    hf_model_id: str,
    model_short_name: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    gpu_memory_util: float = 0.9,
    output_path: Path = None,
    experiment_name: str = None,
    enable_monitoring: bool = True,
) -> Dict[str, Any]:
    """Run a single GPU benchmark with optional monitoring."""

    print(f"\n{'='*70}")
    print(f"Running GPU benchmark: {model_short_name}")
    print(f"Workload: input={input_len}, output={output_len}, requests={num_requests}, batch_size={batch_size}")
    print(f"GPU memory utilization: {gpu_memory_util}")
    print(f"{'='*70}\n")

    monitoring_data = None

    if MONITORING_AVAILABLE and enable_monitoring:
        @track(
            backend='gpu',
            device_ids=[0],
            save=True,
            save_dir='monitoring',
            metadata={
                'experiment': experiment_name or 'single_run',
                'model': model_short_name,
                'batch_size': batch_size,
                'input_len': input_len,
                'output_len': output_len,
            }
        )
        async def run_with_monitoring():
            return await run_benchmark(
                model_path=hf_model_id,
                num_requests=num_requests,
                input_len=input_len,
                output_len=output_len,
                max_model_len=8192,
                max_num_seqs=batch_size,
                max_num_batched_tokens=8192,
                gpu_memory_utilization=gpu_memory_util,
            )

        bench_result = await run_with_monitoring()
        tracker = getattr(run_with_monitoring, "last_tracker", None)
        if tracker is not None:
            monitoring_data = tracker.export()
    else:
        bench_result = await run_benchmark(
            model_path=hf_model_id,
            num_requests=num_requests,
            input_len=input_len,
            output_len=output_len,
            max_model_len=8192,
            max_num_seqs=batch_size,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=gpu_memory_util,
        )

    metadata = build_result_metadata(model_short_name, hf_model_id, batch_size, gpu_memory_util, experiment_name)

    result = {
        **metadata,
        **bench_result,
        "monitoring": format_monitoring_metrics(monitoring_data),
    }

    if output_path:
        save_benchmark_result(result, output_path)

    return result


async def run_experiment(config_path: Path, output_dir: Path, gpu_memory_util: float = 0.9):
    """Run full experiment from config file."""
    with open(config_path) as f:
        config = json.load(f)

    experiment_name = config["name"]
    models = config["models"]
    workloads = config["workloads"]
    batch_sizes = config.get("batch_sizes", [config.get("batch_size", 1)])
    request_patterns = config.get("request_patterns", {})

    print(f"\n{'='*70}")
    print(f"Running GPU experiment: {experiment_name}")
    print(f"Description: {config['description']}")
    print(f"GPU memory utilization: {gpu_memory_util}")
    print(f"Models: {len(models)}, Workloads: {len(workloads)}, Batch sizes: {batch_sizes}")
    print(f"{'='*70}\n")

    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        try:
            model_info = get_model_info(model_name)
            hf_model_id = model_info["hf_id"]

            for batch_size in batch_sizes:
                model_output_dir = output_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)

                # Get request counts for this batch size
                if request_patterns:
                    request_counts = request_patterns.get(str(batch_size), [batch_size])
                else:
                    # Legacy format: workloads contain num_requests
                    request_counts = [workload.get("num_requests", 10) for workload in workloads]
                    request_counts = list(set(request_counts))  # Remove duplicates

                for workload in workloads:
                    input_len = workload["input_len"]
                    output_len = workload["output_len"]

                    # Handle both old and new config formats
                    if "num_requests" in workload:
                        # Old format: use num_requests from workload
                        requests_to_test = [workload["num_requests"]]
                    else:
                        # New format: use request_patterns
                        requests_to_test = request_counts

                    for num_requests in requests_to_test:
                        mem_suffix = get_memory_suffix(gpu_memory_util)
                        output_filename = f"gpu_input_{input_len}_output_{output_len}_bs_{batch_size}_req_{num_requests}{mem_suffix}.json"
                        output_path = model_output_dir / output_filename

                        if output_path.exists():
                            print(f"⏭️  Skipping {model_name} (bs={batch_size}, input={input_len}, req={num_requests}): Result already exists")
                            continue

                        try:
                            await run_single_benchmark(
                                hf_model_id=hf_model_id,
                                model_short_name=model_name,
                                batch_size=batch_size,
                                input_len=input_len,
                                output_len=output_len,
                                num_requests=num_requests,
                                gpu_memory_util=gpu_memory_util,
                                output_path=output_path,
                                experiment_name=experiment_name,
                            )
                        except Exception as e:
                            print(f"❌ Error running benchmark for {model_name}: {e}")
                            continue

        except ValueError as e:
            print(f"⚠️  Skipping {model_name}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models on GPU using vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    single_parser = subparsers.add_parser("single", help="Run single benchmark")
    single_parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., qwen3-0.6b)")
    single_parser.add_argument("--batch-size", type=int, default=1, help="Batch size (max concurrent sequences)")
    single_parser.add_argument("--input-len", type=int, required=True, help="Input context length")
    single_parser.add_argument("--output-len", type=int, required=True, help="Output token length")
    single_parser.add_argument("--num-requests", type=int, required=True, help="Number of requests")
    single_parser.add_argument("--gpu-memory-util", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    single_parser.add_argument("--output", type=str, help="Output JSON path")
    single_parser.add_argument("--no-monitoring", action="store_true", help="Disable hardware monitoring")

    experiment_parser = subparsers.add_parser("experiment", help="Run full experiment from config")
    experiment_parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    experiment_parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    experiment_parser.add_argument("--gpu-memory-util", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")

    args = parser.parse_args()

    if args.command == "single":
        model_info = get_model_info(args.model)
        hf_model_id = model_info["hf_id"]
        output_path = Path(args.output) if args.output else None

        result = asyncio.run(run_single_benchmark(
            hf_model_id=hf_model_id,
            model_short_name=args.model,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            num_requests=args.num_requests,
            gpu_memory_util=args.gpu_memory_util,
            output_path=output_path,
            enable_monitoring=not args.no_monitoring,
        ))

        print("\n" + "="*70)
        print("BENCHMARK RESULTS")
        print("="*70)
        print(json.dumps(result, indent=2))
        print("="*70)

    elif args.command == "experiment":
        asyncio.run(run_experiment(
            config_path=Path(args.config),
            output_dir=Path(args.output_dir),
            gpu_memory_util=args.gpu_memory_util,
        ))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
