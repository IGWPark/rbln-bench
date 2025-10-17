#!/usr/bin/env python3
"""
NPU Benchmark CLI - Run benchmarks on RBLN compiled models.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from rbln_bench.benchmark import run_benchmark, save_benchmark_result
from rbln_bench.models import resolve_model_path, get_model_info
from rbln_bench.monitor import format_monitoring_metrics

try:
    from rbln_mon import track
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


def build_result_metadata(
    model_short_name: str,
    model_path: Path,
    batch_size: int,
    experiment_name: str = None,
) -> Dict[str, Any]:
    """Build model and hardware metadata for result."""
    try:
        model_info = get_model_info(model_short_name)
    except ValueError:
        model_info = {"hf_id": "unknown", "size": "unknown"}

    rbln_config_path = model_path / "rbln_config.json"
    if rbln_config_path.exists():
        with open(rbln_config_path) as f:
            rbln_config = json.load(f)
    else:
        rbln_config = {}

    metadata = {
        "experiment": experiment_name or "single_run",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": {
            "name": model_short_name,
            "hf_id": model_info.get("hf_id", "unknown"),
            "size_params": model_info.get("size", "unknown"),
            "compiled_path": str(model_path),
            "compilation": {
                "batch_size": batch_size,
                "max_seq_len": rbln_config.get("max_seq_len", "unknown"),
                "kvcache_partition_len": rbln_config.get("kvcache_partition_len", "unknown"),
                "tensor_parallel_size": rbln_config.get("_compile_cfgs", [{}])[0].get("tensor_parallel_size", 1) if rbln_config.get("_compile_cfgs") else 1,
                "attn_impl": rbln_config.get("attn_impl", "unknown"),
                "npu": rbln_config.get("_compile_cfgs", [{}])[0].get("npu", "RBLN-CA22") if rbln_config.get("_compile_cfgs") else "RBLN-CA22",
            }
        },
        "hardware": {
            "backend": "npu",
            "device_name": "RBLN-CA22",
            "device_id": 0,
        }
    }

    return metadata


async def run_single_benchmark(
    model_path: Path,
    model_short_name: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_requests: int,
    output_path: Path = None,
    experiment_name: str = None,
    enable_monitoring: bool = True,
) -> Dict[str, Any]:
    """Run a single benchmark with optional monitoring."""

    print(f"\n{'='*70}")
    print(f"Running benchmark: {model_short_name} (bs={batch_size})")
    print(f"Workload: input={input_len}, output={output_len}, requests={num_requests}")
    print(f"{'='*70}\n")

    monitoring_data = None

    if MONITORING_AVAILABLE and enable_monitoring:
        @track(
            backend='npu',
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
                model_path=str(model_path),
                num_requests=num_requests,
                input_len=input_len,
                output_len=output_len,
            )

        bench_result = await run_with_monitoring()
        tracker = getattr(run_with_monitoring, "last_tracker", None)
        if tracker is not None:
            monitoring_data = tracker.export()
    else:
        bench_result = await run_benchmark(
            model_path=str(model_path),
            num_requests=num_requests,
            input_len=input_len,
            output_len=output_len,
        )

    metadata = build_result_metadata(model_short_name, model_path, batch_size, experiment_name)

    result = {
        **metadata,
        **bench_result,
        "monitoring": format_monitoring_metrics(monitoring_data),
    }

    if output_path:
        save_benchmark_result(result, output_path)

    return result


async def run_experiment(config_path: Path, output_dir: Path):
    """Run full experiment from config file."""
    with open(config_path) as f:
        config = json.load(f)

    experiment_name = config["name"]
    models = config["models"]
    workloads = config["workloads"]
    batch_sizes = config.get("batch_sizes", [config.get("batch_size", 1)])
    request_patterns = config.get("request_patterns", {})

    print(f"\n{'='*70}")
    print(f"Running experiment: {experiment_name}")
    print(f"Description: {config['description']}")
    print(f"Models: {len(models)}, Workloads: {len(workloads)}, Batch sizes: {batch_sizes}")
    print(f"{'='*70}\n")

    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        for batch_size in batch_sizes:
            try:
                model_path = resolve_model_path(model_name, batch_size=batch_size)

                if not model_path.exists():
                    print(f"⚠️  Skipping {model_name} (bs={batch_size}): Model not found at {model_path}")
                    continue

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
                        output_filename = f"input_{input_len}_output_{output_len}_bs_{batch_size}_req_{num_requests}.json"
                        output_path = model_output_dir / output_filename

                        if output_path.exists():
                            with open(output_path) as f:
                                existing_result = json.load(f)
                            if "error" in existing_result:
                                print(f"⚠️  Skipping {model_name} (bs={batch_size}, input={input_len}, req={num_requests}): Previous error exists")
                            else:
                                print(f"⏭️  Skipping {model_name} (bs={batch_size}, input={input_len}, req={num_requests}): Result already exists")
                            continue

                        await run_single_benchmark(
                            model_path=model_path,
                            model_short_name=model_name,
                            batch_size=batch_size,
                            input_len=input_len,
                            output_len=output_len,
                            num_requests=num_requests,
                            output_path=output_path,
                            experiment_name=experiment_name,
                        )

            except ValueError as e:
                print(f"⚠️  Skipping {model_name}: {e}")
                continue

    print(f"\n{'='*70}")
    print(f"Experiment complete! Results saved to {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RBLN compiled models on NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    single_parser = subparsers.add_parser("single", help="Run single benchmark")
    single_parser.add_argument("--model", type=str, required=True, help="Model short name (e.g., qwen3-0.6b)")
    single_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    single_parser.add_argument("--input-len", type=int, required=True, help="Input context length")
    single_parser.add_argument("--output-len", type=int, required=True, help="Output token length")
    single_parser.add_argument("--num-requests", type=int, required=True, help="Number of requests")
    single_parser.add_argument("--output", type=str, help="Output JSON path")
    single_parser.add_argument("--no-monitoring", action="store_true", help="Disable hardware monitoring")

    experiment_parser = subparsers.add_parser("experiment", help="Run full experiment from config")
    experiment_parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    experiment_parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")

    args = parser.parse_args()

    if args.command == "single":
        model_path = resolve_model_path(args.model, batch_size=args.batch_size)
        output_path = Path(args.output) if args.output else None

        result = asyncio.run(run_single_benchmark(
            model_path=model_path,
            model_short_name=args.model,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            num_requests=args.num_requests,
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
        ))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
