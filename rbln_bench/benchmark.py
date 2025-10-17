from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


def load_rbln_config(model_path: str) -> dict:
    """Load rbln_config.json from model directory."""
    config_path = Path(model_path) / "rbln_config.json"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return json.load(f)


async def send_request(
    engine: AsyncLLMEngine,
    prompt_tokens: List[int],
    request_id: str,
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    """Send a single request and track TTFT, E2EL, and TPOT."""
    request_start_time = time.perf_counter()

    ttft = None
    output_tokens = []
    first_token_time = None

    async for request_output in engine.generate(
        {"prompt_token_ids": prompt_tokens},
        sampling_params,
        request_id
    ):
        current_time = time.perf_counter()

        if ttft is None and len(request_output.outputs[0].token_ids) > 0:
            ttft = current_time - request_start_time
            first_token_time = current_time

        output_tokens = request_output.outputs[0].token_ids

    request_end_time = time.perf_counter()
    e2el = request_end_time - request_start_time

    tpot = None
    if first_token_time is not None and len(output_tokens) > 1:
        decode_time = request_end_time - first_token_time
        tpot = decode_time / (len(output_tokens) - 1)

    return {
        "request_id": request_id,
        "prompt_len": len(prompt_tokens),
        "output_len": len(output_tokens),
        "ttft": ttft,
        "e2el": e2el,
        "tpot": tpot,
        "success": ttft is not None,
    }


async def run_benchmark(
    model_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    max_model_len: Optional[int] = None,
    block_size: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
) -> Dict[str, Any]:
    """Run benchmark and return comprehensive results."""

    rbln_config = load_rbln_config(model_path)

    block_size = block_size or rbln_config.get("kvcache_block_size", 4096)
    max_model_len = max_model_len or rbln_config.get("max_seq_len", 8192)
    max_num_batched_tokens = max_num_batched_tokens or rbln_config.get("max_seq_len", 8192)
    max_num_seqs = max_num_seqs or rbln_config.get("batch_size", 1)

    print(f"Initializing AsyncLLMEngine for {model_path}...")

    engine_args_dict = {
        "model": model_path,
        "block_size": block_size,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }

    if gpu_memory_utilization is not None:
        engine_args_dict["gpu_memory_utilization"] = gpu_memory_utilization

    try:
        engine_args = AsyncEngineArgs(**engine_args_dict)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to initialize engine: {error_msg}")
        return {
            "error": error_msg,
            "workload": {
                "input_len": input_len,
                "output_len": output_len,
                "num_requests": num_requests,
            },
            "engine_config": {
                "max_num_seqs": max_num_seqs,
                "block_size": block_size,
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_num_batched_tokens,
            },
        }

    prompts = [list(range(input_len)) for _ in range(num_requests)]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
    )

    print(f"Sending {num_requests} concurrent requests...")

    benchmark_start_time = time.perf_counter()

    tasks = [
        send_request(engine, prompt, f"request-{i}", sampling_params)
        for i, prompt in enumerate(prompts)
    ]

    results = await asyncio.gather(*tasks)

    benchmark_end_time = time.perf_counter()
    total_time = benchmark_end_time - benchmark_start_time

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {"error": "No successful requests"}

    ttfts = np.array([r["ttft"] for r in successful_results]) * 1000
    e2els = np.array([r["e2el"] for r in successful_results]) * 1000
    tpots = np.array([r["tpot"] * 1000 for r in successful_results if r["tpot"] is not None])

    total_prompt_tokens = sum(r["prompt_len"] for r in successful_results)
    total_output_tokens = sum(r["output_len"] for r in successful_results)
    total_tokens = total_prompt_tokens + total_output_tokens

    output_tps = total_output_tokens / total_time
    total_tps = total_tokens / total_time
    request_throughput = len(successful_results) / total_time

    result = {
        "workload": {
            "input_len": input_len,
            "output_len": output_len,
            "num_requests": num_requests,
        },
        "engine_config": {
            "max_num_seqs": max_num_seqs,
            "block_size": block_size,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
        },
        "summary": {
            "total_time_s": round(total_time, 2),
            "completed_requests": len(successful_results),
            "failed_requests": num_requests - len(successful_results),
            "request_throughput_per_s": round(request_throughput, 2),
            "output_tokens_per_s": round(output_tps, 2),
            "total_tokens_per_s": round(total_tps, 2),
            "total_input_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "ttft_ms": {
            "mean": round(np.mean(ttfts), 2),
            "median": round(np.median(ttfts), 2),
            "min": round(np.min(ttfts), 2),
            "max": round(np.max(ttfts), 2),
            "std": round(np.std(ttfts), 2),
            "p50": round(np.percentile(ttfts, 50), 2),
            "p90": round(np.percentile(ttfts, 90), 2),
            "p95": round(np.percentile(ttfts, 95), 2),
            "p99": round(np.percentile(ttfts, 99), 2),
        },
        "e2el_ms": {
            "mean": round(np.mean(e2els), 2),
            "median": round(np.median(e2els), 2),
            "min": round(np.min(e2els), 2),
            "max": round(np.max(e2els), 2),
            "std": round(np.std(e2els), 2),
            "p50": round(np.percentile(e2els, 50), 2),
            "p90": round(np.percentile(e2els, 90), 2),
            "p95": round(np.percentile(e2els, 95), 2),
            "p99": round(np.percentile(e2els, 99), 2),
        },
        "individual_requests": [
            {
                "request_id": r["request_id"],
                "prompt_len": r["prompt_len"],
                "output_len": r["output_len"],
                "ttft_ms": round(r["ttft"] * 1000, 2) if r["ttft"] else None,
                "e2el_ms": round(r["e2el"] * 1000, 2),
                "tpot_ms": round(r["tpot"] * 1000, 2) if r["tpot"] else None,
                "success": r["success"],
            }
            for r in results
        ],
    }

    if len(tpots) > 0:
        result["tpot_ms"] = {
            "mean": round(np.mean(tpots), 2),
            "median": round(np.median(tpots), 2),
            "min": round(np.min(tpots), 2),
            "max": round(np.max(tpots), 2),
            "std": round(np.std(tpots), 2),
            "p50": round(np.percentile(tpots, 50), 2),
            "p90": round(np.percentile(tpots, 90), 2),
            "p95": round(np.percentile(tpots, 95), 2),
            "p99": round(np.percentile(tpots, 99), 2),
        }

    return result


def save_benchmark_result(
    result: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save benchmark result to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {output_path}")
