from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from optimum.rbln import RBLNAutoModelForCausalLM

from .utils import build_output_dir


def compile_model(
    model_id: str,
    save_root: str,
    batch_size: int,
    max_seq_len: int,
    tensor_parallel_size: int,
    attn_impl: str,
    kvcache_partition_len: int,
    npu: str,
    create_runtimes: bool,
    overwrite: bool,
) -> Path:
    output_dir = build_output_dir(
        model_id=model_id,
        save_root=save_root,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tensor_parallel_size=tensor_parallel_size,
        attn_impl=attn_impl,
        kvcache_partition_len=kvcache_partition_len,
        npu=npu,
    )

    if output_dir.exists() and not overwrite:
        print(f"[SKIP] {output_dir} already exists.")
        return output_dir

    print(f"[INFO] Compiling {model_id}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,
        rbln_batch_size=batch_size,
        rbln_max_seq_len=max_seq_len,
        rbln_tensor_parallel_size=tensor_parallel_size,
        rbln_attn_impl=attn_impl,
        rbln_kvcache_partition_len=kvcache_partition_len,
        rbln_create_runtimes=create_runtimes,
        rbln_npu=npu
    )

    model.save_pretrained(output_dir.as_posix())
    print(f"[INFO] Saved to {output_dir}")
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compile models for Rebellions NPUs")
    parser.add_argument(
        "job",
        help="Job definition as a JSON string or path to a JSON file containing a single job dictionary.",
    )
    return parser.parse_args(argv)


DEFAULTS: Dict[str, Any] = {
    "save_root": "./compiled",
    "batch_size": 1,
    "max_seq_len": 8,
    "tensor_parallel_size": 1,
    "attn_impl": "flash_attn",
    "kvcache_partition_len": 4096,
    "npu": "RBLN-CA22",
    "create_runtimes": False,
    "overwrite": False,
}


REQUIRED_FIELDS = ("model_id", "max_seq_len", "kvcache_partition_len", "batch_size")


def load_job(job_arg: str) -> Dict[str, Any]:
    candidate_path = Path(job_arg)
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(job_arg)

    if not isinstance(data, dict):
        raise ValueError("Job definition must be a JSON object.")

    job = {**DEFAULTS, **data}

    for field in REQUIRED_FIELDS:
        if job.get(field) in (None, ""):
            raise ValueError(f"Missing required job field '{field}'.")

    return job


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    job = load_job(args.job)
    compile_model(
        model_id=job["model_id"],
        save_root=job["save_root"],
        batch_size=job["batch_size"],
        max_seq_len=job["max_seq_len"],
        tensor_parallel_size=job["tensor_parallel_size"],
        attn_impl=job["attn_impl"],
        kvcache_partition_len=job["kvcache_partition_len"],
        npu=job["npu"],
        create_runtimes=job["create_runtimes"],
        overwrite=job["overwrite"],
    )


if __name__ == "__main__":
    main()
