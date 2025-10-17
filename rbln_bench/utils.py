from __future__ import annotations

from pathlib import Path


def slugify(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def build_signature(
    *,
    batch_size: int,
    max_seq_len: int,
    tensor_parallel_size: int,
    attn_impl: str,
    kvcache_partition_len: int,
    npu: str,
) -> str:
    parts = [
        f"msl{max_seq_len}",
        f"kv{kvcache_partition_len}",
        f"tp{tensor_parallel_size}",
        f"bs{batch_size}",
        f"attn-{attn_impl}",
    ]
    parts.append(f"npu-{slugify(npu)}")
    return "_".join(parts)


def build_output_dir(
    model_id: str,
    save_root: str,
    batch_size: int,
    max_seq_len: int,
    tensor_parallel_size: int,
    attn_impl: str,
    kvcache_partition_len: int,
    npu: str,
) -> Path:
    base = slugify(model_id)
    signature = build_signature(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        tensor_parallel_size=tensor_parallel_size,
        attn_impl=attn_impl,
        kvcache_partition_len=kvcache_partition_len,
        npu=npu,
    )
    return Path(save_root) / f"{base}_{signature}"
