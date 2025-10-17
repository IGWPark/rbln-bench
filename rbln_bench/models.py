from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


MODELS = {
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "size": "0.6B",
        "pattern": "Qwen-Qwen3-0.6B_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "qwen3-1.7b": {
        "hf_id": "Qwen/Qwen3-1.7B",
        "size": "1.7B",
        "pattern": "Qwen-Qwen3-1.7B_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B-Instruct-2507",
        "size": "4B",
        "pattern": "Qwen-Qwen3-4B-Instruct-2507_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "size": "0.5B",
        "pattern": "Qwen-Qwen2.5-0.5B-Instruct_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "size": "1.5B",
        "pattern": "Qwen-Qwen2.5-1.5B-Instruct_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "pattern": "Qwen-Qwen2.5-7B-Instruct_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "size": "3B",
        "pattern": "meta-llama-Llama-3.2-3B-Instruct_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "size": "8B",
        "pattern": "meta-llama-Llama-3.1-8B-Instruct_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "gemma-2b": {
        "hf_id": "google/gemma-2b-it",
        "size": "2B",
        "pattern": "google-gemma-2b-it_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "gpt2-medium": {
        "hf_id": "openai-community/gpt2-medium",
        "size": "0.35B",
        "pattern": "openai-community-gpt2-medium_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "gpt2-xl": {
        "hf_id": "openai-community/gpt2-xl",
        "size": "1.5B",
        "pattern": "openai-community-gpt2-xl_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "phi-2": {
        "hf_id": "microsoft/Phi-2",
        "size": "2.7B",
        "pattern": "microsoft-Phi-2_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "deepseek-r1-qwen3-8b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "size": "8B",
        "pattern": "deepseek-ai-DeepSeek-R1-0528-Qwen3-8B_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "llama-3.1-8b-fp8": {
        "hf_id": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        "size": "8B",
        "quantization": "FP8",
        "pattern": "RedHatAI-Meta-Llama-3.1-8B-Instruct-FP8_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
    "llama-3.1-8b-w8a16": {
        "hf_id": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
        "size": "8B",
        "quantization": "W8A16",
        "pattern": "RedHatAI-Meta-Llama-3.1-8B-Instruct-quantized.w8a16_msl{msl}_kv{kv}_tp{tp}_bs{bs}_attn-{attn}_npu-{npu}",
    },
}


def resolve_model_path(
    short_name: str,
    batch_size: int = 1,
    max_seq_len: int = 8192,
    kvcache_partition_len: int = 4096,
    tensor_parallel_size: int = 1,
    attn_impl: str = "flash_attn",
    npu: str = "RBLN-CA22",
    compiled_dir: str = "compiled",
) -> Path:
    """Resolve short model name to full compiled path."""
    if short_name not in MODELS:
        raise ValueError(f"Unknown model: {short_name}. Available: {list(MODELS.keys())}")

    pattern = MODELS[short_name]["pattern"]
    dirname = pattern.format(
        msl=max_seq_len,
        kv=kvcache_partition_len,
        tp=tensor_parallel_size,
        bs=batch_size,
        attn=attn_impl,
        npu=npu,
    )
    return Path(compiled_dir) / dirname


def get_model_info(short_name: str) -> Dict[str, Any]:
    """Get model metadata by short name."""
    if short_name not in MODELS:
        raise ValueError(f"Unknown model: {short_name}. Available: {list(MODELS.keys())}")
    return MODELS[short_name].copy()


def list_models() -> list[str]:
    """List all available model short names."""
    return list(MODELS.keys())
