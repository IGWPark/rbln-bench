from __future__ import annotations

from typing import Dict, Any, Optional
from functools import wraps
import asyncio

try:
    from rbln_mon import track
    RBLN_MON_AVAILABLE = True
except ImportError:
    RBLN_MON_AVAILABLE = False


def create_monitoring_decorator(
    backend: str = "auto",
    save: bool = True,
    save_dir: str = "monitoring",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Create a monitoring decorator with rbln-mon integration."""
    if not RBLN_MON_AVAILABLE:
        def no_op_decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
            return wrapper
        return no_op_decorator

    return track(
        backend=backend,
        save=save,
        save_dir=save_dir,
        metadata=metadata or {},
    )


def extract_monitoring_data(tracker_result: Any) -> Optional[Dict[str, Any]]:
    """Extract monitoring data from rbln-mon tracker result."""
    if not RBLN_MON_AVAILABLE or tracker_result is None:
        return None

    if hasattr(tracker_result, 'export'):
        return tracker_result.export()

    return None


def format_monitoring_metrics(monitoring_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Format monitoring data for benchmark output."""
    if not monitoring_data:
        return {
            "enabled": False,
            "message": "Monitoring not available or not enabled"
        }

    formatted = {
        "enabled": True,
        "duration_s": monitoring_data.get("duration", 0),
        "samples_collected": len(monitoring_data.get("samples", [])),
        "backend": monitoring_data.get("backend", {}),
    }

    summary = monitoring_data.get("summary", {})
    for device_key, metrics in summary.items():
        formatted[device_key] = {
            "utilization": {
                "avg_percent": round(metrics.get("util_avg", 0), 2),
                "peak_percent": round(metrics.get("util_peak", 0), 2),
            },
            "memory": {
                "peak_gb": round(metrics.get("mem_peak_gb", 0), 2),
            },
            "power": {
                "avg_w": round(metrics.get("power_avg_w", 0), 2),
                "peak_w": round(metrics.get("power_peak_w", 0), 2),
            },
            "temperature": {
                "range_c": metrics.get("temp_range_c", [0, 0]),
            },
        }

    return formatted
