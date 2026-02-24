"""
Performance Monitor
--------------------
Tracks inference latency, request rate, memory usage, and model load time.
"""

from __future__ import annotations

import time
import os
from collections import deque
from backend.logging_config import get_logger

logger = get_logger("performance")

_latencies: deque[float] = deque(maxlen=100)
_request_times: deque[float] = deque(maxlen=300)
_model_load_time: float = 0.0


def record_inference_latency(seconds: float) -> None:
    """Record how long one pipeline inference took."""
    _latencies.append(seconds)


def record_request() -> None:
    """Record that an API request was made."""
    _request_times.append(time.time())


def set_model_load_time(seconds: float) -> None:
    _model_load_time = seconds


def get_performance_stats() -> dict:
    """Return current performance metrics."""
    now = time.time()

    # Average inference latency
    avg_latency = (sum(_latencies) / len(_latencies)) if _latencies else 0.0

    # Requests per minute
    cutoff = now - 60
    recent = sum(1 for t in _request_times if t > cutoff)

    # App memory usage (MB)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = 0.0

    return {
        "avg_inference_latency_ms": round(avg_latency * 1000, 2),
        "max_inference_latency_ms": round(max(_latencies) * 1000, 2) if _latencies else 0.0,
        "min_inference_latency_ms": round(min(_latencies) * 1000, 2) if _latencies else 0.0,
        "inference_samples": len(_latencies),
        "requests_per_minute": recent,
        "app_memory_mb": round(mem_mb, 1),
        "model_load_time_ms": round(_model_load_time * 1000, 2),
    }
