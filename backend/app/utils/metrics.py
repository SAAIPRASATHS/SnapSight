from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is listed in requirements
    psutil = None


@dataclass
class RollingMetrics:
    """
    Thread-safe rolling window metrics collector for inference performance.

    Tracks per-call latency values in a fixed-size deque, and exposes
    derived statistics (mean/p95/p99 latency, FPS) along with process-level
    CPU and memory sampling.
    """

    window_size: int = 200
    _latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _frame_timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _process: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._latencies = deque(maxlen=self.window_size)
        self._frame_timestamps = deque(maxlen=self.window_size)
        if psutil is not None:
            try:
                self._process = psutil.Process()
                self._process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None

    def record_latency(self, latency_seconds: float) -> None:
        """Record a single inference latency value in seconds."""
        if latency_seconds <= 0:
            return
        with self._lock:
            self._latencies.append(float(latency_seconds))

    def record_frame(self) -> None:
        """Record that a frame was processed for FPS calculation."""
        with self._lock:
            self._frame_timestamps.append(time.perf_counter())

    def calculate_fps(self) -> float:
        """
        Return instantaneous FPS based on the frame timestamp window.

        Returns:
            FPS as a float, or 0.0 if fewer than 2 frames have been recorded.
        """
        with self._lock:
            if len(self._frame_timestamps) < 2:
                return 0.0
            elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._frame_timestamps) - 1) / elapsed

    def latency_stats(self) -> Dict[str, Optional[float]]:
        """
        Return mean / p95 / p99 latency statistics from the rolling window.

        Keys: ``mean_ms``, ``p95_ms``, ``p99_ms``, ``min_ms``, ``max_ms``,
        ``count``. Values are ``None`` if the window is empty.
        """
        with self._lock:
            if not self._latencies:
                return {
                    "mean_ms": None,
                    "p95_ms": None,
                    "p99_ms": None,
                    "min_ms": None,
                    "max_ms": None,
                    "count": 0,
                }
            values = sorted(self._latencies)
            n = len(values)

            def _pct(p: float) -> float:
                k = max(0, min(n - 1, int(round((n - 1) * p))))
                return values[k]

            return {
                "mean_ms": 1000.0 * (sum(values) / n),
                "p95_ms": 1000.0 * _pct(0.95),
                "p99_ms": 1000.0 * _pct(0.99),
                "min_ms": 1000.0 * values[0],
                "max_ms": 1000.0 * values[-1],
                "count": n,
            }

    def sample_process(self) -> Dict[str, Any]:
        """
        Sample current process CPU and memory usage.

        Returns a dict with keys:
            - ``cpu_percent``: float or None if psutil unavailable
            - ``memory_rss_mb``: RSS memory in MB or None
            - ``psutil_available``: bool
        """
        result: Dict[str, Any] = {
            "cpu_percent": None,
            "memory_rss_mb": None,
            "psutil_available": psutil is not None,
        }
        if self._process is None:
            return result
        try:
            result["cpu_percent"] = self._process.cpu_percent(interval=None)
            mem = self._process.memory_info()
            result["memory_rss_mb"] = mem.rss / (1024.0 * 1024.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
        return result

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a complete metrics snapshot dictionary.

        Combines latency statistics, FPS, and process sampling into a single
        JSON-serialisable payload suitable for the /metrics endpoint.
        """
        stats = self.latency_stats()
        return {
            "fps": round(self.calculate_fps(), 3),
            "latency": stats,
            "process": self.sample_process(),
            "window_size": self.window_size,
        }


def has_npu_runtime() -> Dict[str, Any]:
    """
    Probe for NPU / Qualcomm AI Hub runtime availability.

    Performs soft import checks only - never contacts an external service.
    """
    info: Dict[str, Any] = {
        "available": False,
        "backends": [],
        "notes": "",
    }
    try:
        import qai_hub  # type: ignore  # noqa: F401

        info["backends"].append("qai_hub")
        info["available"] = True
    except ImportError:
        pass

    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        info["onnxruntime_providers"] = providers
        qnn = [p for p in providers if "QNN" in p.upper()]
        if qnn:
            info["backends"].extend(qnn)
            info["available"] = True
    except ImportError:
        pass

    if not info["backends"]:
        info["notes"] = "No NPU runtime detected. Install qai_hub or QNN-enabled onnxruntime to enable NPU acceleration."
    return info


__all__ = ["RollingMetrics", "has_npu_runtime"]
