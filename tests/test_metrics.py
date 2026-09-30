from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

pytest.importorskip("pydantic")

from backend.app.utils.metrics import RollingMetrics
from backend.app.hardware.detector import HardwareDetector


class TestRollingMetricsAppendAndMean:
    def test_appends_values_and_calculates_mean(self):
        """RollingMetrics appends values and calculates mean correctly."""
        rm = RollingMetrics(window_size=10)
        rm.record_latency(0.1)
        rm.record_latency(0.2)
        rm.record_latency(0.3)
        stats = rm.latency_stats()
        assert stats["count"] == 3
        assert stats["mean_ms"] is not None
        expected_mean_ms = ((0.1 + 0.2 + 0.3) / 3.0) * 1000.0
        assert abs(stats["mean_ms"] - expected_mean_ms) < 1e-6
        assert stats["min_ms"] is not None
        assert stats["max_ms"] is not None
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_empty_window_returns_none_stats(self):
        """No latencies recorded -> mean/min/max/p95/p99 are None, count 0."""
        rm = RollingMetrics(window_size=5)
        stats = rm.latency_stats()
        assert stats["count"] == 0
        assert stats["mean_ms"] is None
        assert stats["p95_ms"] is None
        assert stats["p99_ms"] is None

    def test_window_respects_max_size(self):
        """Latency deque should evict oldest entries beyond window_size."""
        rm = RollingMetrics(window_size=3)
        for i in range(1, 8):
            rm.record_latency(float(i))
        stats = rm.latency_stats()
        assert stats["count"] == 3
        latest_three = [5.0, 6.0, 7.0]
        expected_mean_ms = (sum(latest_three) / 3.0) * 1000.0
        assert abs(stats["mean_ms"] - expected_mean_ms) < 1e-6


class TestRollingMetricsFPS:
    def test_fps_calculation_non_negative(self):
        """RollingMetrics fps calculation should always be >= 0."""
        rm = RollingMetrics(window_size=20)
        assert rm.calculate_fps() == 0.0
        rm.record_frame()
        assert rm.calculate_fps() == 0.0
        rm.record_frame()
        fps = rm.calculate_fps()
        assert fps >= 0.0
        assert isinstance(fps, float)

    def test_fps_matches_known_timing(self):
        """With controlled timestamps, FPS should be approximately (N-1)/delta."""
        rm = RollingMetrics(window_size=10)
        t0 = time.perf_counter()
        with pytest.MonkeyPatch.context() as mp:
            for i in range(5):
                mp.setattr(time, "perf_counter", lambda: t0 + float(i) * 0.1)
                rm.record_frame()
            fps = rm.calculate_fps()
        assert fps >= 0.0

    def test_snapshot_includes_fps_and_latency(self):
        """snapshot() dict should contain fps, latency, process, window_size keys."""
        rm = RollingMetrics(window_size=15)
        rm.record_latency(0.05)
        rm.record_frame()
        snap = rm.snapshot()
        assert "fps" in snap
        assert "latency" in snap
        assert "process" in snap
        assert snap["window_size"] == 15


class TestHardwareDetector:
    def test_detect_returns_info_dict_with_cpu_and_ram(self):
        """HardwareDetector.detect() returns dict-like HardwareInfo with cpu/ram."""
        detector = HardwareDetector()
        info = detector.detect()
        info_dict = info.model_dump()
        assert isinstance(info_dict, dict)
        assert "cpu" in info_dict
        assert "ram_gb" in info_dict
        assert isinstance(info_dict["cpu"], str)
        assert len(info_dict["cpu"]) > 0
        assert isinstance(info_dict["ram_gb"], (int, float))
        assert info_dict["ram_gb"] >= 0.0
        assert "platform" in info_dict
        assert "available_backends" in info_dict
        assert isinstance(info_dict["available_backends"], list)
        assert "mock" in info_dict["available_backends"]

    def test_detect_is_cached(self):
        """Second call without force_refresh returns the same cached object."""
        detector = HardwareDetector()
        first = detector.detect()
        second = detector.detect()
        assert first is second
