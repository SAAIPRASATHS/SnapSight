from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi.testclient import TestClient


def _make_sample_image_bytes(width: int = 640, height: int = 480) -> bytes:
    """Generate a small synthetic JPEG-like image buffer using numpy."""
    try:
        import cv2

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (120, 160, 200)
        cv2.rectangle(frame, (50, 50), (200, 200), (50, 80, 120), -1)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok and buf is not None:
            return buf.tobytes()
    except Exception:
        pass
    return b"\xff\xd8\xff\xe0" + b"\x00" * 1024 + b"\xff\xd9"


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Fixture returning raw JPEG bytes for a small synthetic image."""
    return _make_sample_image_bytes()


@pytest.fixture
def sample_image_file(sample_image_bytes: bytes) -> io.BytesIO:
    """Fixture returning a file-like object wrapping sample_image_bytes."""
    return io.BytesIO(sample_image_bytes)


@pytest.fixture
def sample_frame() -> "np.ndarray":
    """Fixture returning a synthetic BGR numpy frame (480x640x3)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (100, 150, 200)
    return frame


@pytest.fixture
def mock_settings():
    """
    Fixture that overrides get_settings() to return a deterministic test
    configuration that does not depend on .env or hardware.yaml.
    """
    from backend.app.utils.config import Settings

    test_settings = Settings(
        BACKEND_HOST="127.0.0.1",
        BACKEND_PORT=8765,
        PREFERRED_BACKEND="mock",
        FALLBACK_BACKEND="mock",
        CAMERA_INDEX=None,
        CAMERA_WIDTH=640,
        CAMERA_HEIGHT=480,
        INFERENCE_FPS=10,
        CONFIDENCE_THRESHOLD=0.5,
        ENABLE_CLOUD_APIS=False,
        LOG_LEVEL="WARNING",
    )

    with patch("backend.app.utils.config.get_settings", return_value=test_settings):
        yield test_settings


@pytest.fixture
def client(mock_settings):
    """
    Fixture providing a FastAPI TestClient for the backend app.

    Patches service singletons in the routes module so endpoints return
    deterministic MockProvider-based results without starting cameras or
    WebSocket producers.
    """
    from backend.app.inference.providers import MockProvider
    from backend.app.utils.metrics import RollingMetrics

    provider = MockProvider(seed=42)
    provider.load()

    metrics = RollingMetrics(window_size=20)
    metrics.record_latency(0.05)
    metrics.record_latency(0.06)
    metrics.record_frame()
    metrics.record_frame()

    class _FakeInference:
        def start(self):
            pass

        def shutdown(self):
            pass

        def provider_info(self):
            return provider.info()

        def get_metrics(self):
            snap = metrics.snapshot()
            snap["provider"] = provider.name
            snap["backend"] = provider.backend
            snap["frames_processed"] = 2
            snap["model_load_time_ms"] = provider.load_time_ms
            return snap

        def all_provider_infos(self):
            return [provider.info()]

        async def run_chat_async(self, text, context=None):
            return f"Mock reply to: {text[:80]}"

    class _FakeCamera:
        is_available = False

        def start(self, index=None):
            return False

        def stop(self):
            pass

        def get_status(self):
            class _Status:
                index = 0
                opened = False
                width = 0
                height = 0
                frames_read = 0
                last_error = "test-mode"
                backend = "test"
            return _Status()

    inference = _FakeInference()
    camera = _FakeCamera()

    with patch("backend.app.main.lifespan", _build_noop_lifespan(inference, camera)):
        with patch("backend.app.api.routes._inference", inference):
            with patch("backend.app.api.routes._camera", camera):
                with patch("backend.app.main._settings", mock_settings):
                    from backend.app.main import create_app

                    app = create_app()
                    app.state.services = {}
                    with TestClient(app, raise_server_exceptions=False) as tc:
                        yield tc


def _build_noop_lifespan(inference, camera):
    """Return a lifespan context manager that skips real hardware setup."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop(app):
        from backend.app.safety.analyzer import SafetyAnalyzer
        from backend.app.hardware.detector import HardwareDetector

        hw = HardwareDetector()

        app.state.services = {
            "inference": inference,
            "camera": camera,
            "detector": None,
            "ocr": None,
            "stt": None,
            "safety": SafetyAnalyzer(),
            "hardware": hw,
        }
        yield

    return _noop
