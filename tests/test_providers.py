from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

np = pytest.importorskip("numpy")

from backend.app.inference.providers import MockProvider, ProviderFactory
from backend.app.models.schemas import SceneDescription


class TestMockProviderLoad:
    def test_load_succeeds(self):
        """MockProvider.load() should succeed and mark the provider loaded."""
        p = MockProvider(seed=0)
        assert p.loaded is False
        ok = p.load()
        assert ok is True
        assert p.loaded is True
        assert p.load_time_ms is not None
        assert p.load_time_ms >= 0

    def test_load_is_idempotent(self):
        """Calling load() twice should not fail or reset state."""
        p = MockProvider(seed=0)
        p.load()
        first_time = p.load_time_ms
        ok = p.load()
        assert ok is True
        assert p.load_time_ms == first_time


class TestMockProviderDetect:
    def test_detect_returns_list_with_detections(self, sample_frame):
        """detect() should return a list containing labels like Person, Chair etc."""
        p = MockProvider(seed=42)
        p.load()
        dets = p.detect(sample_frame)
        assert isinstance(dets, list)
        assert len(dets) > 0
        labels = [d.label.lower() for d in dets]
        candidate_labels = {"person", "chair", "door", "table", "exit sign"}
        found = candidate_labels.intersection(labels)
        assert len(found) > 0, f"Expected common mock labels, got: {labels}"
        for d in dets:
            assert 0.0 <= d.confidence <= 1.0
            assert d.bbox is not None
            assert d.bbox.xmax > d.bbox.xmin
            assert d.bbox.ymax > d.bbox.ymin


class TestMockProviderOCR:
    def test_ocr_returns_list_with_text(self, sample_frame):
        """ocr() should return a non-empty list of OCRResult objects with text."""
        p = MockProvider(seed=42)
        p.load()
        results = p.ocr(sample_frame)
        assert isinstance(results, list)
        assert len(results) > 0
        for r in results:
            assert isinstance(r.text, str)
            assert len(r.text) > 0
            assert 0.0 <= r.confidence <= 1.0


class TestMockProviderDescribe:
    def test_describe_returns_string_summary(self, sample_frame):
        """describe() should return a SceneDescription with non-empty summary string."""
        p = MockProvider(seed=42)
        p.load()
        desc = p.describe(sample_frame)
        assert isinstance(desc, SceneDescription)
        assert isinstance(desc.summary, str)
        assert len(desc.summary) > 0
        assert isinstance(desc.detailed, str)
        assert isinstance(desc.objects, list)


class TestMockProviderTranscribe:
    def test_transcribe_returns_string(self):
        """transcribe() should return a non-empty string given audio bytes."""
        p = MockProvider(seed=42)
        p.load()
        audio_bytes = b"fake-audio-bytes-for-testing" * 8
        text = p.transcribe(audio_bytes, language="en")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_transcribe_empty_audio_returns_empty(self):
        """Empty audio bytes should produce empty string."""
        p = MockProvider(seed=0)
        p.load()
        assert p.transcribe(b"") == ""


class TestProviderFactory:
    def test_factory_creates_mock_provider(self, mock_settings):
        """ProviderFactory should create a MockProvider when preferred=mock."""
        factory = ProviderFactory()
        provider = factory.create()
        assert provider is not None
        assert isinstance(provider, MockProvider)
        assert provider.name == "mock"
        assert provider.backend == "mock"
        assert provider.loaded is True

    def test_factory_available_names_includes_mock(self):
        """available_backend_names() must always include 'mock'."""
        names = ProviderFactory.available_backend_names()
        assert isinstance(names, list)
        assert "mock" in names
