from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pydantic")


class TestHealthEndpoint:
    def test_health_returns_200_with_status(self, client):
        """GET /health should return 200 with a non-empty status field."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert body["status"] in {"healthy", "degraded", "unhealthy"}
        assert body.get("version", "") != ""
        assert "backend" in body


class TestHardwareEndpoint:
    def test_hardware_returns_200_with_vendor_and_platform(self, client):
        """GET /api/v1/hardware should return 200 with vendor and platform."""
        resp = client.get("/api/v1/hardware")
        assert resp.status_code == 200
        body = resp.json()
        assert "vendor" in body
        assert "platform" in body
        assert isinstance(body["vendor"], str)
        assert isinstance(body["platform"], str)
        assert len(body["platform"]) > 0
        assert "available_backends" in body
        assert isinstance(body["available_backends"], list)


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        """GET /api/v1/models should return 200 with current + models list."""
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "current" in body
        assert isinstance(body["models"], list)


class TestMetricsEndpoint:
    def test_metrics_returns_200_with_fps_and_latency(self, client):
        """GET /api/v1/metrics should return 200 with fps and latency fields."""
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "fps" in body
        assert "latency" in body
        assert isinstance(body["fps"], (int, float))
        assert body["fps"] >= 0
        assert isinstance(body["latency"], dict)


class TestVisionDetectEndpoint:
    def test_detect_without_image_returns_422(self, client):
        """POST /api/v1/vision/detect without image should return 422."""
        resp = client.post("/api/v1/vision/detect")
        assert resp.status_code == 422

    def test_detect_with_empty_form_returns_422(self, client):
        """POST with empty form data should still return 422."""
        resp = client.post("/api/v1/vision/detect", data={})
        assert resp.status_code == 422


class TestChatQueryEndpoint:
    def test_chat_query_valid_body_returns_200(self, client):
        """POST /api/v1/chat/query with valid body returns 200."""
        payload = {"text": "What do you see?", "context": None}
        resp = client.post("/api/v1/chat/query", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("success") is True
        assert "reply" in body
        assert isinstance(body["reply"], str)
        assert "latency_ms" in body

    def test_chat_query_missing_text_returns_422(self, client):
        """Empty text field should be rejected by Pydantic validation."""
        resp = client.post("/api/v1/chat/query", json={"text": ""})
        assert resp.status_code == 422
