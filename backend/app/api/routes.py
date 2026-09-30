from __future__ import annotations

import io
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from ..hardware.detector import HardwareDetector
from ..inference.manager import InferenceManager
from ..models.schemas import (
    ChatQuery,
    ChatResponse,
    DescribeResponse,
    DetectionResponse,
    HardwareInfo,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ModelList,
    OCRResponse,
    SafetyAlert,
    SafetyAnalyzeRequest,
    SafetyAnalyzeResponse,
    SceneDescription,
    SnapshotRequest,
    SnapshotResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from ..ocr.engine import OCREngine
from ..safety.analyzer import SafetyAnalyzer
from ..speech.stt import SpeechToText
from ..utils.config import get_settings
from ..utils.logging import setup_logger
from ..vision.camera import CameraManager
from ..vision.detector import ObjectDetector


logger = setup_logger(__name__)

router = APIRouter(tags=["core"])

_settings = get_settings()
_hardware_detector = HardwareDetector()
_inference = InferenceManager.instance()
_camera = CameraManager()
_detector = ObjectDetector(_inference)
_ocr = OCREngine(_inference)
_stt = SpeechToText(_inference)
_safety = SafetyAnalyzer()

_STARTUP_TIME: float = time.perf_counter()
_VERSION = "1.0.0"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR uint8 numpy array using cv2."""
    try:
        import cv2  # type: ignore
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenCV (cv2) not installed on this backend; image decoding unavailable.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image upload.",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image buffer is empty.")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode uploaded image as JPEG/PNG.",
        )
    return img


# ----------------------------------------------------------------------
# Health & system
# ----------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, summary="Service health")
async def get_health() -> HealthResponse:
    """Return liveness/readiness information for the backend."""
    try:
        provider_info = _inference.provider_info()
        backend = provider_info.get("backend", "unknown")
        model_loaded = bool(provider_info.get("loaded"))
    except Exception as exc:
        logger.exception("Health provider probe failed: %s", exc)
        backend = "unknown"
        model_loaded = False

    camera_available = _camera.is_available
    if not camera_available and _settings.CAMERA_INDEX is not None:
        try:
            camera_available = _camera.start() or _camera.is_available
        except Exception:
            camera_available = False

    degraded = (not model_loaded)
    unhealthy = False
    if degraded and camera_available is False and _settings.CAMERA_INDEX is not None:
        unhealthy = False  # still partially functional

    if unhealthy:
        status_val = "unhealthy"
    elif degraded:
        status_val = "degraded"
    else:
        status_val = "healthy"

    return HealthResponse(
        status=status_val,
        version=_VERSION,
        uptime_seconds=max(0.0, time.perf_counter() - _STARTUP_TIME),
        backend=backend,
        camera_available=camera_available,
        model_loaded=model_loaded,
        details={
            "preferred_backend": _settings.PREFERRED_BACKEND,
            "fallback_backend": _settings.FALLBACK_BACKEND,
            "target_inference_fps": _settings.INFERENCE_FPS,
        },
    )


@router.get("/hardware", response_model=HardwareInfo, summary="Detected hardware capabilities")
async def get_hardware() -> HardwareInfo:
    try:
        return _hardware_detector.detect()
    except Exception as exc:
        logger.exception("Hardware detection failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hardware detection failed: {exc}",
        )


@router.get("/models", response_model=ModelList, summary="List inference providers / models")
async def get_models() -> ModelList:
    try:
        infos = _inference.all_provider_infos()
        models = [ModelInfo(**info) for info in infos]
        current_info = _inference.provider_info()
        current_name = current_info.get("name", "mock")
        return ModelList(current=current_name, models=models)
    except Exception as exc:
        logger.exception("Get models failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/metrics", response_model=MetricsResponse, summary="Performance metrics")
async def get_metrics() -> MetricsResponse:
    try:
        metrics = _inference.get_metrics()
        return MetricsResponse(**metrics)
    except Exception as exc:
        logger.exception("Metrics probe failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ----------------------------------------------------------------------
# Vision endpoints
# ----------------------------------------------------------------------
@router.post("/vision/detect", response_model=DetectionResponse, summary="Run object detection on an image")
async def vision_detect(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
) -> DetectionResponse:
    t0 = time.perf_counter()
    try:
        data = await image.read()
        frame = _decode_image_bytes(data)
        detections = await _detector.detect_objects_async(frame)
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        h, w = frame.shape[:2]
        info = _inference.provider_info()
        return DetectionResponse(
            success=True,
            count=len(detections),
            backend=info.get("backend", "unknown"),
            latency_ms=round(latency_ms, 2),
            detections=detections,
            image_width=w,
            image_height=h,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("vision/detect failed: %s", exc)
        info = _inference.provider_info()
        return DetectionResponse(
            success=False,
            count=0,
            backend=info.get("backend", "unknown"),
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            detections=[],
            error=str(exc),
        )


@router.post("/vision/ocr", response_model=OCRResponse, summary="Run OCR on an image")
async def vision_ocr(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
) -> OCRResponse:
    t0 = time.perf_counter()
    try:
        data = await image.read()
        frame = _decode_image_bytes(data)
        results = await _ocr.run_ocr_async(frame)
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        return OCRResponse(
            success=True,
            backend=_ocr.backend,
            latency_ms=round(latency_ms, 2),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("vision/ocr failed: %s", exc)
        return OCRResponse(
            success=False,
            backend=_ocr.backend,
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            results=[],
            error=str(exc),
        )


@router.post("/vision/describe", response_model=DescribeResponse, summary="Describe a scene / still image")
async def vision_describe(
    image: UploadFile = File(..., description="Image file (JPEG or PNG)"),
) -> DescribeResponse:
    t0 = time.perf_counter()
    try:
        data = await image.read()
        frame = _decode_image_bytes(data)
        description: SceneDescription = await _inference.run_describe_async(frame)
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        info = _inference.provider_info()
        return DescribeResponse(
            success=True,
            backend=info.get("backend", "unknown"),
            latency_ms=round(latency_ms, 2),
            description=description,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("vision/describe failed: %s", exc)
        info = _inference.provider_info()
        return DescribeResponse(
            success=False,
            backend=info.get("backend", "unknown"),
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            description=SceneDescription(summary="", detailed=""),
            error=str(exc),
        )


# ----------------------------------------------------------------------
# Chat endpoint
# ----------------------------------------------------------------------
@router.post("/chat/query", response_model=ChatResponse, summary="Answer a question about the scene / detections")
async def chat_query(payload: ChatQuery) -> ChatResponse:
    t0 = time.perf_counter()
    try:
        reply = await _inference.run_chat_async(payload.text, payload.context)
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        info = _inference.provider_info()
        return ChatResponse(
            success=True,
            backend=info.get("backend", "unknown"),
            latency_ms=round(latency_ms, 2),
            reply=reply,
            sources=["local-rules", "cached-context"],
        )
    except Exception as exc:
        logger.exception("chat/query failed: %s", exc)
        info = _inference.provider_info()
        return ChatResponse(
            success=False,
            backend=info.get("backend", "unknown"),
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            reply="",
            error=str(exc),
        )


# ----------------------------------------------------------------------
# Speech endpoint
# ----------------------------------------------------------------------
@router.post("/speech/transcribe", response_model=TranscribeResponse, summary="Transcribe an audio clip to text")
async def speech_transcribe(
    audio: UploadFile = File(..., description="Audio file (wav/mp3/webm or raw bytes)"),
    language: str | None = Form(default=None),
    sample_rate_hz: int | None = Form(default=None),
) -> TranscribeResponse:
    t0 = time.perf_counter()
    try:
        audio_bytes = await audio.read()
        req = TranscribeRequest(
            language=language,
            sample_rate_hz=sample_rate_hz,
        )
        text, lang_used = await _stt.transcribe_async(
            audio_bytes,
            language=req.language,
            sample_rate_hz=req.sample_rate_hz,
        )
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        return TranscribeResponse(
            success=True,
            backend=_stt.backend,
            latency_ms=round(latency_ms, 2),
            text=text,
            language=lang_used,
            segments=[],
        )
    except Exception as exc:
        logger.exception("speech/transcribe failed: %s", exc)
        return TranscribeResponse(
            success=False,
            backend=_stt.backend,
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            text="",
            error=str(exc),
        )


# ----------------------------------------------------------------------
# Safety endpoint
# ----------------------------------------------------------------------
@router.post("/safety/analyze", response_model=SafetyAnalyzeResponse, summary="Analyze detections for safety alerts")
async def safety_analyze(payload: SafetyAnalyzeRequest) -> SafetyAnalyzeResponse:
    t0 = time.perf_counter()
    try:
        alerts: List[SafetyAlert] = _safety.analyze_frame(
            detections=payload.detections,
            temporal_history=payload.include_history,
        )
        overall = _safety.overall_severity(alerts)
        summary = _safety.summary(alerts)
        latency_ms = 1000.0 * (time.perf_counter() - t0)
        return SafetyAnalyzeResponse(
            success=True,
            latency_ms=round(latency_ms, 2),
            alerts=alerts,
            summary=summary,
            overall_severity=overall,
        )
    except Exception as exc:
        logger.exception("safety/analyze failed: %s", exc)
        return SafetyAnalyzeResponse(
            success=False,
            latency_ms=round(1000.0 * (time.perf_counter() - t0), 2),
            alerts=[],
            summary="",
            overall_severity="info",
            error=str(exc),
        )


# ----------------------------------------------------------------------
# Snapshot endpoint
# ----------------------------------------------------------------------
@router.post("/camera/snapshot", response_model=SnapshotResponse, summary="Capture a camera snapshot (JPEG)")
async def camera_snapshot(
    payload: SnapshotRequest | None = None,
) -> Response:
    """
    Capture a JPEG snapshot from the camera.

    Returns the raw JPEG bytes in the HTTP body with appropriate Content-Type.
    """
    try:
        payload = payload or SnapshotRequest()
        index = payload.camera_index if payload.camera_index is not None else _settings.CAMERA_INDEX
        if payload.camera_index is not None or not _camera.is_available:
            _camera.stop()
            if payload.camera_index is not None:
                _camera.start(index)
            else:
                _camera.start()
        w = payload.width or _settings.CAMERA_WIDTH
        h = payload.height or _settings.CAMERA_HEIGHT
        frame = _camera.read_frame(force=True)
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Camera unavailable or did not return a frame.",
            )
        if payload.annotate:
            dets = _detector.detect_objects(frame)
            jpeg = _detector.render_annotated_jpeg(frame, dets, quality=payload.quality)
        else:
            jpeg = _camera.capture_snapshot(
                quality=payload.quality,
                width=w,
                height=h,
                annotate_frame=frame,
            )
        if not jpeg:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "JPEG encoding returned empty buffer.")
        resp = Response(content=jpeg, media_type="image/jpeg")
        resp.headers["X-Camera-Index"] = str(index)
        resp.headers["X-Image-Width"] = str(frame.shape[1])
        resp.headers["X-Image-Height"] = str(frame.shape[0])
        resp.headers["X-Jpeg-Size-Bytes"] = str(len(jpeg))
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("camera/snapshot failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ----------------------------------------------------------------------
# Exported instances for main.py / ws manager
# ----------------------------------------------------------------------
def get_service_instances() -> Dict[str, Any]:
    """Return singleton service objects used by the WebSocket layer."""
    return {
        "inference": _inference,
        "camera": _camera,
        "detector": _detector,
        "ocr": _ocr,
        "stt": _stt,
        "safety": _safety,
        "hardware": _hardware_detector,
    }


__all__ = [
    "router",
    "get_service_instances",
]
