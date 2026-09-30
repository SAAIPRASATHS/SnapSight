from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for /health endpoint reporting backend status."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall service health status"
    )
    version: str = Field(..., description="Backend version string")
    uptime_seconds: float = Field(..., ge=0, description="Uptime in seconds since service start")
    backend: str = Field(..., description="Active inference backend identifier")
    camera_available: bool = Field(..., description="Whether camera can be opened")
    model_loaded: bool = Field(..., description="Whether the inference model is loaded")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of response")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional status details")


class HardwareInfo(BaseModel):
    """Detected runtime hardware capabilities."""

    vendor: str = Field(default="AUTO_DETECT", description="Hardware vendor name")
    platform: str = Field(default="AUTO_DETECT", description="OS / platform identifier")
    model: str = Field(default="AUTO_DETECT", description="Hardware model name")
    cpu: str = Field(default="", description="CPU model / description string")
    gpu: str = Field(default="", description="GPU description or empty string if unavailable")
    npu: str = Field(default="", description="NPU / AI accelerator description or empty string")
    ram_gb: float = Field(default=0.0, ge=0, description="Total installed RAM in gigabytes")
    available_backends: List[str] = Field(default_factory=list, description="Available inference backends on this host")
    notes: str = Field(default="", description="Human-readable notes about hardware detection")


class ModelInfo(BaseModel):
    """Metadata about a single inference provider / model."""

    name: str = Field(..., description="Short identifier for this model/provider")
    backend: str = Field(..., description="Backend technology (mock, onnx, pytorch, qaihub)")
    version: str = Field(default="unknown", description="Version or revision of the model")
    device: str = Field(default="cpu", description="Execution device: cpu, gpu, npu, tpu")
    loaded: bool = Field(default=False, description="Whether the model is currently loaded in memory")
    load_time_ms: Optional[float] = Field(default=None, ge=0, description="Model load duration in ms (if loaded)")
    supported_tasks: List[str] = Field(
        default_factory=lambda: ["detect", "ocr", "describe", "transcribe"],
        description="Inference tasks this provider supports",
    )
    description: str = Field(default="", description="Human-readable description of the provider")


class ModelList(BaseModel):
    """Container of all available models / providers."""

    current: str = Field(..., description="Name of currently selected model/provider")
    models: List[ModelInfo] = Field(default_factory=list, description="All configured providers")


class BoundingBox(BaseModel):
    """Normalised or absolute bounding box for a detected object."""

    xmin: float = Field(..., description="Left coordinate (pixels or 0..1)")
    ymin: float = Field(..., description="Top coordinate (pixels or 0..1)")
    xmax: float = Field(..., description="Right coordinate (pixels or 0..1)")
    ymax: float = Field(..., description="Bottom coordinate (pixels or 0..1)")
    normalized: bool = Field(default=False, description="True if coordinates are in 0..1 range")

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        if h <= 0:
            return 0.0
        return self.width / h


class DetectionBox(BaseModel):
    """A single object detection result."""

    label: str = Field(..., description="Object class label, e.g. 'person'")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box of the detection")
    class_id: Optional[int] = Field(default=None, description="Optional numeric class id")


class DetectionResponse(BaseModel):
    """Response for POST /vision/detect."""

    success: bool = Field(default=True)
    count: int = Field(default=0, ge=0, description="Number of detections returned")
    backend: str = Field(..., description="Backend that produced these detections")
    latency_ms: float = Field(default=0.0, ge=0, description="Inference latency in milliseconds")
    detections: List[DetectionBox] = Field(default_factory=list)
    image_width: Optional[int] = Field(default=None, ge=0)
    image_height: Optional[int] = Field(default=None, ge=0)
    error: Optional[str] = Field(default=None, description="Error message if success is False")


class OCRResult(BaseModel):
    """A single OCR text extraction."""

    text: str = Field(..., description="Recognised text string")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Recognition confidence")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box of the text region")


class OCRResponse(BaseModel):
    """Response for POST /vision/ocr."""

    success: bool = Field(default=True)
    backend: str = Field(...)
    latency_ms: float = Field(default=0.0, ge=0)
    results: List[OCRResult] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


class SceneDescription(BaseModel):
    """Structured description of a scene / still image."""

    summary: str = Field(..., description="Short one-line scene summary")
    detailed: str = Field(default="", description="Longer detailed description")
    people_count: int = Field(default=0, ge=0, description="Estimated number of people in scene")
    objects: List[str] = Field(default_factory=list, description="Key objects identified in scene")
    scene_type: str = Field(default="indoor", description="Scene category: indoor/outdoor etc.")
    lighting: str = Field(default="normal", description="Lighting condition keyword")


class DescribeResponse(BaseModel):
    """Response for POST /vision/describe."""

    success: bool = Field(default=True)
    backend: str = Field(...)
    latency_ms: float = Field(default=0.0, ge=0)
    description: SceneDescription
    error: Optional[str] = Field(default=None)


class ChatQuery(BaseModel):
    """User query sent to POST /chat/query."""

    text: str = Field(..., min_length=1, max_length=2000, description="User prompt text")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context: last detections, OCR results, scene description etc.",
    )
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")


class ChatResponse(BaseModel):
    """Response for POST /chat/query."""

    success: bool = Field(default=True)
    backend: str = Field(...)
    latency_ms: float = Field(default=0.0, ge=0)
    reply: str = Field(..., description="Assistant reply text")
    sources: List[str] = Field(default_factory=list, description="Optional provenance tags")
    error: Optional[str] = Field(default=None)


class TranscribeRequest(BaseModel):
    """Request body / params for POST /speech/transcribe."""

    language: Optional[str] = Field(default="en", description="Optional ISO language hint")
    sample_rate_hz: Optional[int] = Field(default=None, ge=8000, le=48000, description="Audio sample rate hint")


class TranscribeResponse(BaseModel):
    """Response for POST /speech/transcribe."""

    success: bool = Field(default=True)
    backend: str = Field(...)
    latency_ms: float = Field(default=0.0, ge=0)
    text: str = Field(default="", description="Transcribed text")
    language: Optional[str] = Field(default=None, description="Detected / used language code")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Optional segmented output")
    error: Optional[str] = Field(default=None)


class SafetyAlert(BaseModel):
    """Single safety alert generated by SafetyAnalyzer."""

    severity: Literal["info", "warning", "critical"] = Field(
        ..., description="Alert severity level"
    )
    category: str = Field(..., description="Short category tag: 'fall', 'obstacle', 'smoke', 'person', 'other'")
    message: str = Field(..., description="Human readable alert message")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Confidence in this alert")
    related_label: Optional[str] = Field(default=None, description="Related detection label if applicable")
    bbox: Optional[BoundingBox] = Field(default=None, description="Region of interest if applicable")
    requires_confirmation: bool = Field(default=True, description="Whether UI should ask user to confirm")
    temporal_confirmations: int = Field(default=0, ge=0, description="Consecutive frames this alert was raised")


class SafetyAnalyzeRequest(BaseModel):
    """Request shape for POST /safety/analyze."""

    detections: List[DetectionBox] = Field(default_factory=list)
    include_history: bool = Field(default=True, description="True to enable temporal analysis")


class SafetyAnalyzeResponse(BaseModel):
    """Response for POST /safety/analyze."""

    success: bool = Field(default=True)
    latency_ms: float = Field(default=0.0, ge=0)
    alerts: List[SafetyAlert] = Field(default_factory=list)
    summary: str = Field(default="", description="Human readable summary string")
    overall_severity: Literal["info", "warning", "critical"] = Field(default="info")
    error: Optional[str] = Field(default=None)


class MetricsResponse(BaseModel):
    """Response for GET /metrics endpoint."""

    fps: float = Field(default=0.0, ge=0, description="Measured inference FPS")
    latency: Dict[str, Any] = Field(default_factory=dict, description="Latency distribution statistics")
    process: Dict[str, Any] = Field(default_factory=dict, description="Process CPU/RAM samples")
    provider: str = Field(default="", description="Active provider name")
    backend: str = Field(default="", description="Active backend name")
    frames_processed: int = Field(default=0, ge=0, description="Cumulative frames processed")
    model_load_time_ms: Optional[float] = Field(default=None, description="Model load time if available")
    window_size: int = Field(default=0, ge=0, description="Size of the rolling metrics window")


class SnapshotRequest(BaseModel):
    """Request for capturing a still snapshot from the camera."""

    camera_index: Optional[int] = Field(default=None, description="Override default camera index")
    width: Optional[int] = Field(default=None, ge=160, description="Override capture width")
    height: Optional[int] = Field(default=None, ge=120, description="Override capture height")
    annotate: bool = Field(default=False, description="True to draw detection boxes on the snapshot")
    quality: int = Field(default=90, ge=1, le=100, description="JPEG quality 1-100")


class SnapshotResponse(BaseModel):
    """Response metadata for a snapshot request."""

    success: bool = Field(default=True)
    camera_index: int = Field(...)
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)
    jpeg_size_bytes: int = Field(default=0, ge=0)
    error: Optional[str] = Field(default=None)


class LiveFrameMessage(BaseModel):
    """WebSocket message payload broadcast over /ws/live."""

    type: Literal["frame", "detections", "alerts", "status", "error"] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    backend: Optional[str] = Field(default=None)
    detections: Optional[List[DetectionBox]] = Field(default=None)
    alerts: Optional[List[SafetyAlert]] = Field(default=None)
    status: Optional[Dict[str, Any]] = Field(default=None)
    jpeg_base64: Optional[str] = Field(default=None, description="Optional base64 encoded JPEG frame")
    error: Optional[str] = Field(default=None)


__all__ = [
    "BoundingBox",
    "ChatQuery",
    "ChatResponse",
    "DescribeResponse",
    "DetectionBox",
    "DetectionResponse",
    "HardwareInfo",
    "HealthResponse",
    "LiveFrameMessage",
    "MetricsResponse",
    "ModelInfo",
    "ModelList",
    "OCRResponse",
    "OCRResult",
    "SafetyAlert",
    "SafetyAnalyzeRequest",
    "SafetyAnalyzeResponse",
    "SceneDescription",
    "SnapshotRequest",
    "SnapshotResponse",
    "TranscribeRequest",
    "TranscribeResponse",
]
