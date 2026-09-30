from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.schemas import (
    DetectionBox,
    OCRResult,
    SceneDescription,
)
from ..utils.config import get_settings
from ..utils.logging import setup_logger
from ..utils.metrics import RollingMetrics
from .providers import Detection, ModelProvider, ProviderFactory


logger = setup_logger(__name__)


class InferenceManager:
    """
    Thread-safe singleton facade around the active ModelProvider.

    Responsibilities:
      * Owns the lifetime of the provider selected by ProviderFactory.
      * Tracks per-call latency and FPS metrics via RollingMetrics.
      * Implements configurable frame-throttling (min inter-call interval)
        so callers can spam ``run_detect`` at camera rate while we only
        actually run inference at the configured ``INFERENCE_FPS``.
      * Caches the last detection / OCR / description / transcription
        so throttled frames can still return data without crashing.
      * Serialises calls to the (potentially non-threadsafe) provider
        via a single reentrant lock.
    """

    _instance: Optional["InferenceManager"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.RLock()
        self._provider: Optional[ModelProvider] = None
        self._factory = ProviderFactory()
        self._metrics = RollingMetrics(window_size=200)
        self._frames_processed: int = 0
        self._min_interval: float = 1.0 / max(1, self._settings.INFERENCE_FPS)
        self._last_inference_ts: float = 0.0

        self._last_detections: List[Detection] = []
        self._last_ocr: List[OCRResult] = []
        self._last_description: Optional[SceneDescription] = None
        self._last_transcribe: str = ""

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "InferenceManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> ModelProvider:
        """
        Initialise the provider (idempotent).

        Called automatically on first inference call, but applications are
        encouraged to call it explicitly during startup so model load time
        is accounted for in the readiness check.
        """
        with self._lock:
            if self._provider is not None and self._provider.loaded:
                return self._provider
            logger.info("InferenceManager.start() building provider via ProviderFactory.")
            self._provider = self._factory.create()
            if not self._provider.loaded:
                logger.warning(
                    "Provider '%s' reports loaded=False after factory creation.",
                    self._provider.name,
                )
            return self._provider

    def shutdown(self) -> None:
        with self._lock:
            self._provider = None
            self._last_detections = []
            self._last_ocr = []
            self._last_description = None
            self._last_transcribe = ""
            self._frames_processed = 0
            self._last_inference_ts = 0.0
            logger.info("InferenceManager.shutdown() complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def provider(self) -> ModelProvider:
        if self._provider is None:
            return self.start()
        return self._provider

    def _should_run_inference(self, now: float) -> bool:
        return (now - self._last_inference_ts) >= (self._min_interval - 1e-6)

    def _filter_detections(self, detections: List[Detection]) -> List[Detection]:
        threshold = self._settings.CONFIDENCE_THRESHOLD
        if threshold <= 0:
            return detections
        return [d for d in detections if d.confidence >= threshold]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def run_detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """
        Run object detection (subject to throttling) and return schema objects.

        If throttling skips inference, the cached last detections are
        returned with a confidence penalty of 1% so callers can distinguish
        stale results if they wish.
        """
        self.start()
        assert self._provider is not None
        now = time.perf_counter()
        with self._lock:
            if self._should_run_inference(now):
                t0 = time.perf_counter()
                try:
                    raw = self._provider.detect(frame)
                except Exception as exc:
                    logger.exception("provider.detect raised: %s", exc)
                    raw = []
                latency = time.perf_counter() - t0
                self._metrics.record_latency(latency)
                self._metrics.record_frame()
                self._frames_processed += 1
                self._last_inference_ts = now
                self._last_detections = self._filter_detections(raw)
            else:
                self._last_detections = [
                    Detection(
                        label=d.label,
                        confidence=max(0.0, d.confidence - 0.01),
                        bbox=d.bbox,
                        class_id=d.class_id,
                    )
                    for d in self._last_detections
                ]
            return [d.to_schema() for d in self._last_detections]

    async def run_detect_async(self, frame: np.ndarray) -> List[DetectionBox]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_detect, frame)

    # ------------------------------------------------------------------
    # Describe
    # ------------------------------------------------------------------
    def run_describe(self, frame: np.ndarray) -> SceneDescription:
        self.start()
        assert self._provider is not None
        t0 = time.perf_counter()
        with self._lock:
            try:
                desc = self._provider.describe(frame)
            except Exception as exc:
                logger.exception("provider.describe raised: %s", exc)
                desc = SceneDescription(summary="", detailed="")
            latency = time.perf_counter() - t0
            self._metrics.record_latency(latency)
            self._metrics.record_frame()
            self._frames_processed += 1
            self._last_description = desc
            return desc

    async def run_describe_async(self, frame: np.ndarray) -> SceneDescription:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_describe, frame)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def run_ocr(self, frame: np.ndarray) -> List[OCRResult]:
        self.start()
        assert self._provider is not None
        t0 = time.perf_counter()
        with self._lock:
            try:
                results = self._provider.ocr(frame)
            except Exception as exc:
                logger.exception("provider.ocr raised: %s", exc)
                results = []
            latency = time.perf_counter() - t0
            self._metrics.record_latency(latency)
            self._metrics.record_frame()
            self._frames_processed += 1
            self._last_ocr = list(results)
            return results

    async def run_ocr_async(self, frame: np.ndarray) -> List[OCRResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_ocr, frame)

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------
    def run_transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
    ) -> str:
        self.start()
        assert self._provider is not None
        t0 = time.perf_counter()
        with self._lock:
            try:
                text = self._provider.transcribe(audio, language, sample_rate_hz)
            except Exception as exc:
                logger.exception("provider.transcribe raised: %s", exc)
                text = ""
            latency = time.perf_counter() - t0
            self._metrics.record_latency(latency)
            self._metrics.record_frame()
            self._frames_processed += 1
            self._last_transcribe = text
            return text

    async def run_transcribe_async(
        self,
        audio: bytes,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.run_transcribe(audio, language, sample_rate_hz)
        )

    # ------------------------------------------------------------------
    # Chat / query
    # ------------------------------------------------------------------
    def run_chat(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Lightweight rule-based chat handler driven by provider capabilities.

        The provider itself is not required to implement a "chat" primitive;
        instead this method composes context from the last detections/OCR/
        description and produces a best-effort reply without any network
        calls or LLM invocation.
        """
        self.start()
        with self._lock:
            t0 = time.perf_counter()
            q = (text or "").strip().lower()
            detections = [d.to_schema() for d in self._last_detections]
            ocr = [r.text for r in self._last_ocr]
            desc = self._last_description

            reply_parts: List[str] = []
            if not q:
                reply_parts.append("I didn't receive a question. Try asking 'What do you see?'")
            elif any(word in q for word in ("what", "see", "describe", "scene")):
                if desc and desc.summary:
                    reply_parts.append(desc.summary)
                if detections:
                    labels = sorted({d.label for d in detections})
                    reply_parts.append(
                        f"I can see {len(detections)} objects: " + ", ".join(labels) + "."
                    )
                else:
                    reply_parts.append("No objects have been detected yet.")
            elif any(word in q for word in ("read", "text", "ocr", "sign", "say")):
                if ocr:
                    reply_parts.append("I read the following text: " + "; ".join(ocr) + ".")
                else:
                    reply_parts.append("I don't currently see any readable text.")
            elif any(word in q for word in ("person", "people", "someone")):
                people = [d for d in detections if d.label.lower() == "person"]
                if people:
                    reply_parts.append(f"I see {len(people)} person(s) in the latest frame.")
                elif desc and desc.people_count:
                    reply_parts.append(f"Roughly {desc.people_count} person(s) reported by the scene describer.")
                else:
                    reply_parts.append("I do not see any people right now.")
            elif any(word in q for word in ("exit", "door", "leave", "escape")):
                exits = [r for r in ocr if "EXIT" in r.upper()]
                exit_objs = [d for d in detections if d.label.lower() in ("exit sign", "door")]
                if exits:
                    reply_parts.append("Found signs: " + ", ".join(exits) + ".")
                if exit_objs:
                    reply_parts.append(f"Detected {len(exit_objs)} exit sign(s)/door(s).")
                if not exits and not exit_objs:
                    reply_parts.append("I cannot see an exit sign or door right now.")
            elif any(word in q for word in ("obstacle", "hazard", "safe", "blocked")):
                if detections:
                    reply_parts.append(
                        "Current visible objects include: "
                        + ", ".join(sorted({d.label for d in detections}))
                        + ". Check safety/analyze for proximity alerts."
                    )
                else:
                    reply_parts.append("No obstacles detected in the latest frame.")
            else:
                reply_parts.append(
                    "I can help with: describing the scene, reading text, "
                    "counting people, finding the exit, and reporting obstacles. "
                    "Ask one of those!"
                )

            latency = time.perf_counter() - t0
            self._metrics.record_latency(latency)
            self._metrics.record_frame()
            self._frames_processed += 1
            return " ".join(reply_parts)

    async def run_chat_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run_chat(text, context))

    # ------------------------------------------------------------------
    # Metrics & introspection
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        snap = self._metrics.snapshot()
        load_ms = None
        if self._provider is not None:
            load_ms = self._provider.load_time_ms
        return {
            **snap,
            "provider": self._provider.name if self._provider else "",
            "backend": self._provider.backend if self._provider else "",
            "frames_processed": self._frames_processed,
            "model_load_time_ms": load_ms,
            "inference_fps_target": self._settings.INFERENCE_FPS,
            "throttle_interval_seconds": self._min_interval,
        }

    def provider_info(self) -> Dict[str, Any]:
        self.start()
        assert self._provider is not None
        return self._provider.info()

    def all_provider_infos(self) -> List[Dict[str, Any]]:
        infos: List[Dict[str, Any]] = []
        for name in ProviderFactory.available_backend_names():
            cls_provider = None
            try:
                if name == (self._provider.name if self._provider else ""):
                    cls_provider = self._provider
                else:
                    # Instantiate lightweight info via class defaults - do not
                    # force heavy model load for catalog purposes.
                    from .providers import _BACKEND_REGISTRY

                    cls = _BACKEND_REGISTRY[name]
                    info_defaults = {
                        "name": cls.name,
                        "backend": cls.backend,
                        "version": cls.version,
                        "device": cls.device,
                        "loaded": False,
                        "load_time_ms": None,
                        "supported_tasks": ["detect", "ocr", "describe", "transcribe"],
                        "description": cls.description,
                    }
                    infos.append(info_defaults)
                    continue
            except Exception as exc:
                logger.debug("Skip info for %s: %s", name, exc)
                continue
            if cls_provider is not None:
                infos.append(cls_provider.info())
        if self._provider is not None:
            current = self._provider.name
            if not any(info.get("name") == current for info in infos):
                infos.insert(0, self._provider.info())
        return infos


__all__ = ["InferenceManager"]
