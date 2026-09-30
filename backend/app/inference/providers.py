from __future__ import annotations

import importlib.util
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.schemas import BoundingBox, DetectionBox, OCRResult, SceneDescription
from ..utils.config import get_settings
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


@dataclass
class Detection:
    """Internal detection representation used before conversion to Pydantic."""

    label: str
    confidence: float
    bbox: BoundingBox
    class_id: Optional[int] = None

    def to_schema(self) -> DetectionBox:
        return DetectionBox(
            label=self.label,
            confidence=self.confidence,
            bbox=self.bbox,
            class_id=self.class_id,
        )


class ModelProvider(ABC):
    """
    Abstract base class for all inference backends.

    Concrete subclasses implement the four core tasks: detect, describe,
    ocr and transcribe. Each method receives raw numpy/image or audio
    bytes and must return a domain object. Providers are expected to be
    long-lived and to cache heavy model state once :meth:`load` has been
    invoked.
    """

    name: str = "base"
    backend: str = "base"
    device: str = "cpu"
    version: str = "unknown"
    description: str = "Base ModelProvider - do not use directly"

    def __init__(self) -> None:
        self._loaded: bool = False
        self._load_time_ms: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def load(self) -> bool:
        """
        Load model weights and runtime state.

        Implementations MUST be safe to call multiple times (idempotent)
        and return True on successful load, False otherwise.
        """

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_time_ms(self) -> Optional[float]:
        return self._load_time_ms

    @property
    def supported_tasks(self) -> List[str]:
        return ["detect", "ocr", "describe", "transcribe"]

    # ------------------------------------------------------------------
    # Inference primitives
    # ------------------------------------------------------------------
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a BGR uint8 numpy frame.

        Args:
            frame: HxWx3 numpy array in BGR channel order (OpenCV convention).

        Returns:
            List of :class:`Detection` objects. Empty list means nothing
            detected or an internal non-fatal error occurred.
        """

    @abstractmethod
    def describe(self, frame: np.ndarray) -> SceneDescription:
        """
        Produce a structured scene description of *frame*.
        """

    @abstractmethod
    def ocr(self, frame: np.ndarray) -> List[OCRResult]:
        """
        Run optical character recognition on *frame*.
        """

    @abstractmethod
    def transcribe(self, audio: bytes, language: Optional[str] = None,
                   sample_rate_hz: Optional[int] = None) -> str:
        """
        Transcribe audio bytes (any reasonable format) to text.

        Args:
            audio: Raw audio bytes.
            language: Optional ISO-639-1 language code hint.
            sample_rate_hz: Optional sample rate hint in Hz.

        Returns:
            Transcribed text string. Empty string on failure.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "version": self.version,
            "device": self.device,
            "loaded": self.loaded,
            "load_time_ms": self.load_time_ms,
            "supported_tasks": self.supported_tasks,
            "description": self.description,
        }


# ----------------------------------------------------------------------
# MockProvider - default, always available, realistic but synthetic data
# ----------------------------------------------------------------------
class MockProvider(ModelProvider):
    """
    Deterministic-ish mock inference backend. Always loads successfully.

    Used as the default and fallback provider. Produces realistic detection
    boxes, OCR strings, scene descriptions and transcriptions so the rest
    of the stack can be exercised without any ML runtime installed.
    """

    name = "mock"
    backend = "mock"
    device = "cpu"
    version = "1.0.0"
    description = (
        "Synthetic mock backend for development and testing. "
        "Returns realistic but deterministic-looking detections and text."
    )

    _COCO_STYLE_LABELS: Sequence[Tuple[str, float]] = (
        ("person", 0.94),
        ("chair", 0.91),
        ("door", 0.87),
        ("table", 0.82),
        ("exit sign", 0.79),
        ("fire extinguisher", 0.75),
        ("backpack", 0.68),
        ("bottle", 0.62),
        ("cell phone", 0.58),
        ("suitcase", 0.54),
    )

    _OCR_STRINGS: Sequence[Tuple[str, float]] = (
        ("EXIT", 0.96),
        ("EMERGENCY EXIT", 0.93),
        ("PUSH", 0.88),
        ("FIRE EXTINGUISHER", 0.85),
        ("KEEP CLEAR", 0.80),
        ("WASH HANDS", 0.77),
        ("STAIRS", 0.74),
        ("CAUTION WET FLOOR", 0.71),
    )

    _SCENE_TEMPLATES: Sequence[Tuple[str, str, int, Sequence[str], str, str]] = (
        (
            "A person standing near a table with chairs in an indoor office corridor.",
            "The scene shows a well-lit indoor corridor with a single person standing next to a "
            "wooden table surrounded by chairs. An exit sign is visible on the back wall "
            "alongside a fire extinguisher. The floor appears clean and there are no "
            "obstacles blocking the main walkway.",
            1,
            ["person", "chair", "table", "exit sign", "fire extinguisher"],
            "indoor",
            "normal",
        ),
        (
            "A room with multiple chairs, a table and bags left unattended.",
            "The scene depicts a brightly lit indoor area, likely a break room or meeting space. "
            "Several chairs are arranged around a central table. A backpack and a suitcase rest "
            "on the floor near the table wall. An exit sign above the door provides clear "
            "egress guidance.",
            0,
            ["chair", "table", "backpack", "suitcase", "door"],
            "indoor",
            "bright",
        ),
        (
            "Outdoor walkway with person approaching a door.",
            "A daytime outdoor scene near a building entrance. A person is walking towards a "
            "door which is marked with a push sign. The lighting is natural daylight. "
            "No other pedestrians are visible in the immediate area.",
            1,
            ["person", "door"],
            "outdoor",
            "daylight",
        ),
    )

    _TRANSCRIBE_TEMPLATES: Sequence[str] = (
        "What do you see?",
        "Read the text in front of me.",
        "Describe the scene.",
        "Is there any obstacle near me?",
        "Find the exit sign.",
        "How many people are there?",
        "Is it safe to move forward?",
        "Tell me about the room.",
    )

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def load(self) -> bool:
        if self._loaded:
            return True
        t0 = time.perf_counter()
        self._loaded = True
        self._load_time_ms = 1000.0 * (time.perf_counter() - t0) + 12.5
        logger.info("MockProvider loaded in %.1fms", self._load_time_ms)
        return True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded:
            self.load()
        h, w = frame.shape[:2] if frame.ndim >= 2 else (480, 640)
        n = self._rng.randint(2, min(6, len(self._COCO_STYLE_LABELS)))
        chosen = self._rng.sample(list(self._COCO_STYLE_LABELS), n)
        detections: List[Detection] = []
        for idx, (label, base_conf) in enumerate(chosen):
            x = (idx * 37) % max(1, w - 160)
            y = (idx * 53) % max(1, h - 160)
            bw = min(w - x, self._rng.randint(80, 220))
            bh = min(h - y, self._rng.randint(90, 260))
            conf = max(0.5, base_conf - 0.02 * idx)
            detections.append(
                Detection(
                    label=label,
                    confidence=round(conf, 4),
                    bbox=BoundingBox(
                        xmin=float(x),
                        ymin=float(y),
                        xmax=float(x + bw),
                        ymax=float(y + bh),
                        normalized=False,
                    ),
                    class_id=idx,
                )
            )
        return detections

    def describe(self, frame: np.ndarray) -> SceneDescription:
        if not self._loaded:
            self.load()
        summary, detailed, people, objects, scene, lighting = self._rng.choice(
            self._SCENE_TEMPLATES
        )
        return SceneDescription(
            summary=summary,
            detailed=detailed,
            people_count=people,
            objects=list(objects),
            scene_type=scene,
            lighting=lighting,
        )

    def ocr(self, frame: np.ndarray) -> List[OCRResult]:
        if not self._loaded:
            self.load()
        h, w = frame.shape[:2] if frame.ndim >= 2 else (480, 640)
        n = self._rng.randint(1, min(3, len(self._OCR_STRINGS)))
        chosen = self._rng.sample(list(self._OCR_STRINGS), n)
        results: List[OCRResult] = []
        for i, (text, conf) in enumerate(chosen):
            x = (i * 97) % max(1, w - 300)
            y = (i * 41) % max(1, h - 100)
            bw = min(w - x, self._rng.randint(100, 260))
            bh = min(h - y, self._rng.randint(30, 70))
            results.append(
                OCRResult(
                    text=text,
                    confidence=round(conf, 3),
                    bbox=BoundingBox(
                        xmin=float(x),
                        ymin=float(y),
                        xmax=float(x + bw),
                        ymax=float(y + bh),
                        normalized=False,
                    ),
                )
            )
        return results

    def transcribe(self, audio: bytes, language: Optional[str] = None,
                   sample_rate_hz: Optional[int] = None) -> str:
        if not self._loaded:
            self.load()
        if not audio:
            return ""
        idx = self._rng.randrange(len(self._TRANSCRIBE_TEMPLATES))
        return self._TRANSCRIBE_TEMPLATES[idx]


# ----------------------------------------------------------------------
# ONNXProvider - skeleton with graceful fallback
# ----------------------------------------------------------------------
class ONNXProvider(ModelProvider):
    """
    ONNX Runtime inference backend skeleton.

    Probes for ``onnxruntime`` and optionally ``onnxruntime-gpu`` / QNN
    execution providers. If the runtime is installed but no model path is
    configured, the provider logs a clear warning and falls back to
    returning empty / mock-equivalent data rather than crashing. It is the
    responsibility of the deployer to place model files at the configured
    path or extend this class with model-specific pre/post-processing.
    """

    name = "onnx"
    backend = "onnx"
    version = "0.1.0-skeleton"
    description = (
        "ONNX Runtime backend skeleton. Model loading and pre/post-processing "
        "are stubs that degrade gracefully when onnxruntime or model files "
        "are unavailable. Reports ACTUAL providers (CPU/CUDA/QNN) via info()."
    )

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        self._model_path = model_path
        self._session: Any = None
        self._providers: List[str] = []

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    def load(self) -> bool:
        if self._loaded:
            return True
        t0 = time.perf_counter()
        if not self.is_available():
            logger.error("ONNXProvider: onnxruntime package not installed. Cannot load.")
            return False
        try:
            import onnxruntime as ort  # type: ignore

            self._providers = ort.get_available_providers()
            self.device = "npu" if any("QNN" in p.upper() for p in self._providers) else (
                "gpu" if any("CUDA" in p.upper() for p in self._providers) else "cpu"
            )
            if self._model_path:
                try:
                    self._session = ort.InferenceSession(
                        self._model_path,
                        providers=self._providers,
                    )
                    logger.info(
                        "ONNXProvider loaded model from %s with providers %s",
                        self._model_path,
                        self._session.get_providers(),
                    )
                except Exception as exc:
                    logger.warning(
                        "ONNXProvider could not load model at %s: %s. "
                        "Operating in degraded mode (no inference).",
                        self._model_path,
                        exc,
                    )
                    self._session = None
            else:
                logger.info(
                    "ONNXProvider: onnxruntime available (providers=%s). "
                    "No model_path configured; running in skeleton mode.",
                    self._providers,
                )
            self._loaded = True
            self._load_time_ms = 1000.0 * (time.perf_counter() - t0)
            return True
        except Exception as exc:
            logger.error("ONNXProvider load failed: %s", exc)
            return False

    def _degraded(self, what: str) -> None:
        logger.warning("ONNXProvider: %s is unavailable - session not loaded.", what)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded:
            self.load()
        if self._session is None:
            self._degraded("detect")
            return []
        try:
            # --- Integration point: model-specific pre/post processing ---
            # input_name = self._session.get_inputs()[0].name
            # tensor = _preprocess(frame)
            # outputs = self._session.run(None, {input_name: tensor})
            # return _postprocess(outputs)
            logger.warning("ONNXProvider.detect: post-processing skeleton not implemented.")
            return []
        except Exception as exc:
            logger.error("ONNXProvider.detect failed: %s", exc)
            return []

    def describe(self, frame: np.ndarray) -> SceneDescription:
        if not self._loaded:
            self.load()
        self._degraded("describe")
        return SceneDescription(
            summary="Scene description via ONNX provider not implemented.",
            detailed="",
            people_count=0,
            objects=[],
            scene_type="indoor",
            lighting="normal",
        )

    def ocr(self, frame: np.ndarray) -> List[OCRResult]:
        if not self._loaded:
            self.load()
        self._degraded("ocr")
        return []

    def transcribe(self, audio: bytes, language: Optional[str] = None,
                   sample_rate_hz: Optional[int] = None) -> str:
        if not self._loaded:
            self.load()
        self._degraded("transcribe")
        return ""


# ----------------------------------------------------------------------
# PyTorchProvider - skeleton with graceful fallback
# ----------------------------------------------------------------------
class PyTorchProvider(ModelProvider):
    """
    PyTorch / torchvision backend skeleton.

    Supports loading a YOLO-style detector via torch.hub or a local model
    file. If torch/torchvision are not installed, :meth:`load` returns
    False and the ProviderFactory will try the fallback backend. The
    detection / description / OCR / transcribe methods are skeletons with
    clear integration TODOs for real model weights.
    """

    name = "pytorch"
    backend = "pytorch"
    version = "0.1.0-skeleton"
    description = (
        "PyTorch + torchvision backend skeleton. Supports CUDA where available "
        "and falls back to CPU. Model weights are not bundled; deploy a YOLO / "
        "vision model and extend pre/post-processing to use."
    )

    def __init__(self, model_name: str = "yolov5s", model_path: Optional[str] = None) -> None:
        super().__init__()
        self._model_name = model_name
        self._model_path = model_path
        self._model: Any = None
        self._torch_device: Optional[str] = None

    @staticmethod
    def is_available() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("torchvision") is not None
        )

    def load(self) -> bool:
        if self._loaded:
            return True
        t0 = time.perf_counter()
        if not self.is_available():
            logger.error("PyTorchProvider: torch / torchvision not installed.")
            return False
        try:
            import torch  # type: ignore

            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = "gpu" if self._torch_device == "cuda" else "cpu"
            try:
                if self._model_path:
                    # Example integration point: torch.load(local_checkpoint)
                    logger.info(
                        "PyTorchProvider: local model_path configured (%s). "
                        "Weights loading skeleton - extend to load checkpoint.",
                        self._model_path,
                    )
                else:
                    logger.info(
                        "PyTorchProvider: attempting torch.hub YOLOv5 skeleton load "
                        "(requires network). Installing weights locally is recommended.",
                    )
                    try:
                        model = torch.hub.load(
                            "ultralytics/yolov5", self._model_name, pretrained=False,
                            trust_repo=True,
                        )
                        self._model = model.to(self._torch_device)
                    except Exception as exc:
                        logger.warning(
                            "PyTorchProvider: torch.hub YOLOv5 load skipped: %s. "
                            "Running in skeleton mode.",
                            exc,
                        )
                self._loaded = True
                self._load_time_ms = 1000.0 * (time.perf_counter() - t0)
                return True
            except Exception as exc:
                logger.error("PyTorchProvider model init failed: %s", exc)
                return False
        except Exception as exc:
            logger.error("PyTorchProvider load failed: %s", exc)
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded:
            self.load()
        if self._model is None:
            logger.warning("PyTorchProvider.detect: model not loaded.")
            return []
        try:
            # Integration point: pass frame through model and parse outputs
            # results = self._model(frame)
            # return _parse_yolo(results)
            logger.warning("PyTorchProvider.detect: output parsing not implemented.")
            return []
        except Exception as exc:
            logger.error("PyTorchProvider.detect failed: %s", exc)
            return []

    def describe(self, frame: np.ndarray) -> SceneDescription:
        if not self._loaded:
            self.load()
        logger.warning("PyTorchProvider.describe not implemented.")
        return SceneDescription(summary="", detailed="")

    def ocr(self, frame: np.ndarray) -> List[OCRResult]:
        if not self._loaded:
            self.load()
        logger.warning("PyTorchProvider.ocr not implemented.")
        return []

    def transcribe(self, audio: bytes, language: Optional[str] = None,
                   sample_rate_hz: Optional[int] = None) -> str:
        if not self._loaded:
            self.load()
        logger.warning("PyTorchProvider.transcribe not implemented.")
        return ""


# ----------------------------------------------------------------------
# QualcommAIHubProvider - skeleton with clear runtime error
# ----------------------------------------------------------------------
class QualcommAIHubProvider(ModelProvider):
    """
    Qualcomm AI Hub (qai_hub) backend skeleton.

    Integration points:
      1. Compile / upload models via ``qai_hub.submit_model()``
      2. Download / run compiled models on-device via HubRunner / runtime APIs
      3. Profile performance with ``qai_hub.profile_model()``

    If ``qai_hub`` is not importable, :meth:`load` raises a clear
    RuntimeError (after logging) and reports *actual* backend used = N/A.
    """

    name = "qaihub"
    backend = "qaihub"
    device = "npu"
    version = "0.1.0-skeleton"
    description = (
        "Qualcomm AI Hub backend skeleton. Requires qai_hub package and a "
        "compiled model package. NPU execution backend is reported as 'npu' "
        "when the runtime is installed."
    )

    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__()
        self._model_id = model_id
        self._hub: Any = None
        self._runner: Any = None

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("qai_hub") is not None

    def load(self) -> bool:
        if self._loaded:
            return True
        t0 = time.perf_counter()
        if not self.is_available():
            msg = (
                "QualcommAIHubProvider: 'qai_hub' package is not installed. "
                "Install it with `pip install qai_hub` (requires Qualcomm AI "
                "Hub access) or set PREFERRED_BACKEND/FALLBACK_BACKEND to a "
                "provider that is available on this device."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        try:
            import qai_hub as hub  # type: ignore

            self._hub = hub
            logger.info(
                "QualcommAIHubProvider: qai_hub imported. model_id=%s. "
                "Integration points: submit_model / profile_model / HubRunner.",
                self._model_id,
            )
            if self._model_id:
                logger.info(
                    "QualcommAIHubProvider skeleton: model_id=%s configured. "
                    "Extend this method to instantiate HubRunner and load compiled package.",
                    self._model_id,
                )
            self._loaded = True
            self._load_time_ms = 1000.0 * (time.perf_counter() - t0)
            return True
        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("QualcommAIHubProvider load failed: %s", exc)
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded:
            self.load()
        if self._runner is None:
            logger.warning("QualcommAIHubProvider.detect: no compiled model runner loaded.")
            return []
        try:
            # inputs = _preprocess_qnn(frame)
            # outputs = self._runner(inputs)
            # return _parse(outputs)
            logger.warning("QualcommAIHubProvider.detect: runner integration not implemented.")
            return []
        except Exception as exc:
            logger.error("QualcommAIHubProvider.detect failed: %s", exc)
            return []

    def describe(self, frame: np.ndarray) -> SceneDescription:
        if not self._loaded:
            self.load()
        logger.warning("QualcommAIHubProvider.describe not implemented.")
        return SceneDescription(summary="", detailed="")

    def ocr(self, frame: np.ndarray) -> List[OCRResult]:
        if not self._loaded:
            self.load()
        logger.warning("QualcommAIHubProvider.ocr not implemented.")
        return []

    def transcribe(self, audio: bytes, language: Optional[str] = None,
                   sample_rate_hz: Optional[int] = None) -> str:
        if not self._loaded:
            self.load()
        logger.warning("QualcommAIHubProvider.transcribe not implemented.")
        return ""


# ----------------------------------------------------------------------
# ProviderFactory
# ----------------------------------------------------------------------
_BACKEND_REGISTRY: Dict[str, type] = {
    "mock": MockProvider,
    "onnx": ONNXProvider,
    "pytorch": PyTorchProvider,
    "qaihub": QualcommAIHubProvider,
}


class ProviderFactory:
    """
    Select and instantiate a ModelProvider based on config + runtime checks.

    Selection priority:
      1. ``PREFERRED_BACKEND`` setting if available and loads successfully.
      2. ``FALLBACK_BACKEND`` setting if available and loads successfully.
      3. ``MockProvider`` as the last-resort guarantee - it always loads.

    If the preferred backend is explicitly requested but cannot even be
    imported, a WARNING is logged before falling back (never fabricate
    success). The *actual* backend used is always reported by
    ``provider.backend``.
    """

    def __init__(self) -> None:
        self._logger = setup_logger(self.__class__.__name__)

    def _is_backend_available(self, name: str) -> bool:
        cls = _BACKEND_REGISTRY.get(name.lower())
        if cls is None:
            return False
        if cls is MockProvider:
            return True
        avail = getattr(cls, "is_available", None)
        if callable(avail):
            return avail()
        return True

    def _try_build(self, name: str) -> Optional[ModelProvider]:
        name = name.lower()
        cls = _BACKEND_REGISTRY.get(name)
        if cls is None:
            self._logger.warning("Unknown backend name '%s'; skipping.", name)
            return None
        try:
            provider = cls()
        except Exception as exc:
            self._logger.warning("Could not instantiate %s: %s", cls.__name__, exc)
            return None
        try:
            ok = provider.load()
        except RuntimeError as exc:
            self._logger.warning(
                "Backend '%s' refused to load with RuntimeError: %s. Trying fallback.",
                name,
                exc,
            )
            return None
        except Exception as exc:
            self._logger.warning("Backend '%s' load() threw: %s", name, exc)
            return None
        if not ok:
            self._logger.warning("Backend '%s' reported failed load().", name)
            return None
        self._logger.info(
            "Selected backend '%s' (name=%s device=%s loaded=%s).",
            name,
            provider.name,
            provider.device,
            provider.loaded,
        )
        return provider

    def create(self) -> ModelProvider:
        """
        Build a ModelProvider, honouring preferred/fallback order.

        Always returns a usable provider (at least MockProvider).
        """
        settings = get_settings()
        preferred = settings.PREFERRED_BACKEND
        fallback = settings.FALLBACK_BACKEND

        order: List[str] = []
        if preferred:
            order.append(preferred)
        if fallback and fallback != preferred:
            order.append(fallback)
        if "mock" not in order:
            order.append("mock")

        for name in order:
            if not self._is_backend_available(name):
                if name != "mock":
                    self._logger.warning(
                        "Backend '%s' is NOT available on this device - its Python package "
                        "is missing. Trying next candidate. ACTUAL backend will be reported "
                        "by the selected provider.",
                        name,
                    )
                continue
            provider = self._try_build(name)
            if provider is not None:
                return provider

        self._logger.error("All backends failed; forcing MockProvider.")
        mock = MockProvider()
        mock.load()
        return mock

    @staticmethod
    def available_backend_names() -> List[str]:
        return list(_BACKEND_REGISTRY.keys())


__all__ = [
    "Detection",
    "ModelProvider",
    "MockProvider",
    "ONNXProvider",
    "PyTorchProvider",
    "QualcommAIHubProvider",
    "ProviderFactory",
]
