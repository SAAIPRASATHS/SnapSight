from __future__ import annotations

import platform
import sys
from typing import List

from ..models.schemas import HardwareInfo
from ..utils.logging import setup_logger
from ..utils.metrics import has_npu_runtime


logger = setup_logger(__name__)


class HardwareDetector:
    """
    Probe the runtime environment for CPU / RAM / GPU / NPU capabilities.

    All detection is soft: missing packages or unsupported hardware simply
    results in empty strings / zero values rather than raising exceptions.
    No network calls are made.
    """

    def __init__(self) -> None:
        self._cached: HardwareInfo | None = None
        self._notes: List[str] = []

    # ------------------------------------------------------------------
    # CPU / RAM
    # ------------------------------------------------------------------
    def _detect_cpu(self) -> str:
        """Return a human-readable CPU description string."""
        try:
            raw = platform.processor() or platform.machine() or ""
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read CPU info: %s", exc)
            raw = ""
        if not raw:
            raw = f"{platform.system()} {platform.machine()}"
        return raw.strip() or "Unknown CPU"

    def _detect_ram_gb(self) -> float:
        """Return total RAM in gigabytes, or 0.0 if undetectable."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            return round(mem.total / (1024.0 ** 3), 2)
        except ImportError:
            self._notes.append("psutil not installed; RAM detection skipped.")
            return 0.0
        except Exception as exc:  # pragma: no cover
            logger.warning("RAM detection failed: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------
    def _detect_gpu(self) -> str:
        """
        Return GPU info string by probing CUDA / torch / onnxruntime in order.

        Never raises; returns empty string if nothing is available.
        """
        parts: List[str] = []
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                dev_count = torch.cuda.device_count()
                names = []
                for i in range(min(dev_count, 4)):
                    try:
                        names.append(torch.cuda.get_device_name(i))
                    except Exception:
                        names.append("Unknown CUDA GPU")
                if names:
                    parts.append(f"CUDA x{dev_count}: {', '.join(names)}")
        except ImportError:
            self._notes.append("torch not installed; GPU detection via CUDA skipped.")
        except Exception as exc:
            logger.debug("torch CUDA probe failed: %s", exc)

        if not parts:
            try:
                import onnxruntime as ort  # type: ignore

                providers = ort.get_available_providers()
                cuda_providers = [p for p in providers if "CUDA" in p.upper()]
                if cuda_providers:
                    parts.append(f"ONNX providers: {', '.join(cuda_providers)}")
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("onnxruntime GPU probe failed: %s", exc)

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # NPU
    # ------------------------------------------------------------------
    def _detect_npu(self) -> str:
        """
        Detect NPU / Qualcomm AI Hub runtime availability.

        Returns a descriptive string or empty string if no NPU runtime found.
        """
        info = has_npu_runtime()
        if not info["available"]:
            if info.get("notes"):
                self._notes.append(info["notes"])
            return ""
        return "NPU: " + ", ".join(info["backends"])

    # ------------------------------------------------------------------
    # Backends availability
    # ------------------------------------------------------------------
    def _available_backends(self) -> List[str]:
        """Return list of backend identifiers actually available."""
        backends: List[str] = ["mock"]

        try:
            import onnxruntime  # type: ignore  # noqa: F401

            backends.append("onnx")
        except ImportError:
            pass

        try:
            import torch  # type: ignore  # noqa: F401
            import torchvision  # type: ignore  # noqa: F401

            backends.append("pytorch")
        except ImportError:
            pass

        try:
            import qai_hub  # type: ignore  # noqa: F401

            backends.append("qaihub")
        except ImportError:
            pass

        return backends

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, force_refresh: bool = False) -> HardwareInfo:
        """
        Perform hardware detection and return a HardwareInfo record.

        Results are cached by default. Pass ``force_refresh=True`` to re-run
        detection after a capability change (e.g. USB device hotplug).
        """
        if self._cached is not None and not force_refresh:
            return self._cached

        self._notes = []
        cpu = self._detect_cpu()
        ram_gb = self._detect_ram_gb()
        gpu = self._detect_gpu()
        npu = self._detect_npu()
        backends = self._available_backends()

        info = HardwareInfo(
            vendor="AUTO_DETECT",
            platform=f"{platform.system()} {platform.release()} ({sys.platform})",
            model="AUTO_DETECT",
            cpu=cpu,
            gpu=gpu,
            npu=npu,
            ram_gb=ram_gb,
            available_backends=backends,
            notes="; ".join(self._notes) if self._notes else "",
        )
        logger.info(
            "Hardware detected: cpu=%s ram=%.1fGB gpu=%s npu=%s backends=%s",
            cpu[:60],
            ram_gb,
            bool(gpu),
            bool(npu),
            backends,
        )
        self._cached = info
        return info


__all__ = ["HardwareDetector"]
