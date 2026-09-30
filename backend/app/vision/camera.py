from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.config import get_settings
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


@dataclass
class CameraStatus:
    """Snapshottable status returned by CameraManager.get_status()."""

    index: int
    running: bool
    opened: bool
    width: int
    height: int
    fps_target: int
    frames_read: int
    last_read_ts: Optional[float]
    last_error: Optional[str]
    backend: str


class CameraManager:
    """
    OpenCV VideoCapture wrapper with resolution configuration, frame-rate
    sampling, graceful fallback behaviour, and thread-safe access.

    If no camera is available (no device, missing driver, wrong index),
    the manager degrades gracefully: :meth:`start` returns False,
    :meth:`read_frame` returns None and :meth:`capture_snapshot` returns
    an empty byte string, all with clear log messages.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._lock = threading.RLock()
        self._cap: Optional[Any] = None
        self._index: int = self._settings.CAMERA_INDEX
        self._width: int = self._settings.CAMERA_WIDTH
        self._height: int = self._settings.CAMERA_HEIGHT
        self._fps_target: int = max(1, self._settings.INFERENCE_FPS)
        self._min_interval: float = 1.0 / self._fps_target
        self._last_read_ts: float = 0.0
        self._frames_read: int = 0
        self._last_error: Optional[str] = None
        self._opened: bool = False
        self._running: bool = False
        self._cv2 = self._import_cv2()

    # ------------------------------------------------------------------
    # cv2 import helper
    # ------------------------------------------------------------------
    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore

            return cv2
        except ImportError:
            logger.warning(
                "OpenCV (cv2) not installed. CameraManager will run in degraded "
                "mode (no actual frames). All camera APIs will return empty/safe defaults."
            )
            return None

    @property
    def backend_name(self) -> str:
        return "opencv" if self._cv2 is not None else "none"

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def set_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """
        Try to update the resolution. Returns the actually applied values.

        If the capture is already open, applies the change immediately
        (best effort - OpenCV silently ignores unsupported resolutions).
        """
        with self._lock:
            width = max(160, int(width))
            height = max(120, int(height))
            self._width = width
            self._height = height
            if self._cap is not None and self._cv2 is not None:
                try:
                    self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, width)
                    self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_w = int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)) or width
                    actual_h = int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)) or height
                    self._width = actual_w
                    self._height = actual_h
                except Exception as exc:
                    logger.warning("set_resolution apply failed: %s", exc)
            return self._width, self._height

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, index: Optional[int] = None) -> bool:
        """
        Open the camera device. Idempotent.

        Args:
            index: Optional camera index override. Defaults to settings.CAMERA_INDEX.

        Returns:
            True if the camera is usable after this call, False otherwise.
        """
        with self._lock:
            if self._running and self._opened:
                return True
            if self._cv2 is None:
                self._last_error = "OpenCV (cv2) not installed; camera disabled."
                logger.warning(self._last_error)
                self._running = False
                self._opened = False
                return False
            if index is not None:
                self._index = int(index)
            try:
                cap = self._cv2.VideoCapture(self._index)
                if not cap.isOpened():
                    self._last_error = (
                        f"VideoCapture({self._index}) could not be opened. "
                        "The device may be in use, missing, or the index wrong."
                    )
                    logger.warning(self._last_error)
                    self._opened = False
                    self._running = True
                    return False
                cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
                cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._width = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)) or self._width
                self._height = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)) or self._height
                self._cap = cap
                self._opened = True
                self._running = True
                self._last_error = None
                logger.info(
                    "Camera started index=%d resolution=%dx%d backend=%s",
                    self._index,
                    self._width,
                    self._height,
                    self.backend_name,
                )
                return True
            except Exception as exc:
                self._last_error = f"Camera.start() raised: {exc}"
                logger.exception(self._last_error)
                self._opened = False
                self._running = True
                return False

    def stop(self) -> None:
        """Release the VideoCapture (if any) and reset counters."""
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    logger.debug("Camera.release() warning: %s", exc)
                self._cap = None
            self._opened = False
            self._running = False
            self._frames_read = 0
            self._last_read_ts = 0.0
            logger.info("Camera stopped.")

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------
    def read_frame(self, force: bool = False) -> Optional[np.ndarray]:
        """
        Return the latest frame as a BGR uint8 numpy array, or None.

        Args:
            force: If False, throttles reads to the configured FPS target
                by returning the last-known-good read timestamp check.
        """
        with self._lock:
            if not self._running:
                self.start()
            now = time.perf_counter()
            if not force and not self._should_sample(now):
                return None
            if not self._opened or self._cap is None or self._cv2 is None:
                return None
            try:
                ok, frame = self._cap.read()
            except Exception as exc:
                self._last_error = f"VideoCapture.read() raised: {exc}"
                logger.debug(self._last_error)
                return None
            if not ok or frame is None:
                self._last_error = "VideoCapture.read() returned ok=False."
                return None
            self._last_read_ts = now
            self._frames_read += 1
            return frame

    def _should_sample(self, now: float) -> bool:
        return (now - self._last_read_ts) >= (self._min_interval - 1e-6)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def capture_snapshot(
        self,
        quality: int = 90,
        width: Optional[int] = None,
        height: Optional[int] = None,
        annotate_frame: Optional[np.ndarray] = None,
    ) -> bytes:
        """
        Capture a JPEG snapshot as bytes.

        Args:
            quality: JPEG encoding quality, 1..100.
            width: Optional resize width.
            height: Optional resize height.
            annotate_frame: Optional already-rendered BGR frame to encode
                instead of reading a new frame. Useful for drawing boxes.

        Returns:
            JPEG bytes, empty bytes on failure.
        """
        quality = max(1, min(100, int(quality)))
        frame = annotate_frame if annotate_frame is not None else self.read_frame(force=True)
        if frame is None or self._cv2 is None:
            return b""
        try:
            if (width and width > 0) or (height and height > 0):
                target_w = width or frame.shape[1]
                target_h = height or frame.shape[0]
                frame = self._cv2.resize(frame, (int(target_w), int(target_h)))
            encode_params = [int(self._cv2.IMWRITE_JPEG_QUALITY), quality]
            ok, buf = self._cv2.imencode(".jpg", frame, encode_params)
            if not ok or buf is None:
                return b""
            return buf.tobytes()
        except Exception as exc:
            self._last_error = f"capture_snapshot encode failed: {exc}"
            logger.debug(self._last_error)
            return b""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                index=self._index,
                running=self._running,
                opened=self._opened,
                width=self._width,
                height=self._height,
                fps_target=self._fps_target,
                frames_read=self._frames_read,
                last_read_ts=self._last_read_ts if self._last_read_ts else None,
                last_error=self._last_error,
                backend=self.backend_name,
            )

    @property
    def is_available(self) -> bool:
        return self._opened and self._cap is not None and self._cv2 is not None


__all__ = ["CameraManager", "CameraStatus"]
