from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..inference.manager import InferenceManager
from ..models.schemas import DetectionBox
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


_BOX_COLORS: Sequence[Tuple[int, int, int]] = (
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 128),
    (255, 165, 0),
    (128, 128, 0),
    (0, 128, 128),
)


class ObjectDetector:
    """
    High level helper that wraps the InferenceManager for object detection
    and optional bounding-box rendering on a frame.

    This class intentionally owns no state beyond a reference to the
    singleton InferenceManager; it exists to provide a clean API boundary
    for the routes / websocket layers.
    """

    def __init__(self, manager: Optional[InferenceManager] = None) -> None:
        self._manager: InferenceManager = manager or InferenceManager.instance()
        self._cv2 = self._import_cv2()

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # type: ignore

            return cv2
        except ImportError:
            logger.warning(
                "ObjectDetector: cv2 not installed. draw_bounding_boxes will return "
                "original frame bytes without visualisation."
            )
            return None

    # ------------------------------------------------------------------
    # Detection API
    # ------------------------------------------------------------------
    def detect_objects(self, frame: np.ndarray) -> List[DetectionBox]:
        """
        Run synchronous detection on *frame* using the inference manager.

        This method is thread-safe with respect to other callers because
        InferenceManager serialises all provider calls internally.
        """
        if frame is None or frame.size == 0:
            return []
        return self._manager.run_detect(frame)

    async def detect_objects_async(self, frame: np.ndarray) -> List[DetectionBox]:
        if frame is None or frame.size == 0:
            return []
        return await self._manager.run_detect_async(frame)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw_bounding_boxes(
        self,
        frame: np.ndarray,
        detections: Sequence[DetectionBox],
        *,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        draw_label: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Draw detection boxes onto a copy of *frame*.

        Returns a new numpy array (BGR uint8) with boxes drawn, or the
        original frame unchanged if cv2 is not available.
        """
        if frame is None:
            return None
        if self._cv2 is None or not detections:
            return frame
        try:
            canvas = frame.copy()
            h, w = canvas.shape[:2]
            for i, det in enumerate(detections):
                color = _BOX_COLORS[i % len(_BOX_COLORS)]
                bbox = det.bbox
                if bbox.normalized:
                    x1, y1, x2, y2 = (
                        int(bbox.xmin * w),
                        int(bbox.ymin * h),
                        int(bbox.xmax * w),
                        int(bbox.ymax * h),
                    )
                else:
                    x1, y1, x2, y2 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                self._cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
                if draw_label:
                    label = f"{det.label} {det.confidence:.0%}"
                    (tw, th), _ = self._cv2.getTextSize(
                        label, self._cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                    )
                    lx1, ly1 = x1, max(th + 4, y1 - 8)
                    lx2, ly2 = x1 + tw + 8, y1
                    self._cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), color, -1)
                    self._cv2.putText(
                        canvas,
                        label,
                        (x1 + 4, y1 - 4),
                        self._cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        1,
                        self._cv2.LINE_AA,
                    )
            return canvas
        except Exception as exc:
            logger.warning("draw_bounding_boxes failed: %s", exc)
            return frame

    def render_annotated_jpeg(
        self,
        frame: np.ndarray,
        detections: Sequence[DetectionBox],
        *,
        quality: int = 90,
    ) -> bytes:
        """
        Draw boxes on *frame* then encode to JPEG bytes.

        Returns an empty byte string if cv2 is unavailable or encoding fails.
        """
        if self._cv2 is None:
            return b""
        annotated = self.draw_bounding_boxes(frame, detections)
        if annotated is None:
            return b""
        try:
            q = max(1, min(100, int(quality)))
            ok, buf = self._cv2.imencode(
                ".jpg", annotated, [int(self._cv2.IMWRITE_JPEG_QUALITY), q]
            )
            if not ok or buf is None:
                return b""
            return buf.tobytes()
        except Exception as exc:
            logger.warning("render_annotated_jpeg encode failed: %s", exc)
            return b""


__all__ = ["ObjectDetector"]
