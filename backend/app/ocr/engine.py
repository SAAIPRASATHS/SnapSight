from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..inference.manager import InferenceManager
from ..models.schemas import OCRResult
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


class OCREngine:
    """
    Thin wrapper over InferenceManager for OCR tasks.

    Provides both sync and async (thread-pool backed) entry points so the
    HTTP layer can remain fully async while heavy OCR inference runs in a
    worker thread.
    """

    def __init__(self, manager: Optional[InferenceManager] = None) -> None:
        self._manager: InferenceManager = manager or InferenceManager.instance()

    def run_ocr(self, frame: np.ndarray) -> List[OCRResult]:
        """
        Run OCR synchronously on the given BGR frame.

        Args:
            frame: HxWx3 BGR uint8 numpy array.

        Returns:
            List of :class:`OCRResult` objects, potentially empty.
        """
        if frame is None or frame.size == 0:
            return []
        results = self._manager.run_ocr(frame)
        return results or []

    async def run_ocr_async(self, frame: np.ndarray) -> List[OCRResult]:
        """Run OCR on the default thread pool executor."""
        if frame is None or frame.size == 0:
            return []
        return await self._manager.run_ocr_async(frame)

    @property
    def backend(self) -> str:
        info = self._manager.provider_info()
        return info.get("backend", "unknown")


__all__ = ["OCREngine"]
