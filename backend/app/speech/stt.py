from __future__ import annotations

from typing import Optional, Tuple

from ..inference.manager import InferenceManager
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


class SpeechToText:
    """
    Speech-to-text wrapper over the inference provider's transcribe method.

    Accepts audio bytes in any reasonable format. The mock provider ignores
    content and returns a canned question; real backends should implement
    waveform parsing / resampling as needed.
    """

    def __init__(self, manager: Optional[InferenceManager] = None) -> None:
        self._manager: InferenceManager = manager or InferenceManager.instance()

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: bytes,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Synchronous transcription.

        Args:
            audio: Raw audio bytes.
            language: Optional ISO-639-1 language hint, e.g. 'en'.
            sample_rate_hz: Optional sample rate hint in Hz.

        Returns:
            Tuple of ``(text, language_used_or_None)``.
        """
        if not audio:
            return "", language
        text = self._manager.run_transcribe(audio, language, sample_rate_hz)
        return text or "", language

    async def transcribe_async(
        self,
        audio: bytes,
        language: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
    ) -> Tuple[str, Optional[str]]:
        if not audio:
            return "", language
        text = await self._manager.run_transcribe_async(audio, language, sample_rate_hz)
        return text or "", language

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        info = self._manager.provider_info()
        return info.get("backend", "unknown")


class TextToSpeech:
    """
    Minimal Text-to-Speech stub.

    Returns empty bytes on all calls. The stub exists so future
    integrations (Piper, Coqui, ElevenLabs local, etc.) have a stable API
    to target without changing call sites.
    """

    def __init__(self) -> None:
        self._logger = setup_logger(self.__class__.__name__)

    def synthesize(self, text: str, language: str = "en") -> bytes:
        if not text:
            return b""
        self._logger.info("TextToSpeech.synthesize() called (stub). text=%r", text[:80])
        return b""

    async def synthesize_async(self, text: str, language: str = "en") -> bytes:
        return self.synthesize(text, language)


__all__ = ["SpeechToText", "TextToSpeech"]
