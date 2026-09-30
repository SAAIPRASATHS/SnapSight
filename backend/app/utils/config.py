from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"


def load_hardware_config() -> Dict[str, Any]:
    """
    Load hardware configuration from config/hardware.yaml.

    Returns:
        Dictionary containing hardware configuration values.
        Returns empty dict if file is not found or cannot be parsed.
    """
    config_path = CONFIG_DIR / "hardware.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("hardware", {})
    except (yaml.YAMLError, OSError) as exc:
        print(f"Warning: Could not load hardware config: {exc}")
        return {}


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with pydantic-settings.

    Environment variables take precedence over hardware.yaml defaults.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    BACKEND_HOST: str = Field(default="0.0.0.0", description="Host to bind the FastAPI server")
    BACKEND_PORT: int = Field(default=8000, ge=1, le=65535, description="Port to bind the FastAPI server")
    PREFERRED_BACKEND: str = Field(default="mock", description="Preferred inference backend: mock, onnx, pytorch, qaihub")
    FALLBACK_BACKEND: str = Field(default="mock", description="Fallback inference backend if preferred is unavailable")
    CAMERA_INDEX: int = Field(default=0, ge=0, description="OpenCV VideoCapture camera index")
    CAMERA_WIDTH: int = Field(default=1280, ge=160, description="Camera capture width in pixels")
    CAMERA_HEIGHT: int = Field(default=720, ge=120, description="Camera capture height in pixels")
    INFERENCE_FPS: int = Field(default=10, ge=1, le=120, description="Target inference frames per second")
    CONFIDENCE_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for detections")
    ENABLE_CLOUD_APIS: bool = Field(default=False, description="Whether to allow external cloud API calls")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    def apply_hardware_defaults(self, hw_cfg: Optional[Dict[str, Any]] = None) -> "Settings":
        """
        Apply defaults from hardware.yaml for any setting that still has its
        factory default and has a matching key in the hardware config.

        Args:
            hw_cfg: Pre-loaded hardware config dict. If None, loads from file.

        Returns:
            Self with potentially updated fields.
        """
        if hw_cfg is None:
            hw_cfg = load_hardware_config()
        if not hw_cfg:
            return self

        inf = hw_cfg.get("inference", {}) or {}
        cam = hw_cfg.get("camera", {}) or {}

        if "preferred_backend" in inf:
            self.PREFERRED_BACKEND = inf["preferred_backend"]
        if "fallback_backend" in inf:
            self.FALLBACK_BACKEND = inf["fallback_backend"]
        if "fps" in inf:
            self.INFERENCE_FPS = int(inf["fps"])
        if "confidence_threshold" in inf:
            self.CONFIDENCE_THRESHOLD = float(inf["confidence_threshold"])
        if "enable_cloud_apis" in inf:
            self.ENABLE_CLOUD_APIS = bool(inf["enable_cloud_apis"])
        if "index" in cam:
            self.CAMERA_INDEX = int(cam["index"])
        if "width" in cam:
            self.CAMERA_WIDTH = int(cam["width"])
        if "height" in cam:
            self.CAMERA_HEIGHT = int(cam["height"])
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a singleton Settings instance.

    Loads hardware.yaml defaults and caches the result for the process lifetime.
    """
    hw_cfg = load_hardware_config()
    settings = Settings()
    settings.apply_hardware_defaults(hw_cfg)
    return settings


__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "Settings",
    "get_settings",
    "load_hardware_config",
]
