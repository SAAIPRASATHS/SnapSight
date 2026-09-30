from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import get_settings


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized_handlers: dict[str, bool] = {}


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger for *name* with a console handler.

    The log level is resolved from Settings.LOG_LEVEL unless explicitly
    provided. Calling multiple times with the same name is safe: the
    handler is attached only once and subsequent calls return the same
    (possibly reconfigured) logger instance.

    Args:
        name: Logger name (usually ``__name__`` of the caller module).
        level: Optional explicit level string. Overrides settings.LOG_LEVEL.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    settings = get_settings()
    resolved_level = (level or settings.LOG_LEVEL).upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    if _initialized_handlers.get(name):
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    _initialized_handlers[name] = True

    return logger


__all__ = ["setup_logger"]
