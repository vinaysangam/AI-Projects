"""Logging configuration using structured JSON logging."""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    """Set up structured JSON logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # Clear existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
