"""Convenience logger accessor."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance.

    Args:
        name: Dotted module name for the logger hierarchy.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
