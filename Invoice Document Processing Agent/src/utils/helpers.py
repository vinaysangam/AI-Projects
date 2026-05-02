"""General-purpose helper functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_json(path: str | Path) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Source file path.

    Returns:
        Parsed JSON content.
    """
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def safe_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON; returning raw text wrapper")
        return {"raw_response": text}
