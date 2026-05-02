"""Step 1 — Extract raw text from a PDF file using pdfplumber."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from src.models.alert_models import PipelineState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pdf_text(state: PipelineState) -> PipelineState:
    """Read the PDF at ``state.pdf_path`` and populate ``state.raw_text``.

    Args:
        state: Current pipeline state (must have ``pdf_path`` set).

    Returns:
        The same *state* with ``raw_text`` filled in.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If no text could be extracted.
    """
    path = Path(state.pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    raw = "\n\n".join(pages_text).strip()
    if not raw:
        raise ValueError(f"No text extracted from PDF: {path}")

    state.raw_text = raw
    logger.info("Extracted %d characters from %s (%d pages)", len(raw), path.name, len(pages_text))
    return state
