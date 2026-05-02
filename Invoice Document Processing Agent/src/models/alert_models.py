"""Pipeline state model — shared context flowing through the 3-step pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.patient_vitals import ExtractedInvoice, ValidationReport


class PipelineState(BaseModel):
    """Mutable state object passed through every pipeline step.

    Step 1 (PDF extraction) writes ``raw_text``.
    Step 2 (field extraction) writes ``extracted``.
    Step 3 (validation) writes ``validation``.
    """

    pdf_path: str = ""
    raw_text: str = ""
    extracted: ExtractedInvoice = Field(default_factory=ExtractedInvoice)
    expected: ExtractedInvoice = Field(default_factory=ExtractedInvoice)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    token_usage: dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
