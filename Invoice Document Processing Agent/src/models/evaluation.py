"""Pydantic models for evaluation results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.patient_vitals import ExtractedInvoice, InvoiceResponse


class EvalCase(BaseModel):
    """A single test case from the evaluation dataset."""

    case_id: str = ""
    pdf_path: str = ""
    expected: ExtractedInvoice = Field(default_factory=ExtractedInvoice)


class DeterministicScore(BaseModel):
    """Field-by-field deterministic comparison score."""

    total_fields: int = 0
    matched_fields: int = 0
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    details: list[dict[str, object]] = Field(default_factory=list)


class LLMJudgeScore(BaseModel):
    """Holistic score assigned by an LLM judge."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class EvalResult(BaseModel):
    """Evaluation result for a single test case."""

    case_id: str = ""
    pipeline_response: InvoiceResponse = Field(default_factory=InvoiceResponse)
    deterministic: DeterministicScore = Field(default_factory=DeterministicScore)
    llm_judge: LLMJudgeScore = Field(default_factory=LLMJudgeScore)


class EvalSummary(BaseModel):
    """Aggregated evaluation results."""

    total_cases: int = 0
    average_deterministic_score: float = 0.0
    average_llm_judge_score: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)
