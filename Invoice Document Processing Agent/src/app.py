"""FastAPI application for the Invoice Document Processing Agent.

Endpoints:
  POST /api/process_invoice  — process a single invoice (PDF → extraction → validation)
  POST /api/evaluate         — run evaluation on a test dataset
  GET  /health               — liveness check
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config.logging_config import configure_logging
from src.config.settings import Settings, get_settings
from src.evaluation.deterministic import score_deterministic
from src.evaluation.llm_judge import score_with_llm_judge
from src.models.evaluation import EvalCase, EvalResult, EvalSummary
from src.models.patient_vitals import ExtractedInvoice, InvoiceRequest, InvoiceResponse
from src.pipeline.orchestrator import InvoicePipeline
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singletons initialised at startup.
_settings: Settings | None = None
_llm: LLMClient | None = None
_pipeline: InvoicePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _settings, _llm, _pipeline
    _settings = get_settings()
    configure_logging(_settings.log_level)

    _llm = LLMClient(_settings)
    _pipeline = InvoicePipeline(_settings, _llm)

    logger.info("Invoice Processing Agent started")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Invoice Document Processing Agent",
    version="1.0.0",
    description=(
        "AI-powered invoice processing: PDF text extraction, "
        "LLM field extraction, LLM validation, and dual evaluation."
    ),
    lifespan=lifespan,
)


# --- Request / Response schemas -----------------------------------------------

class EvaluateRequest(BaseModel):
    """POST body for /api/evaluate — path to a JSON test-case dataset."""

    dataset_path: str = Field(..., min_length=1, description="Path to test_cases.json")


# --- Endpoints ----------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> dict[str, Any]:
    """Liveness check."""
    return {
        "status": "healthy",
        "service": "invoice-processing-agent",
        "version": "1.0.0",
    }


@app.post("/api/process_invoice", response_model=InvoiceResponse, tags=["Invoice"])
async def process_invoice(request: InvoiceRequest) -> InvoiceResponse:
    """Process a single invoice through the full pipeline.

    Accepts a JSON body with ``pdf_path`` and ``expected`` fields.
    Returns the extracted data and a validation report.
    """
    assert _pipeline is not None

    try:
        result = _pipeline.process(request.pdf_path, request.expected)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return result


@app.post("/api/evaluate", response_model=EvalSummary, tags=["Evaluation"])
async def evaluate(request: EvaluateRequest) -> EvalSummary:
    """Run evaluation on a test dataset.

    Each case is processed through the full pipeline, then scored with
    both a deterministic evaluator and an LLM judge.

    The dataset JSON must be a list of objects with keys:
    ``case_id``, ``pdf_path``, ``expected`` (matching ExtractedInvoice schema).
    """
    assert _pipeline is not None and _llm is not None and _settings is not None

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset not found: {dataset_path}")

    try:
        raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid JSON: {exc}")

    if not isinstance(raw_cases, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset must be a JSON array")

    results: list[EvalResult] = []
    total_det = 0.0
    total_judge = 0.0

    for raw in raw_cases:
        case = EvalCase(**raw)

        # Run pipeline
        try:
            response = _pipeline.process(case.pdf_path, case.expected)
        except Exception as exc:
            logger.warning("Case %s failed: %s", case.case_id, exc)
            results.append(EvalResult(case_id=case.case_id))
            continue

        # Deterministic scoring
        det_score = score_deterministic(case.expected, response.extracted)

        # LLM judge scoring
        judge_score = score_with_llm_judge(
            case.expected,
            response.extracted,
            response.validation,
            _llm,
            temperature=_settings.judge_temperature,
        )

        total_det += det_score.score
        total_judge += judge_score.score

        results.append(
            EvalResult(
                case_id=case.case_id,
                pipeline_response=response,
                deterministic=det_score,
                llm_judge=judge_score,
            )
        )

    n = len(results) or 1
    return EvalSummary(
        total_cases=len(results),
        average_deterministic_score=round(total_det / n, 4),
        average_llm_judge_score=round(total_judge / n, 4),
        results=results,
    )
