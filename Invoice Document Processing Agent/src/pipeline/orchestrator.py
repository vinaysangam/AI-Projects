"""Pipeline orchestrator — chains PDF extraction → field extraction → validation."""

from __future__ import annotations

from src.config.settings import Settings
from src.models.alert_models import PipelineState
from src.models.patient_vitals import ExtractedInvoice, InvoiceResponse
from src.pipeline.pdf_extractor import extract_pdf_text
from src.pipeline.field_extractor import extract_fields
from src.pipeline.validator import validate_extraction
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InvoicePipeline:
    """Runs the 3-step invoice processing pipeline."""

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    def process(self, pdf_path: str, expected: ExtractedInvoice) -> InvoiceResponse:
        """Run the full pipeline for a single invoice.

        Args:
            pdf_path: File-system path to the PDF.
            expected: Expected field values for validation.

        Returns:
            An :class:`InvoiceResponse` with extraction and validation results.
        """
        state = PipelineState(pdf_path=pdf_path, expected=expected)

        # Step 1 — PDF text extraction
        logger.info("Step 1: Extracting text from %s", pdf_path)
        state = extract_pdf_text(state)

        # Step 2 — LLM field extraction
        logger.info("Step 2: Extracting fields via LLM")
        state = extract_fields(
            state,
            self._llm,
            temperature=self._settings.extraction_temperature,
        )

        # Step 3 — LLM validation
        logger.info("Step 3: Validating extraction via LLM")
        state = validate_extraction(
            state,
            self._llm,
            temperature=self._settings.validation_temperature,
        )

        return InvoiceResponse(
            raw_text=state.raw_text,
            extracted=state.extracted,
            validation=state.validation,
        )
