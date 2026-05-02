"""Tests for the pipeline orchestrator (end-to-end with mocks)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.models.patient_vitals import ExtractedInvoice
from src.pipeline.orchestrator import InvoicePipeline
from src.utils.llm_client import LLMClient


_EXTRACTION_RESPONSE = {
    "invoice_number": "INV-001",
    "invoice_date": "2026-01-15",
    "vendor_name": "TestVendor",
    "base_amount": 100.0,
    "tax_amount": 10.0,
    "total_amount": 110.0,
    "currency": "USD",
    "line_items": [{"description": "Item A", "quantity": 2, "unit_price": 50.0, "amount": 100.0}],
}

_VALIDATION_RESPONSE = {
    "field_results": [
        {"field": "invoice_number", "expected": "INV-001", "extracted": "INV-001", "match": True},
    ],
    "line_item_results": [],
    "summary": "All good",
    "all_match": True,
}


@pytest.fixture
def mock_settings():
    s = MagicMock(spec=Settings)
    s.extraction_temperature = 0.0
    s.validation_temperature = 0.0
    s.max_completion_tokens = 2000
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.call_json.side_effect = [_EXTRACTION_RESPONSE, _VALIDATION_RESPONSE]
    llm.last_token_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return llm


class TestInvoicePipeline:
    def test_full_pipeline_returns_response(self, mock_settings, mock_llm, tmp_path) -> None:
        """Pipeline produces an InvoiceResponse with extracted data and validation."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        expected = ExtractedInvoice(invoice_number="INV-001", total_amount=110.0)

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Invoice #INV-001\nTotal: $110.00"
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            pipeline = InvoicePipeline(mock_settings, mock_llm)
            result = pipeline.process(str(pdf_file), expected)

        assert result.extracted.invoice_number == "INV-001"
        assert result.validation.all_match is True
        assert "Invoice #INV-001" in result.raw_text

    def test_pipeline_raises_on_missing_pdf(self, mock_settings, mock_llm) -> None:
        expected = ExtractedInvoice()
        pipeline = InvoicePipeline(mock_settings, mock_llm)
        with pytest.raises(FileNotFoundError):
            pipeline.process("/nonexistent/file.pdf", expected)
