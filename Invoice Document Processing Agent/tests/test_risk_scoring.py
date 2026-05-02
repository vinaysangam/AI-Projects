"""Tests for LLM field extraction (Step 2)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.models.alert_models import PipelineState
from src.pipeline.field_extractor import extract_fields, _safe_float, _count_non_null
from src.models.patient_vitals import ExtractedInvoice


_MOCK_LLM_RESPONSE = {
    "invoice_number": "INV-2026-001",
    "invoice_date": "2026-01-15",
    "due_date": "2026-02-15",
    "vendor_name": "Acme Corp",
    "customer_name": "Widget Inc",
    "base_amount": 1000.00,
    "tax_amount": 80.00,
    "total_amount": 1080.00,
    "currency": "USD",
    "line_items": [
        {"description": "Widget A", "quantity": 10, "unit_price": 50.0, "amount": 500.0},
        {"description": "Widget B", "quantity": 5, "unit_price": 100.0, "amount": 500.0},
    ],
}


class TestFieldExtractor:
    def test_extracts_all_fields(self) -> None:
        state = PipelineState(raw_text="Invoice #INV-2026-001\nTotal: $1080.00")
        llm = MagicMock()
        llm.call_json.return_value = _MOCK_LLM_RESPONSE
        llm.last_token_usage = {"prompt_tokens": 100, "completion_tokens": 50}

        result = extract_fields(state, llm)

        assert result.extracted.invoice_number == "INV-2026-001"
        assert result.extracted.total_amount == 1080.00
        assert len(result.extracted.line_items) == 2
        assert result.extracted.line_items[0].description == "Widget A"

    def test_accumulates_token_usage(self) -> None:
        state = PipelineState(raw_text="Some text")
        state.token_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        llm = MagicMock()
        llm.call_json.return_value = _MOCK_LLM_RESPONSE
        llm.last_token_usage = {"prompt_tokens": 100, "completion_tokens": 50}

        result = extract_fields(state, llm)
        assert result.token_usage["prompt_tokens"] == 110
        assert result.token_usage["completion_tokens"] == 55

    def test_empty_raw_text_raises(self) -> None:
        state = PipelineState(raw_text="")
        llm = MagicMock()
        with pytest.raises(ValueError, match="raw_text is empty"):
            extract_fields(state, llm)

    def test_handles_null_fields_gracefully(self) -> None:
        state = PipelineState(raw_text="Sparse invoice text")
        llm = MagicMock()
        llm.call_json.return_value = {"invoice_number": "X-1", "line_items": []}
        llm.last_token_usage = {"prompt_tokens": 50, "completion_tokens": 20}

        result = extract_fields(state, llm)
        assert result.extracted.invoice_number == "X-1"
        assert result.extracted.total_amount is None
        assert result.extracted.line_items == []

    def test_safe_float_valid(self) -> None:
        assert _safe_float(42.5) == 42.5
        assert _safe_float("100") == 100.0
        assert _safe_float(0) == 0.0

    def test_safe_float_invalid(self) -> None:
        assert _safe_float(None) is None
        assert _safe_float("abc") is None

    def test_count_non_null(self) -> None:
        inv = ExtractedInvoice(invoice_number="X", vendor_name="V")
        assert _count_non_null(inv) == 2

        inv_full = ExtractedInvoice(**{k: v for k, v in _MOCK_LLM_RESPONSE.items() if k != "line_items"})
        assert _count_non_null(inv_full) == 9
