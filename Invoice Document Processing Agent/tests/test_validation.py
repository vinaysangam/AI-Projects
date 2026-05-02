"""Tests for LLM validation (Step 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.alert_models import PipelineState
from src.models.patient_vitals import ExtractedInvoice
from src.pipeline.validator import validate_extraction


_MOCK_VALIDATION_RESPONSE = {
    "field_results": [
        {"field": "invoice_number", "expected": "INV-001", "extracted": "INV-001", "match": True, "note": None},
        {"field": "total_amount", "expected": 500.0, "extracted": 500.0, "match": True, "note": None},
        {"field": "vendor_name", "expected": "Acme", "extracted": "ACME Corp", "match": False, "note": "Partial match"},
    ],
    "line_item_results": [
        {"index": 0, "field": "amount", "expected": 500.0, "extracted": 500.0, "match": True, "note": None},
    ],
    "summary": "2 of 3 top-level fields matched; 1 line-item field matched.",
    "all_match": False,
}


class TestValidator:
    def test_populates_validation_report(self) -> None:
        state = PipelineState(
            raw_text="text",
            extracted=ExtractedInvoice(invoice_number="INV-001", total_amount=500.0, vendor_name="ACME Corp"),
            expected=ExtractedInvoice(invoice_number="INV-001", total_amount=500.0, vendor_name="Acme"),
        )
        llm = MagicMock()
        llm.call_json.return_value = _MOCK_VALIDATION_RESPONSE
        llm.last_token_usage = {"prompt_tokens": 200, "completion_tokens": 100}

        result = validate_extraction(state, llm)

        assert len(result.validation.field_results) == 3
        assert result.validation.field_results[0].match is True
        assert result.validation.field_results[2].match is False
        assert result.validation.all_match is False
        assert "2 of 3" in result.validation.summary

    def test_accumulates_token_usage(self) -> None:
        state = PipelineState(
            raw_text="text",
            extracted=ExtractedInvoice(),
            expected=ExtractedInvoice(),
        )
        state.token_usage = {"prompt_tokens": 50, "completion_tokens": 25}

        llm = MagicMock()
        llm.call_json.return_value = {"field_results": [], "line_item_results": [], "summary": "", "all_match": True}
        llm.last_token_usage = {"prompt_tokens": 100, "completion_tokens": 50}

        result = validate_extraction(state, llm)
        assert result.token_usage["prompt_tokens"] == 150
        assert result.token_usage["completion_tokens"] == 75

    def test_all_match_true(self) -> None:
        state = PipelineState(
            raw_text="text",
            extracted=ExtractedInvoice(invoice_number="X"),
            expected=ExtractedInvoice(invoice_number="X"),
        )
        llm = MagicMock()
        llm.call_json.return_value = {
            "field_results": [{"field": "invoice_number", "expected": "X", "extracted": "X", "match": True}],
            "line_item_results": [],
            "summary": "All matched",
            "all_match": True,
        }
        llm.last_token_usage = {"prompt_tokens": 50, "completion_tokens": 30}

        result = validate_extraction(state, llm)
        assert result.validation.all_match is True

    def test_line_item_results_populated(self) -> None:
        state = PipelineState(raw_text="text", extracted=ExtractedInvoice(), expected=ExtractedInvoice())
        llm = MagicMock()
        llm.call_json.return_value = {
            "field_results": [],
            "line_item_results": [
                {"index": 0, "field": "description", "expected": "Gadget", "extracted": "Gadget", "match": True},
                {"index": 0, "field": "amount", "expected": 99.0, "extracted": 100.0, "match": False, "note": "Off by 1"},
            ],
            "summary": "Line items partially matched",
            "all_match": False,
        }
        llm.last_token_usage = {"prompt_tokens": 80, "completion_tokens": 40}

        result = validate_extraction(state, llm)
        assert len(result.validation.line_item_results) == 2
        assert result.validation.line_item_results[1].match is False
