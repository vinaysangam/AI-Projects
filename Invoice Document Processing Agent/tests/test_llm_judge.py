"""Tests for the LLM-as-Judge evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation.llm_judge import score_with_llm_judge
from src.models.patient_vitals import ExtractedInvoice, ValidationReport


class TestLLMJudge:
    def test_returns_score_and_reasoning(self) -> None:
        llm = MagicMock()
        llm.call_json.return_value = {"score": 0.85, "reasoning": "Good extraction"}

        result = score_with_llm_judge(
            expected=ExtractedInvoice(invoice_number="X"),
            extracted=ExtractedInvoice(invoice_number="X"),
            validation=ValidationReport(summary="ok", all_match=True),
            llm=llm,
        )
        assert result.score == 0.85
        assert result.reasoning == "Good extraction"

    def test_clamps_score_to_0_1(self) -> None:
        llm = MagicMock()
        llm.call_json.return_value = {"score": 1.5, "reasoning": "Overshot"}

        result = score_with_llm_judge(
            expected=ExtractedInvoice(),
            extracted=ExtractedInvoice(),
            validation=ValidationReport(),
            llm=llm,
        )
        assert result.score == 1.0

    def test_negative_score_clamped_to_zero(self) -> None:
        llm = MagicMock()
        llm.call_json.return_value = {"score": -0.5, "reasoning": "Negative"}

        result = score_with_llm_judge(
            expected=ExtractedInvoice(),
            extracted=ExtractedInvoice(),
            validation=ValidationReport(),
            llm=llm,
        )
        assert result.score == 0.0

    def test_non_numeric_score_defaults_to_zero(self) -> None:
        llm = MagicMock()
        llm.call_json.return_value = {"score": "invalid", "reasoning": "Bad data"}

        result = score_with_llm_judge(
            expected=ExtractedInvoice(),
            extracted=ExtractedInvoice(),
            validation=ValidationReport(),
            llm=llm,
        )
        assert result.score == 0.0

    def test_missing_keys_handled(self) -> None:
        llm = MagicMock()
        llm.call_json.return_value = {}

        result = score_with_llm_judge(
            expected=ExtractedInvoice(),
            extracted=ExtractedInvoice(),
            validation=ValidationReport(),
            llm=llm,
        )
        assert result.score == 0.0
        assert result.reasoning == ""
