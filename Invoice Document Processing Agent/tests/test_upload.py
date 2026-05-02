"""Tests for FastAPI endpoints (/health, /api/process_invoice, /api/evaluate)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Mock LLM responses
_EXTRACTION_LLM = json.dumps({
    "invoice_number": "INV-001",
    "invoice_date": "2026-01-15",
    "due_date": None,
    "vendor_name": "Acme Corp",
    "customer_name": None,
    "base_amount": 100.0,
    "tax_amount": 10.0,
    "total_amount": 110.0,
    "currency": "USD",
    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 50.0, "amount": 100.0}],
})

_VALIDATION_LLM = json.dumps({
    "field_results": [
        {"field": "invoice_number", "expected": "INV-001", "extracted": "INV-001", "match": True, "note": None},
        {"field": "total_amount", "expected": 110.0, "extracted": 110.0, "match": True, "note": None},
    ],
    "line_item_results": [],
    "summary": "All matched",
    "all_match": True,
})

_JUDGE_LLM = json.dumps({
    "score": 0.95,
    "reasoning": "Nearly perfect extraction",
})


@pytest.fixture
def _mock_settings():
    """Patch settings to avoid requiring real Azure credentials."""
    with patch("src.app.get_settings") as mock:
        settings = MagicMock()
        settings.log_level = "WARNING"
        settings.azure_openai_endpoint = "https://mock.openai.azure.com/"
        settings.azure_openai_deployment = "gpt-4o"
        settings.azure_openai_api_version = "2024-12-01-preview"
        settings.extraction_temperature = 0.0
        settings.validation_temperature = 0.0
        settings.judge_temperature = 0.3
        settings.max_prompt_tokens = 4000
        settings.max_completion_tokens = 2000
        mock.return_value = settings
        yield settings


# Track which call we're on to return the right mock response
_llm_call_count = 0


def _mock_llm_call(prompt: str, *, temperature: float | None = None) -> str:
    """Return the right mock response based on prompt content."""
    if "Extract structured data" in prompt or "extract" in prompt.lower()[:200]:
        return _EXTRACTION_LLM
    elif "Compare extracted" in prompt or "validation" in prompt.lower()[:200]:
        return _VALIDATION_LLM
    elif "Score the overall quality" in prompt or "judge" in prompt.lower()[:200]:
        return _JUDGE_LLM
    # Default to extraction
    return _EXTRACTION_LLM


@pytest.fixture
def client(_mock_settings):
    """Create a test client with mocked LLM."""
    with patch("src.utils.llm_client.AzureOpenAI"), \
         patch("src.utils.llm_client.DefaultAzureCredential"), \
         patch("src.utils.llm_client.get_bearer_token_provider"), \
         patch("src.utils.llm_client.LLMClient.call", side_effect=_mock_llm_call):
        from src.app import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a fake PDF file for testing."""
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake invoice content")
    return pdf


# --- Tests: Health ------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "invoice-processing-agent"


# --- Tests: Process Invoice ---------------------------------------------------

class TestProcessInvoice:
    def test_process_single_invoice(self, client: TestClient, sample_pdf: Path) -> None:
        """Full pipeline with mocked PDF and LLM."""
        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Invoice #INV-001\nVendor: Acme Corp\nTotal: $110.00"
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            payload = {
                "pdf_path": str(sample_pdf),
                "expected": {
                    "invoice_number": "INV-001",
                    "total_amount": 110.0,
                },
            }
            resp = client.post("/api/process_invoice", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["extracted"]["invoice_number"] == "INV-001"
        assert "validation" in body

    def test_process_missing_pdf_returns_404(self, client: TestClient) -> None:
        payload = {
            "pdf_path": "/nonexistent/path/invoice.pdf",
            "expected": {"invoice_number": "X"},
        }
        resp = client.post("/api/process_invoice", json=payload)
        assert resp.status_code == 404

    def test_process_missing_pdf_path_returns_422(self, client: TestClient) -> None:
        """Missing required field pdf_path should return 422."""
        resp = client.post("/api/process_invoice", json={"expected": {}})
        assert resp.status_code == 422


# --- Tests: Evaluate Endpoint -------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_summary(self, client: TestClient, sample_pdf: Path, tmp_path: Path) -> None:
        """Evaluation with a single test case."""
        dataset = [
            {
                "case_id": "case-001",
                "pdf_path": str(sample_pdf),
                "expected": {"invoice_number": "INV-001", "total_amount": 110.0},
            }
        ]
        dataset_file = tmp_path / "test_cases.json"
        dataset_file.write_text(json.dumps(dataset), encoding="utf-8")

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Invoice #INV-001\nTotal: $110.00"
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            resp = client.post("/api/evaluate", json={"dataset_path": str(dataset_file)})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_cases"] == 1
        assert "average_deterministic_score" in body
        assert "average_llm_judge_score" in body
        assert len(body["results"]) == 1

    def test_evaluate_missing_dataset_returns_404(self, client: TestClient) -> None:
        resp = client.post("/api/evaluate", json={"dataset_path": "/nonexistent/data.json"})
        assert resp.status_code == 404

    def test_evaluate_invalid_json_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        resp = client.post("/api/evaluate", json={"dataset_path": str(bad_file)})
        assert resp.status_code == 400

    def test_evaluate_non_array_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        obj_file = tmp_path / "obj.json"
        obj_file.write_text('{"key": "value"}', encoding="utf-8")
        resp = client.post("/api/evaluate", json={"dataset_path": str(obj_file)})
        assert resp.status_code == 400
