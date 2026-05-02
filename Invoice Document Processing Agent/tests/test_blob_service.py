"""Tests for the deterministic evaluator."""

from __future__ import annotations

import pytest

from src.evaluation.deterministic import score_deterministic, _values_match
from src.models.patient_vitals import ExtractedInvoice, LineItem


class TestValuesMatch:
    def test_both_none(self) -> None:
        assert _values_match(None, None) is True

    def test_one_none(self) -> None:
        assert _values_match(None, "x") is False
        assert _values_match("x", None) is False

    def test_floats_match_within_tolerance(self) -> None:
        assert _values_match(100.0, 100.005) is True

    def test_floats_mismatch(self) -> None:
        assert _values_match(100.0, 101.0) is False

    def test_strings_case_insensitive(self) -> None:
        assert _values_match("Acme Corp", "acme corp") is True

    def test_strings_with_whitespace(self) -> None:
        assert _values_match("  Hello ", "hello") is True


class TestDeterministicScorer:
    def test_perfect_match(self) -> None:
        inv = ExtractedInvoice(
            invoice_number="INV-1",
            invoice_date="2026-01-01",
            vendor_name="Acme",
            total_amount=500.0,
            line_items=[LineItem(description="Widget", quantity=10, unit_price=50.0, amount=500.0)],
        )
        result = score_deterministic(inv, inv)
        assert result.score == 1.0
        assert result.matched_fields == result.total_fields

    def test_partial_match(self) -> None:
        expected = ExtractedInvoice(invoice_number="INV-1", vendor_name="Acme", total_amount=500.0)
        extracted = ExtractedInvoice(invoice_number="INV-1", vendor_name="WRONG", total_amount=500.0)
        result = score_deterministic(expected, extracted)
        assert 0.0 < result.score < 1.0
        mismatches = [d for d in result.details if not d["match"]]
        assert any(d["field"] == "vendor_name" for d in mismatches)

    def test_no_match(self) -> None:
        expected = ExtractedInvoice(invoice_number="A", vendor_name="B", total_amount=100.0)
        extracted = ExtractedInvoice(invoice_number="X", vendor_name="Y", total_amount=999.0)
        result = score_deterministic(expected, extracted)
        assert result.score < 1.0

    def test_line_item_comparison(self) -> None:
        expected = ExtractedInvoice(
            line_items=[LineItem(description="Gadget", quantity=5, unit_price=20.0, amount=100.0)]
        )
        extracted = ExtractedInvoice(
            line_items=[LineItem(description="Gadget", quantity=5, unit_price=20.0, amount=100.0)]
        )
        result = score_deterministic(expected, extracted)
        li_details = [d for d in result.details if "line_items" in d["field"]]
        assert all(d["match"] for d in li_details)

    def test_extra_extracted_line_items(self) -> None:
        expected = ExtractedInvoice(line_items=[])
        extracted = ExtractedInvoice(
            line_items=[LineItem(description="Extra", quantity=1, unit_price=10.0, amount=10.0)]
        )
        result = score_deterministic(expected, extracted)
        extras = [d for d in result.details if "extra" in str(d.get("field", ""))]
        assert len(extras) == 1
        assert extras[0]["match"] is False

    def test_empty_invoices_score_high(self) -> None:
        """Two empty invoices should score 1.0 (all Nones match)."""
        result = score_deterministic(ExtractedInvoice(), ExtractedInvoice())
        assert result.score == 1.0

from unittest.mock import MagicMock, patch

import pytest

from src.utils.blob_service import BlobStorageService


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.azure_storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=mock;AccountKey=bW9jaw==;EndpointSuffix=core.windows.net"
    settings.azure_storage_container_name = "test-container"
    return settings


@pytest.fixture
def disabled_settings():
    settings = MagicMock()
    settings.azure_storage_connection_string = ""
    return settings


class TestBlobServiceDisabled:
    def test_disabled_when_no_config(self, disabled_settings) -> None:
        service = BlobStorageService(disabled_settings)
        assert service.enabled is False

    def test_list_uploads_empty_when_disabled(self, disabled_settings) -> None:
        service = BlobStorageService(disabled_settings)
        assert service.list_uploads() == []

    def test_upload_raises_when_disabled(self, disabled_settings) -> None:
        service = BlobStorageService(disabled_settings)
        with pytest.raises(RuntimeError, match="not configured"):
            service.upload_document(b"data", "test.json")


class TestBlobServiceValidation:
    @patch("src.utils.blob_service.BlobServiceClient")
    def test_rejects_disallowed_extension(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        service._container_client = MagicMock()
        with pytest.raises(ValueError, match="not allowed"):
            service.upload_document(b"data", "test.exe")

    @patch("src.utils.blob_service.BlobServiceClient")
    def test_rejects_oversized_file(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        service._container_client = MagicMock()
        # 11 MB of data
        big_data = b"x" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds limit"):
            service.upload_document(big_data, "large.json")

    @patch("src.utils.blob_service.BlobServiceClient")
    def test_accepts_json_extension(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        mock_container = MagicMock()
        mock_blob = MagicMock()
        mock_blob.url = "https://mock.blob.core.windows.net/test/blob.json"
        mock_container.get_blob_client.return_value = mock_blob
        service._container_client = mock_container

        result = service.upload_document(b'{"test": true}', "data.json")
        assert result["upload_id"]
        assert result["blob_name"]
        mock_blob.upload_blob.assert_called_once()


class TestParseUploadedJson:
    @patch("src.utils.blob_service.BlobServiceClient")
    def test_parses_single_object(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        result = service.parse_uploaded_json(b'{"patient_id": "P-001"}')
        assert len(result) == 1
        assert result[0]["patient_id"] == "P-001"

    @patch("src.utils.blob_service.BlobServiceClient")
    def test_parses_array(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        result = service.parse_uploaded_json(b'[{"a": 1}, {"b": 2}]')
        assert len(result) == 2

    @patch("src.utils.blob_service.BlobServiceClient")
    def test_rejects_non_object_json(self, mock_client, mock_settings) -> None:
        service = BlobStorageService(mock_settings)
        with pytest.raises(ValueError, match="must be an object or array"):
            service.parse_uploaded_json(b'"just a string"')
