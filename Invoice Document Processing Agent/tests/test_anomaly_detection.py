"""Tests for PDF text extraction (Step 1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.alert_models import PipelineState
from src.pipeline.pdf_extractor import extract_pdf_text


class TestPDFExtractor:
    def test_extracts_text_from_pdf(self, tmp_path: Path) -> None:
        """Mock pdfplumber to simulate reading a single-page PDF."""
        state = PipelineState(pdf_path=str(tmp_path / "invoice.pdf"))

        # Create a dummy file so FileNotFoundError is not raised
        (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 fake")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Invoice #12345\nTotal: $100.00"

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            result = extract_pdf_text(state)

        assert "Invoice #12345" in result.raw_text
        assert "Total: $100.00" in result.raw_text

    def test_multi_page_extraction(self, tmp_path: Path) -> None:
        """Multiple pages are joined with double newline."""
        (tmp_path / "multi.pdf").write_bytes(b"%PDF-1.4 fake")
        state = PipelineState(pdf_path=str(tmp_path / "multi.pdf"))

        page1 = MagicMock()
        page1.extract_text.return_value = "Page 1 text"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page 2 text"

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_pdf = MagicMock()
            mock_pdf.pages = [page1, page2]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            result = extract_pdf_text(state)

        assert "Page 1 text" in result.raw_text
        assert "Page 2 text" in result.raw_text

    def test_file_not_found_raises(self) -> None:
        state = PipelineState(pdf_path="/nonexistent/path/invoice.pdf")
        with pytest.raises(FileNotFoundError):
            extract_pdf_text(state)

    def test_empty_pdf_raises_value_error(self, tmp_path: Path) -> None:
        """A PDF with no extractable text should raise ValueError."""
        (tmp_path / "empty.pdf").write_bytes(b"%PDF-1.4 fake")
        state = PipelineState(pdf_path=str(tmp_path / "empty.pdf"))

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(ValueError, match="No text extracted"):
                extract_pdf_text(state)

    def test_none_page_text_skipped(self, tmp_path: Path) -> None:
        """Pages returning None text are skipped."""
        (tmp_path / "partial.pdf").write_bytes(b"%PDF-1.4 fake")
        state = PipelineState(pdf_path=str(tmp_path / "partial.pdf"))

        page1 = MagicMock()
        page1.extract_text.return_value = None
        page2 = MagicMock()
        page2.extract_text.return_value = "Real content"

        with patch("src.pipeline.pdf_extractor.pdfplumber") as mock_plumber:
            mock_pdf = MagicMock()
            mock_pdf.pages = [page1, page2]
            mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
            mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

            result = extract_pdf_text(state)

        assert result.raw_text == "Real content"
