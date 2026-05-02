"""Pydantic models for invoice data."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    amount: float = 0.0


class ExtractedInvoice(BaseModel):
    """Structured data extracted from an invoice by the LLM."""

    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    vendor_name: str | None = None
    customer_name: str | None = None
    base_amount: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class FieldResult(BaseModel):
    """Comparison result for a single field."""

    field: str
    expected: str | float | None = None
    extracted: str | float | None = None
    match: bool = False
    note: str | None = None


class LineItemResult(BaseModel):
    """Comparison result for a single line-item field."""

    index: int
    field: str
    expected: str | float | None = None
    extracted: str | float | None = None
    match: bool = False
    note: str | None = None


class ValidationReport(BaseModel):
    """Full validation report comparing extracted vs expected data."""

    field_results: list[FieldResult] = Field(default_factory=list)
    line_item_results: list[LineItemResult] = Field(default_factory=list)
    summary: str = ""
    all_match: bool = False


class InvoiceRequest(BaseModel):
    """Incoming API request for processing a single invoice."""

    pdf_path: str = Field(..., min_length=1, description="Path to the PDF file")
    expected: ExtractedInvoice = Field(..., description="Expected field values for validation")


class InvoiceResponse(BaseModel):
    """API response for a processed invoice."""

    raw_text: str = ""
    extracted: ExtractedInvoice = Field(default_factory=ExtractedInvoice)
    validation: ValidationReport = Field(default_factory=ValidationReport)
