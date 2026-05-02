"""Step 2 — Use an LLM to extract structured invoice fields from raw text."""

from __future__ import annotations

import json

from src.config.prompts import FIELD_EXTRACTION_PROMPT
from src.models.alert_models import PipelineState
from src.models.patient_vitals import ExtractedInvoice, LineItem
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_fields(state: PipelineState, llm: LLMClient, *, temperature: float = 0.0) -> PipelineState:
    """Send raw invoice text to the LLM and parse the structured result.

    Args:
        state: Pipeline state with ``raw_text`` populated.
        llm: Configured LLM client.
        temperature: LLM temperature override.

    Returns:
        The same *state* with ``extracted`` populated.
    """
    if not state.raw_text:
        raise ValueError("raw_text is empty — run PDF extraction first")

    prompt = FIELD_EXTRACTION_PROMPT.format(raw_text=state.raw_text)
    data = llm.call_json(prompt, temperature=temperature)

    # Accumulate token usage
    usage = llm.last_token_usage
    state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)

    # Parse line items
    raw_items = data.get("line_items") or []
    line_items = [
        LineItem(
            description=item.get("description", ""),
            quantity=float(item.get("quantity", 0)),
            unit_price=float(item.get("unit_price", 0)),
            amount=float(item.get("amount", 0)),
        )
        for item in raw_items
    ]

    state.extracted = ExtractedInvoice(
        invoice_number=data.get("invoice_number"),
        invoice_date=data.get("invoice_date"),
        due_date=data.get("due_date"),
        vendor_name=data.get("vendor_name"),
        customer_name=data.get("customer_name"),
        base_amount=_safe_float(data.get("base_amount")),
        tax_amount=_safe_float(data.get("tax_amount")),
        total_amount=_safe_float(data.get("total_amount")),
        currency=data.get("currency"),
        line_items=line_items,
    )

    logger.info("Extracted %d fields and %d line items", _count_non_null(state.extracted), len(line_items))
    return state


def _safe_float(val: object) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _count_non_null(inv: ExtractedInvoice) -> int:
    """Count how many top-level fields are non-null."""
    count = 0
    for field_name in ["invoice_number", "invoice_date", "due_date", "vendor_name",
                        "customer_name", "base_amount", "tax_amount", "total_amount", "currency"]:
        if getattr(inv, field_name) is not None:
            count += 1
    return count
