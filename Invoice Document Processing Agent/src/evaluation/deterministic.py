"""Deterministic evaluator — field-by-field comparison against ground truth."""

from __future__ import annotations

from src.models.patient_vitals import ExtractedInvoice
from src.models.evaluation import DeterministicScore
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Top-level scalar fields to compare.
_SCALAR_FIELDS = [
    "invoice_number",
    "invoice_date",
    "due_date",
    "vendor_name",
    "customer_name",
    "base_amount",
    "tax_amount",
    "total_amount",
    "currency",
]


def _values_match(expected: object, extracted: object) -> bool:
    """Flexible comparison: normalise strings, allow small float tolerance."""
    if expected is None and extracted is None:
        return True
    if expected is None or extracted is None:
        return False
    # Float comparison with tolerance
    if isinstance(expected, (int, float)) and isinstance(extracted, (int, float)):
        return abs(float(expected) - float(extracted)) < 0.01
    # String comparison (case-insensitive, stripped)
    return str(expected).strip().lower() == str(extracted).strip().lower()


def score_deterministic(expected: ExtractedInvoice, extracted: ExtractedInvoice) -> DeterministicScore:
    """Compare extracted invoice against expected, field by field.

    Args:
        expected: Ground-truth invoice data.
        extracted: LLM-extracted invoice data.

    Returns:
        A :class:`DeterministicScore` with per-field details.
    """
    details: list[dict[str, object]] = []
    matched = 0
    total = 0

    # Scalar fields
    for field in _SCALAR_FIELDS:
        exp_val = getattr(expected, field, None)
        ext_val = getattr(extracted, field, None)
        match = _values_match(exp_val, ext_val)
        if match:
            matched += 1
        total += 1
        details.append({
            "field": field,
            "expected": exp_val,
            "extracted": ext_val,
            "match": match,
        })

    # Line items
    for idx, exp_item in enumerate(expected.line_items):
        ext_item = extracted.line_items[idx] if idx < len(extracted.line_items) else None
        for attr in ("description", "quantity", "unit_price", "amount"):
            exp_v = getattr(exp_item, attr, None)
            ext_v = getattr(ext_item, attr, None) if ext_item else None
            match = _values_match(exp_v, ext_v)
            if match:
                matched += 1
            total += 1
            details.append({
                "field": f"line_items[{idx}].{attr}",
                "expected": exp_v,
                "extracted": ext_v,
                "match": match,
            })

    # Extra extracted line items count as mismatches
    if len(extracted.line_items) > len(expected.line_items):
        extra = len(extracted.line_items) - len(expected.line_items)
        for i in range(len(expected.line_items), len(extracted.line_items)):
            total += 1
            details.append({
                "field": f"line_items[{i}] (extra)",
                "expected": None,
                "extracted": "present",
                "match": False,
            })

    score = matched / total if total > 0 else 0.0

    logger.info("Deterministic score: %d/%d = %.2f", matched, total, score)
    return DeterministicScore(
        total_fields=total,
        matched_fields=matched,
        score=round(score, 4),
        details=details,
    )
