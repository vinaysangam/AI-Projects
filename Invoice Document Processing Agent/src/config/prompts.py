"""Prompt templates for the invoice-processing AI pipeline."""

FIELD_EXTRACTION_PROMPT = """\
You are an expert invoice-processing AI. Extract structured data from the raw text of an invoice.

Raw Invoice Text:
---
{raw_text}
---

Instructions:
1. Extract the following fields exactly:
   - invoice_number (string)
   - invoice_date (string, ISO-8601 format YYYY-MM-DD if possible)
   - due_date (string, ISO-8601 if available, otherwise null)
   - vendor_name (string)
   - customer_name (string or null)
   - base_amount (number — subtotal before tax)
   - tax_amount (number)
   - total_amount (number)
   - currency (string, e.g. "USD", "EUR")
   - line_items — an array of objects each containing:
       description (string), quantity (number), unit_price (number), amount (number)
2. If a field cannot be found, set it to null.
3. Return ONLY valid JSON — no markdown fences, no commentary.

Respond with a JSON object matching the schema above.
"""

VALIDATION_PROMPT = """\
You are an invoice-validation AI. Compare extracted invoice data against the expected \
values and produce a field-by-field validation report.

Extracted Data:
{extracted_json}

Expected Data:
{expected_json}

Instructions:
1. For each field in the expected data, compare it to the extracted value.
2. For line_items, compare each item's description, quantity, unit_price, and amount.
3. Produce a JSON object with:
   - "field_results": a list of objects, each with:
       "field" (string), "expected", "extracted", "match" (bool), "note" (string or null)
   - "line_item_results": a list of objects, each with:
       "index" (int), "field" (string), "expected", "extracted", "match" (bool), "note" (string or null)
   - "summary": a short overall assessment string
   - "all_match" (bool): true only if every field and line-item matched
4. Return ONLY valid JSON.
"""

LLM_JUDGE_PROMPT = """\
You are a quality-evaluation AI. Score the overall quality of an invoice extraction result.

Extracted Invoice:
{extracted_json}

Expected Invoice:
{expected_json}

Validation Report:
{validation_json}

Instructions:
1. Consider accuracy of every field and line item.
2. Consider partial matches (e.g., minor formatting differences in dates).
3. Return a JSON object with:
   - "score": a float between 0.0 and 1.0 (1.0 = perfect extraction)
   - "reasoning": a brief explanation of the score
4. Return ONLY valid JSON.
"""
