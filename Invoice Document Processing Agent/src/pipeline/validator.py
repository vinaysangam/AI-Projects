"""Step 3 — Use an LLM to validate extracted fields against expected values."""

from __future__ import annotations

import json

from src.config.prompts import VALIDATION_PROMPT
from src.models.alert_models import PipelineState
from src.models.patient_vitals import (
    FieldResult,
    LineItemResult,
    ValidationReport,
)
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_extraction(state: PipelineState, llm: LLMClient, *, temperature: float = 0.0) -> PipelineState:
    """Compare extracted invoice data against expected values using an LLM.

    Args:
        state: Pipeline state with ``extracted`` and ``expected`` populated.
        llm: Configured LLM client.
        temperature: LLM temperature override.

    Returns:
        The same *state* with ``validation`` populated.
    """
    extracted_json = state.extracted.model_dump_json(indent=2)
    expected_json = state.expected.model_dump_json(indent=2)

    prompt = VALIDATION_PROMPT.format(
        extracted_json=extracted_json,
        expected_json=expected_json,
    )
    data = llm.call_json(prompt, temperature=temperature)

    # Accumulate token usage
    usage = llm.last_token_usage
    state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)

    # Parse field results
    field_results = [
        FieldResult(
            field=fr.get("field", ""),
            expected=fr.get("expected"),
            extracted=fr.get("extracted"),
            match=bool(fr.get("match", False)),
            note=fr.get("note"),
        )
        for fr in data.get("field_results", [])
    ]

    # Parse line-item results
    line_item_results = [
        LineItemResult(
            index=int(lr.get("index", 0)),
            field=lr.get("field", ""),
            expected=lr.get("expected"),
            extracted=lr.get("extracted"),
            match=bool(lr.get("match", False)),
            note=lr.get("note"),
        )
        for lr in data.get("line_item_results", [])
    ]

    state.validation = ValidationReport(
        field_results=field_results,
        line_item_results=line_item_results,
        summary=data.get("summary", ""),
        all_match=bool(data.get("all_match", False)),
    )

    matched = sum(1 for fr in field_results if fr.match)
    logger.info("Validation: %d/%d fields matched, all_match=%s", matched, len(field_results), state.validation.all_match)
    return state
