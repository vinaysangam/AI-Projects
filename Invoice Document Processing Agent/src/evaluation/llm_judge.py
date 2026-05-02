"""LLM-as-Judge evaluator — holistic quality scoring on a 0–1 scale."""

from __future__ import annotations

from src.config.prompts import LLM_JUDGE_PROMPT
from src.models.patient_vitals import ExtractedInvoice, ValidationReport
from src.models.evaluation import LLMJudgeScore
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def score_with_llm_judge(
    expected: ExtractedInvoice,
    extracted: ExtractedInvoice,
    validation: ValidationReport,
    llm: LLMClient,
    *,
    temperature: float = 0.3,
) -> LLMJudgeScore:
    """Ask an LLM to holistically score the extraction quality.

    Args:
        expected: Ground-truth invoice.
        extracted: LLM-extracted invoice.
        validation: The validation report from step 3.
        llm: Configured LLM client.
        temperature: LLM temperature.

    Returns:
        An :class:`LLMJudgeScore` with score and reasoning.
    """
    prompt = LLM_JUDGE_PROMPT.format(
        extracted_json=extracted.model_dump_json(indent=2),
        expected_json=expected.model_dump_json(indent=2),
        validation_json=validation.model_dump_json(indent=2),
    )

    data = llm.call_json(prompt, temperature=temperature)

    raw_score = data.get("score", 0.0)
    try:
        score = max(0.0, min(1.0, float(raw_score)))
    except (TypeError, ValueError):
        score = 0.0

    reasoning = str(data.get("reasoning", ""))

    logger.info("LLM judge score: %.2f", score)
    return LLMJudgeScore(score=round(score, 4), reasoning=reasoning)
