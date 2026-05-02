# This module has been removed — the invoice pipeline does not use alert resolution.

import json
from typing import Any

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.resolver_prompts import RESOLVER_PROMPT
from src.config.settings import Settings
from src.models.alert_models import PatientAlert
from src.utils.logger import get_logger
from src.utils.tiktoken_utils import truncate_to_token_limit

logger = get_logger(__name__)


class AlertResolver:
    """Resolves, deduplicates, and triages a batch of patient alerts via LLM."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment

    def resolve(self, alerts: list[PatientAlert]) -> dict[str, Any]:
        """Run alert resolution on a batch of alerts.

        Args:
            alerts: List of active patient alerts.

        Returns:
            Parsed JSON resolution from the LLM (incidents, suppressed alerts).
        """
        if not alerts:
            return {"incidents": [], "suppressed_alerts": []}

        alerts_json = json.dumps(
            [a.model_dump(mode="json") for a in alerts],
            indent=2,
            default=str,
        )

        prompt = RESOLVER_PROMPT.format(alerts_json=alerts_json)
        prompt = truncate_to_token_limit(prompt, self._settings.max_prompt_tokens)

        response = self._call_llm(prompt)
        return self._safe_parse_json(response)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        """Call Azure OpenAI with exponential backoff retry."""
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": "You are a clinical alert triage AI."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self._settings.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse resolver JSON; returning raw text")
            return {"raw_response": text}
