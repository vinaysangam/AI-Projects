"""Singleton Azure OpenAI wrapper with retry and JSON parsing."""

from __future__ import annotations

import json
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import Settings
from src.utils.helpers import safe_parse_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Thin wrapper around Azure OpenAI chat completions using managed identity."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default",
        )
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment
        self._last_token_usage: dict[str, int] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def call(self, prompt: str, *, temperature: float | None = None) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            prompt: The full prompt string.
            temperature: Override for generation temperature.

        Returns:
            The assistant message content as a string.
        """
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self._settings.max_completion_tokens,
        )

        usage = response.usage
        if usage:
            self._last_token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
            logger.info(
                "LLM call: prompt=%d completion=%d tokens",
                usage.prompt_tokens,
                usage.completion_tokens,
            )

        content = response.choices[0].message.content or ""
        return content.strip()

    def call_json(self, prompt: str, *, temperature: float | None = None) -> dict[str, Any]:
        """Call the LLM and parse the response as JSON."""
        raw = self.call(prompt, temperature=temperature)
        return safe_parse_json(raw)

    @property
    def last_token_usage(self) -> dict[str, int]:
        return dict(self._last_token_usage)
