"""Tests for the LLM client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.utils.llm_client import LLMClient
from src.utils.helpers import safe_parse_json


class TestSafeParseJson:
    def test_parses_valid_json(self) -> None:
        assert safe_parse_json('{"key": "value"}') == {"key": "value"}

    def test_strips_markdown_fences(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert safe_parse_json(text) == {"a": 1}

    def test_returns_raw_on_invalid_json(self) -> None:
        result = safe_parse_json("not json at all")
        assert "raw_response" in result
        assert result["raw_response"] == "not json at all"

    def test_handles_empty_string(self) -> None:
        result = safe_parse_json("")
        assert "raw_response" in result


class TestLLMClient:
    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        s = MagicMock(spec=Settings)
        s.azure_openai_endpoint = "https://mock.openai.azure.com/"
        s.azure_openai_api_version = "2024-12-01-preview"
        s.azure_openai_deployment = "gpt-4o"
        s.max_completion_tokens = 2000
        return s

    def test_call_returns_content(self, mock_settings) -> None:
        with patch("src.utils.llm_client.AzureOpenAI") as mock_aoai, \
             patch("src.utils.llm_client.DefaultAzureCredential"), \
             patch("src.utils.llm_client.get_bearer_token_provider"):
            mock_client = MagicMock()
            mock_aoai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"result": "ok"}'
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response

            llm = LLMClient(mock_settings)
            result = llm.call("test prompt")

        assert result == '{"result": "ok"}'
        assert llm.last_token_usage["prompt_tokens"] == 50

    def test_call_json_parses_response(self, mock_settings) -> None:
        with patch("src.utils.llm_client.AzureOpenAI") as mock_aoai, \
             patch("src.utils.llm_client.DefaultAzureCredential"), \
             patch("src.utils.llm_client.get_bearer_token_provider"):
            mock_client = MagicMock()
            mock_aoai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"result": "ok"}'
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 20
            mock_client.chat.completions.create.return_value = mock_response

            llm = LLMClient(mock_settings)
            result = llm.call_json("test prompt")

        assert result == {"result": "ok"}

    def test_empty_content_returns_empty_string(self, mock_settings) -> None:
        with patch("src.utils.llm_client.AzureOpenAI") as mock_aoai, \
             patch("src.utils.llm_client.DefaultAzureCredential"), \
             patch("src.utils.llm_client.get_bearer_token_provider"):
            mock_client = MagicMock()
            mock_aoai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = None
            mock_response.usage = None
            mock_client.chat.completions.create.return_value = mock_response

            llm = LLMClient(mock_settings)
            result = llm.call("test")

        assert result == ""
