"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration sourced from .env / environment."""

    # Azure OpenAI
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI resource endpoint")
    azure_openai_deployment: str = Field(default="gpt-4o", description="Model deployment name")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", description="API version")

    # Application
    log_level: str = Field(default="INFO", description="Logging level")

    # LLM temperature tuning per step
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for field extraction")
    validation_temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for validation")
    judge_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature for LLM judge")

    # Token guardrails
    max_prompt_tokens: int = Field(default=4000, ge=100, description="Max tokens for prompts")
    max_completion_tokens: int = Field(default=2000, ge=100, description="Max tokens for completions")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
