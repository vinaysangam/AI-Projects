# This module has been removed — the invoice pipeline uses llm_client.py instead.

import json
from typing import Any

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.prompts import (
    ALERT_EXPLANATION_PROMPT,
    DETERIORATION_ASSESSMENT_PROMPT,
    INTERVENTION_RECOMMENDATION_PROMPT,
    QUALITY_CHECK_PROMPT,
)
from src.config.settings import Settings
from src.models.alert_models import (
    AnomalyFlag,
    InterventionRecommendation,
    MonitoringSummary,
    PatientAlert,
    RiskScore,
)
from src.models.patient_vitals import PatientTelemetry
from src.utils.logger import get_logger
from src.utils.tiktoken_utils import count_tokens, truncate_to_token_limit

logger = get_logger(__name__)


class InterventionAgent:
    """Orchestrates Azure OpenAI calls for clinical recommendations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.azure_openai_deployment
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._last_token_usage: dict[str, int] = {}

    # --- Public API -----------------------------------------------------------

    def generate_recommendation(
        self,
        telemetry: PatientTelemetry,
        anomalies: list[AnomalyFlag],
        risk: RiskScore,
    ) -> InterventionRecommendation:
        """Generate an intervention recommendation for a patient.

        Args:
            telemetry: Current telemetry payload.
            anomalies: Detected anomalies.
            risk: Calculated risk score.

        Returns:
            An :class:`InterventionRecommendation`.
        """
        prompt = INTERVENTION_RECOMMENDATION_PROMPT.format(
            patient_id=telemetry.patient_id,
            timestamp=telemetry.timestamp.isoformat(),
            vitals_summary=self._format_vitals(telemetry),
            anomalies_summary=self._format_anomalies(anomalies),
            risk_score=risk.score,
            risk_level=risk.level.value,
            contributing_factors=", ".join(risk.contributing_factors),
        )
        response = self._call_llm(prompt)
        parsed = self._safe_parse_json(response)

        return InterventionRecommendation(
            patient_id=telemetry.patient_id,
            timestamp=telemetry.timestamp,
            recommendation=json.dumps(parsed.get("recommendations", []), indent=2),
            reasoning=parsed.get("reasoning", response),
            confidence=0.0,  # will be set by quality check
            token_usage=self._last_token_usage,
        )

    def generate_assessment(
        self,
        telemetry: PatientTelemetry,
        anomalies: list[AnomalyFlag],
        risk: RiskScore,
    ) -> str:
        """Generate a deterioration assessment summary."""
        prompt = DETERIORATION_ASSESSMENT_PROMPT.format(
            patient_id=telemetry.patient_id,
            timestamp=telemetry.timestamp.isoformat(),
            vitals_summary=self._format_vitals(telemetry),
            anomalies_summary=self._format_anomalies(anomalies),
            risk_score=risk.score,
            risk_level=risk.level.value,
        )
        return self._call_llm(prompt)

    def explain_alert(self, alert: PatientAlert) -> str:
        """Generate a human-readable explanation for an alert."""
        prompt = ALERT_EXPLANATION_PROMPT.format(
            patient_id=alert.patient_id,
            severity=alert.severity.value,
            anomalies_detail=self._format_anomalies(alert.anomalies),
        )
        return self._call_llm(prompt)

    def quality_check(
        self,
        recommendation: InterventionRecommendation,
        telemetry: PatientTelemetry,
    ) -> dict[str, Any]:
        """Run a quality check on a generated recommendation."""
        prompt = QUALITY_CHECK_PROMPT.format(
            recommendation_text=recommendation.recommendation,
            patient_context=self._format_vitals(telemetry),
        )
        response = self._call_llm(prompt)
        return self._safe_parse_json(response)

    @property
    def total_tokens_used(self) -> dict[str, int]:
        """Cumulative token usage across all calls."""
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total": self._total_prompt_tokens + self._total_completion_tokens,
        }

    # --- Orchestration --------------------------------------------------------

    def run_full_pipeline(
        self,
        telemetry: PatientTelemetry,
        anomalies: list[AnomalyFlag],
        risk: RiskScore,
        alert: PatientAlert | None,
    ) -> MonitoringSummary:
        """Run the full multi-agent recommendation pipeline.

        Agents:
        1. Monitoring Agent — collects data (already done upstream).
        2. Anomaly Detection Agent — detects anomalies (already done upstream).
        3. Risk Prediction Agent — scores risk (already done upstream).
        4. Recommendation Agent — generates intervention + explanation.

        Args:
            telemetry: Patient telemetry.
            anomalies: Detected anomalies.
            risk: Calculated risk score.
            alert: Constructed alert (may be None).

        Returns:
            A complete :class:`MonitoringSummary`.
        """
        recommendation: InterventionRecommendation | None = None
        explanation: str = ""

        # Only invoke LLM for medium/high risk to conserve tokens.
        if risk.level.value in ("medium", "high"):
            logger.info("Patient %s: risk is %s — invoking recommendation agent", telemetry.patient_id, risk.level.value)
            recommendation = self.generate_recommendation(telemetry, anomalies, risk)

            # Quality check
            qc = self.quality_check(recommendation, telemetry)
            recommendation.confidence = qc.get("confidence", 0.0)

            if alert:
                explanation = self.explain_alert(alert)
                alert.explanation = explanation
        else:
            logger.info("Patient %s: risk is LOW — skipping LLM recommendation", telemetry.patient_id)

        return MonitoringSummary(
            patient_id=telemetry.patient_id,
            timestamp=telemetry.timestamp,
            alert=alert,
            risk=risk,
            recommendation=recommendation,
            raw_readings=telemetry.readings_dict(),
        )

    # --- Private helpers ------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_llm(self, prompt: str) -> str:
        """Call Azure OpenAI with retry logic and token guardrails."""
        prompt = truncate_to_token_limit(
            prompt,
            self._settings.max_prompt_tokens,
        )

        token_count = count_tokens(prompt)
        logger.debug("Sending prompt (%d tokens) to %s", token_count, self._deployment)

        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": "You are a clinical AI decision-support assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self._settings.summary_temperature,
            max_tokens=self._settings.max_completion_tokens,
        )

        usage = response.usage
        self._last_token_usage: dict[str, int] = {}
        if usage:
            self._last_token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
            self._total_prompt_tokens += usage.prompt_tokens
            self._total_completion_tokens += usage.completion_tokens
            logger.info(
                "LLM call: prompt=%d completion=%d tokens",
                usage.prompt_tokens,
                usage.completion_tokens,
            )

        content = response.choices[0].message.content or ""
        return content.strip()

    @staticmethod
    def _format_vitals(telemetry: PatientTelemetry) -> str:
        lines = []
        for r in telemetry.readings:
            lines.append(f"- {r.vital_type.value}: {r.value} {r.unit}")
        return "\n".join(lines)

    @staticmethod
    def _format_anomalies(anomalies: list[AnomalyFlag]) -> str:
        if not anomalies:
            return "None"
        lines = []
        for a in anomalies:
            lines.append(
                f"- {a.vital_type}: {a.value} (normal {a.normal_range[0]}-{a.normal_range[1]}) "
                f"— deviation {a.deviation_score:.2f} — {a.description}"
            )
        return "\n".join(lines)

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any]:
        """Attempt to parse JSON from LLM response, handling markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response; returning raw text")
            return {"raw_response": text}
