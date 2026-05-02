# This module has been removed — the invoice pipeline does not use risk scoring.

import numpy as np

from src.models.alert_models import AnomalyFlag, RiskLevel, RiskScore
from src.models.patient_vitals import PatientTelemetry
from src.utils.anomaly_detection import detect_multi_signal_patterns
from src.utils.logger import get_logger
from src.utils.telemetry_ingestion import normalise_telemetry

logger = get_logger(__name__)

# Weights reflect clinical importance of each vital sign for deterioration.
_VITAL_WEIGHTS: dict[str, float] = {
    "spo2": 0.25,
    "heart_rate": 0.18,
    "systolic_bp": 0.15,
    "diastolic_bp": 0.08,
    "respiratory_rate": 0.15,
    "glucose": 0.10,
    "temperature": 0.09,
}


def calculate_risk_score(
    telemetry: PatientTelemetry,
    anomalies: list[AnomalyFlag],
    threshold: float = 0.6,
) -> RiskScore:
    """Calculate a composite deterioration risk score.

    The score is a weighted combination of normalised vital deviations,
    boosted when multi-signal patterns are present.

    Args:
        telemetry: Patient telemetry payload.
        anomalies: Detected anomalies.
        threshold: Score above which risk is elevated (used for level assignment).

    Returns:
        A :class:`RiskScore` instance.
    """
    normalised = normalise_telemetry(telemetry)

    # Weighted sum of normalised deviations.
    weighted_scores: list[float] = []
    for vtype, norm_val in normalised.items():
        weight = _VITAL_WEIGHTS.get(vtype.value, 0.1)
        weighted_scores.append(norm_val * weight)

    base_score = float(np.clip(sum(weighted_scores) / max(sum(_VITAL_WEIGHTS.values()), 1e-6) * 2.0, 0.0, 1.0))

    # Boost for multi-signal patterns.
    patterns = detect_multi_signal_patterns(anomalies)
    pattern_boost = min(len(patterns) * 0.1, 0.25)

    # Boost for number of anomalies.
    anomaly_boost = min(len(anomalies) * 0.05, 0.2)

    final_score = float(np.clip(base_score + pattern_boost + anomaly_boost, 0.0, 1.0))

    # Determine level.
    if final_score >= 0.75:
        level = RiskLevel.HIGH
    elif final_score >= threshold:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    contributing = [a.description for a in anomalies]
    contributing.extend(patterns)

    risk = RiskScore(
        patient_id=telemetry.patient_id,
        timestamp=telemetry.timestamp,
        score=round(final_score, 3),
        level=level,
        contributing_factors=contributing,
    )

    logger.info(
        "Patient %s risk: score=%.3f level=%s factors=%d",
        telemetry.patient_id,
        risk.score,
        risk.level.value,
        len(contributing),
    )
    return risk
