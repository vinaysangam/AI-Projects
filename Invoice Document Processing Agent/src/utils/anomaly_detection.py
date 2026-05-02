# This module has been removed — the invoice pipeline does not use anomaly detection.

from src.models.alert_models import AnomalyFlag, PatientAlert, Severity
from src.models.patient_vitals import PatientTelemetry, VitalType, VITAL_RANGES
from src.utils.logger import get_logger
from src.utils.telemetry_ingestion import normalise_reading

logger = get_logger(__name__)

# Clinically meaningful anomaly descriptions.
_ANOMALY_LABELS: dict[VitalType, dict[str, str]] = {
    VitalType.HEART_RATE: {
        "high": "Tachycardia detected",
        "low": "Bradycardia detected",
    },
    VitalType.SPO2: {
        "low": "Hypoxia detected — SpO2 below safe threshold",
    },
    VitalType.GLUCOSE: {
        "high": "Hyperglycaemia — glucose spike detected",
        "low": "Hypoglycaemia — dangerously low glucose",
    },
    VitalType.SYSTOLIC_BP: {
        "high": "Hypertension — elevated systolic BP",
        "low": "Hypotension — low systolic BP",
    },
    VitalType.DIASTOLIC_BP: {
        "high": "Elevated diastolic blood pressure",
        "low": "Low diastolic blood pressure",
    },
    VitalType.RESPIRATORY_RATE: {
        "high": "Tachypnoea — elevated respiratory rate",
        "low": "Bradypnoea — abnormally low respiratory rate",
    },
    VitalType.TEMPERATURE: {
        "high": "Fever / hyperthermia detected",
        "low": "Hypothermia detected",
    },
}


def detect_anomalies(
    telemetry: PatientTelemetry,
    threshold: float = 0.7,
) -> list[AnomalyFlag]:
    """Detect anomalous vital-sign readings.

    A reading is flagged as anomalous when its normalised deviation score
    exceeds *threshold*.

    Args:
        telemetry: Patient telemetry payload.
        threshold: Normalised score above which a reading is anomalous.

    Returns:
        List of :class:`AnomalyFlag` instances for readings that exceed the
        threshold.
    """
    flags: list[AnomalyFlag] = []

    for reading in telemetry.readings:
        score = normalise_reading(reading)
        if score < threshold:
            continue

        normal_range = VITAL_RANGES.get(reading.vital_type, (0.0, 0.0))
        direction = "high" if reading.value > normal_range[1] else "low"
        labels = _ANOMALY_LABELS.get(reading.vital_type, {})
        description = labels.get(direction, f"Abnormal {reading.vital_type.value}")

        flags.append(
            AnomalyFlag(
                vital_type=reading.vital_type.value,
                value=reading.value,
                normal_range=normal_range,
                deviation_score=round(score, 3),
                description=description,
            )
        )

    logger.info(
        "Patient %s: %d anomalies detected out of %d readings",
        telemetry.patient_id,
        len(flags),
        len(telemetry.readings),
    )
    return flags


def detect_multi_signal_patterns(anomalies: list[AnomalyFlag]) -> list[str]:
    """Identify clinically significant multi-signal deterioration patterns.

    Args:
        anomalies: List of detected anomalies.

    Returns:
        List of pattern description strings.
    """
    vitals_flagged = {a.vital_type for a in anomalies}
    patterns: list[str] = []

    # Respiratory failure pattern: low SpO2 + high respiratory rate
    if "spo2" in vitals_flagged and "respiratory_rate" in vitals_flagged:
        patterns.append("Respiratory distress pattern: low SpO2 with compensatory tachypnoea")

    # Cardiogenic shock pattern: low BP + high heart rate
    if ("systolic_bp" in vitals_flagged or "diastolic_bp" in vitals_flagged) and "heart_rate" in vitals_flagged:
        patterns.append("Possible cardiogenic shock: hypotension with tachycardia")

    # Sepsis pattern: fever + tachycardia + tachypnoea
    if "temperature" in vitals_flagged and "heart_rate" in vitals_flagged and "respiratory_rate" in vitals_flagged:
        patterns.append("Sepsis-like pattern: fever with tachycardia and tachypnoea")

    # Diabetic emergency: glucose spike + tachycardia
    if "glucose" in vitals_flagged and "heart_rate" in vitals_flagged:
        patterns.append("Diabetic emergency pattern: glucose anomaly with cardiac stress")

    if patterns:
        logger.warning("Multi-signal patterns detected: %s", patterns)

    return patterns


def build_alert(
    telemetry: PatientTelemetry,
    anomalies: list[AnomalyFlag],
    patterns: list[str],
) -> PatientAlert | None:
    """Build a patient alert from anomalies and detected patterns.

    Returns ``None`` if no anomalies are present.

    Args:
        telemetry: The source telemetry.
        anomalies: Detected anomalies.
        patterns: Multi-signal pattern descriptions.

    Returns:
        A :class:`PatientAlert` or ``None``.
    """
    if not anomalies:
        return None

    max_score = max(a.deviation_score for a in anomalies)
    has_patterns = len(patterns) > 0

    if max_score >= 0.95 or (has_patterns and max_score >= 0.85):
        severity = Severity.CRITICAL
    elif max_score >= 0.85 or has_patterns:
        severity = Severity.HIGH
    elif max_score >= 0.75:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    summary_parts = [a.description for a in anomalies]
    if patterns:
        summary_parts.extend(patterns)

    return PatientAlert(
        patient_id=telemetry.patient_id,
        timestamp=telemetry.timestamp,
        severity=severity,
        anomalies=anomalies,
        summary="; ".join(summary_parts),
    )
