# This module has been removed — the invoice pipeline does not use telemetry validation.

from src.models.patient_vitals import PatientTelemetry, VitalReading, VitalType, VITAL_RANGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when telemetry data fails validation."""


def validate_telemetry(telemetry: PatientTelemetry) -> list[str]:
    """Validate a patient telemetry payload.

    Returns a list of warning strings.  Raises :class:`ValidationError` for
    fatal issues.
    """
    warnings: list[str] = []

    if not telemetry.patient_id.strip():
        raise ValidationError("patient_id must be a non-empty string")

    if not telemetry.readings:
        raise ValidationError("At least one vital reading is required")

    for reading in telemetry.readings:
        w = _validate_reading(reading)
        warnings.extend(w)

    if warnings:
        logger.info("Validation warnings for patient %s: %s", telemetry.patient_id, warnings)

    return warnings


def _validate_reading(reading: VitalReading) -> list[str]:
    """Validate a single vital reading and return warnings."""
    warnings: list[str] = []

    if reading.value < 0:
        warnings.append(f"{reading.vital_type.value}: negative value ({reading.value})")

    # Physiological plausibility checks
    plausibility: dict[VitalType, tuple[float, float]] = {
        VitalType.HEART_RATE: (20.0, 300.0),
        VitalType.SYSTOLIC_BP: (40.0, 300.0),
        VitalType.DIASTOLIC_BP: (20.0, 200.0),
        VitalType.SPO2: (0.0, 100.0),
        VitalType.GLUCOSE: (10.0, 800.0),
        VitalType.RESPIRATORY_RATE: (4.0, 60.0),
        VitalType.TEMPERATURE: (30.0, 45.0),
    }

    bounds = plausibility.get(reading.vital_type)
    if bounds and not (bounds[0] <= reading.value <= bounds[1]):
        warnings.append(
            f"{reading.vital_type.value}: value {reading.value} outside plausible range {bounds}"
        )

    return warnings


def is_within_normal_range(vital_type: VitalType, value: float) -> bool:
    """Check whether *value* falls within the normal range for *vital_type*."""
    low, high = VITAL_RANGES.get(vital_type, (0.0, 0.0))
    return low <= value <= high
