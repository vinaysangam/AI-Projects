# This module has been removed — the invoice pipeline does not use telemetry ingestion.

import json
from pathlib import Path
from typing import Any

from src.models.patient_vitals import PatientTelemetry, VitalReading, VitalType, VITAL_RANGES
from src.utils.logger import get_logger
from src.utils.validation import validate_telemetry

logger = get_logger(__name__)


def load_telemetry_from_file(path: str | Path) -> list[PatientTelemetry]:
    """Load telemetry records from a JSON file.

    The file should contain either a single telemetry object or a JSON array.

    Args:
        path: Path to the JSON file.

    Returns:
        List of validated :class:`PatientTelemetry` instances.
    """
    path = Path(path)
    raw: Any = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        raw = [raw]

    records: list[PatientTelemetry] = []
    for idx, item in enumerate(raw):
        try:
            telemetry = PatientTelemetry(**item)
            validate_telemetry(telemetry)
            records.append(telemetry)
        except Exception as exc:
            logger.warning("Skipping record %d: %s", idx, exc)

    logger.info("Loaded %d telemetry records from %s", len(records), path)
    return records


def parse_telemetry_payload(payload: dict[str, Any]) -> PatientTelemetry:
    """Parse and validate a single telemetry JSON payload.

    Args:
        payload: Raw dict (e.g. from an API request body).

    Returns:
        Validated :class:`PatientTelemetry`.

    Raises:
        ValueError: If the payload is invalid.
    """
    telemetry = PatientTelemetry(**payload)
    warnings = validate_telemetry(telemetry)
    if warnings:
        logger.info("Payload warnings: %s", warnings)
    return telemetry


def normalise_reading(reading: VitalReading) -> float:
    """Normalise a vital reading to a 0-1 scale based on its normal range.

    Values within the normal range map to 0.0-0.5; values outside scale
    linearly beyond 0.5 up to 1.0 based on how far they deviate.

    Args:
        reading: A single vital reading.

    Returns:
        Normalised score between 0.0 and 1.0.
    """
    low, high = VITAL_RANGES.get(reading.vital_type, (0.0, 100.0))
    mid = (low + high) / 2.0
    half_range = (high - low) / 2.0

    if half_range == 0:
        return 0.0

    deviation = abs(reading.value - mid) / half_range

    # Within normal range → 0.0–0.5
    if low <= reading.value <= high:
        return min(deviation * 0.5, 0.5)

    # Outside normal range → 0.5–1.0
    return min(0.5 + (deviation - 1.0) * 0.5, 1.0)


def normalise_telemetry(telemetry: PatientTelemetry) -> dict[VitalType, float]:
    """Normalise all readings in a telemetry record.

    Args:
        telemetry: Patient telemetry payload.

    Returns:
        Mapping of vital type to normalised score (0-1).
    """
    return {r.vital_type: normalise_reading(r) for r in telemetry.readings}
