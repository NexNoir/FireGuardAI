from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import config


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if result != result:  # NaN
            return None

        return result
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_reading(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    اعتبارسنجی داده واقعی سنسور.

    این تابع هیچ مقدار سنسوری تولید نمی‌کند.
    داده نامعتبر را معتبر اعلام نمی‌کند.
    """

    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "temperature": None,
        "humidity": None,
        "smoke": None,
        "flame": None,
        "brightness": None,
        "wind": None,
        "timestamp": None,
        "source": raw.get("source", "unknown"),
    }

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    ts = raw.get("timestamp") or raw.get("time")

    if ts is None:
        result["is_valid"] = False
        result["errors"].append("timestamp_missing")

    elif isinstance(ts, datetime):
        result["timestamp"] = ts

    elif isinstance(ts, str):
        try:
            result["timestamp"] = datetime.fromisoformat(ts)
        except ValueError:
            result["is_valid"] = False
            result["errors"].append("timestamp_invalid_format")

    else:
        result["is_valid"] = False
        result["errors"].append("timestamp_invalid_type")

    # ---------------------------------------------------------
    # Temperature
    # ---------------------------------------------------------

    temp = _safe_float(
        raw.get("temperature", raw.get("temp"))
    )

    if temp is None:
        result["is_valid"] = False
        result["errors"].append("temperature_missing_or_invalid")
    else:
        result["temperature"] = temp

        if (
            temp < config.TEMP_IMPOSSIBLE_LOW
            or temp > config.TEMP_IMPOSSIBLE_HIGH
        ):
            result["is_valid"] = False
            result["errors"].append("temperature_impossible")

        elif temp < config.TEMP_MIN or temp > config.TEMP_MAX:
            result["warnings"].append(
                "temperature_out_of_normal_range"
            )

    # ---------------------------------------------------------
    # Humidity
    # ---------------------------------------------------------

    humidity = _safe_float(raw.get("humidity"))

    if humidity is None:
        result["is_valid"] = False
        result["errors"].append("humidity_missing_or_invalid")

    else:
        result["humidity"] = humidity

        if (
            humidity < config.HUMIDITY_MIN
            or humidity > config.HUMIDITY_MAX
        ):
            result["is_valid"] = False
            result["errors"].append("humidity_impossible")

    # ---------------------------------------------------------
    # Smoke
    # ---------------------------------------------------------

    smoke = _safe_int(raw.get("smoke"))

    if smoke is None:
        result["is_valid"] = False
        result["errors"].append("smoke_missing_or_invalid")

    else:
        result["smoke"] = smoke

        if (
            smoke < config.SMOKE_MIN
            or smoke > config.SMOKE_IMPOSSIBLE_HIGH
        ):
            result["is_valid"] = False
            result["errors"].append("smoke_impossible")

        elif smoke > config.SMOKE_MAX:
            result["warnings"].append(
                "smoke_above_typical_adc"
            )

    # ---------------------------------------------------------
    # Flame
    # ---------------------------------------------------------

    flame = _safe_int(raw.get("flame"))

    if flame is None:
        result["is_valid"] = False
        result["errors"].append("flame_missing_or_invalid")

    elif flame not in (0, 1):
        result["is_valid"] = False
        result["errors"].append("flame_invalid_value")

    else:
        result["flame"] = flame

    # ---------------------------------------------------------
    # Optional sensors
    # ---------------------------------------------------------

    result["brightness"] = _safe_float(
        raw.get("brightness")
    )

    result["wind"] = _safe_float(
        raw.get("wind")
    )

    return result


def detect_stuck(
    history: List[Dict[str, Any]],
    key: str,
    threshold: int | None = None,
) -> bool:
    """
    تشخیص stuck sensor.
    اگر مقدار یک سنسور در چند خوانش متوالی دقیقاً یکسان باشد.
    """

    if threshold is None:
        threshold = config.STUCK_THRESHOLD

    if len(history) < threshold:
        return False

    recent = [
        item.get(key)
        for item in history[-threshold:]
    ]

    if any(value is None for value in recent):
        return False

    return len(set(recent)) == 1