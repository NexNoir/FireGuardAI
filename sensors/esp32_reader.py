from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import os
import requests


class SensorReadError(RuntimeError):
    """خطای کنترل‌شده در خواندن سنسور."""


class Esp32Reader:
    """
    Reader واقعی ESP32.

    هیچ داده‌ای تولید نمی‌کند.
    فقط داده‌ای را که ESP32 از /sensor برمی‌گرداند دریافت می‌کند.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        timeout_seconds: float = 5.0,
        stale_after_seconds: float = 15.0,
    ):
        self.url = (
            url
            or os.getenv("ESP32_URL")
            or "http://10.138.144.37/sensor"
        )

        self.timeout_seconds = float(timeout_seconds)
        self.stale_after_seconds = float(stale_after_seconds)

    def read(self) -> Dict[str, Any]:
        """
        دریافت یک خوانش واقعی از ESP32.

        در صورت خطا SensorReadError ایجاد می‌شود.
        """

        try:
            response = requests.get(
                self.url,
                timeout=self.timeout_seconds,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            raise SensorReadError(
                f"ESP32 unavailable: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise SensorReadError(
                "ESP32 returned invalid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise SensorReadError(
                "ESP32 response must be a JSON object"
            )

        # فقط فیلدهایی که واقعاً از ESP32 آمده‌اند.
        required = (
            "temp",
            "humidity",
            "smoke",
            "flame",
        )

        missing = [
            key for key in required
            if key not in payload
        ]

        if missing:
            raise SensorReadError(
                "ESP32 response missing fields: "
                + ", ".join(missing)
            )

        received_at = datetime.now(timezone.utc)

        result = {
            "temperature": payload["temp"],
            "humidity": payload["humidity"],
            "smoke": payload["smoke"],
            "flame": payload["flame"],
            "brightness": payload.get("brightness"),
            "wind": payload.get("wind"),

            # زمان دریافت واقعی توسط FireGuard
            "timestamp": received_at,

            # وضعیت freshness
            "is_live": True,
            "is_stale": False,

            # منبع داده
            "source": "esp32",

            # URL فقط برای traceability
            "source_url": self.url,
        }

        return result

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        سازگاری با کدهای قدیمی پروژه.
        این متد هیچ داده‌ای تولید یا تغییر نمی‌دهد.
        """
        if not isinstance(data, dict):
            raise SensorReadError(
                "Sensor data must be a dictionary"
            )

        return data