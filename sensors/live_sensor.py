from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .esp32_reader import Esp32Reader, SensorReadError
from .data_validator import validate_reading


class LiveSensorService:
    """
    مسیر واقعی سنسور:

    ESP32
       ↓
    Esp32Reader
       ↓
    Validator
       ↓
    Database

    داده جعلی تولید نمی‌شود.
    داده stale به عنوان live ثبت نمی‌شود.
    """

    def __init__(
        self,
        reader: Esp32Reader | None = None,
        database=None,
    ):
        self.reader = reader or Esp32Reader()
        self.database = database

    def read(self) -> Dict[str, Any]:
        """
        یک خوانش واقعی ESP32 را دریافت و اعتبارسنجی می‌کند.
        """

        try:
            raw = self.reader.read()

        except SensorReadError as exc:
            return {
                "is_valid": False,
                "is_live": False,
                "is_stale": True,
                "source": "esp32",
                "errors": [str(exc)],
                "warnings": [],
                "timestamp": datetime.now(timezone.utc),
            }

        validated = validate_reading(raw)

        validated["is_live"] = bool(
            validated.get("is_valid")
        )

        validated["is_stale"] = not validated["is_live"]

        validated["source"] = "esp32"
        validated["source_url"] = raw.get("source_url")

        return validated

    def read_and_store(self) -> Dict[str, Any]:
        """
        خواندن سنسور واقعی و در صورت معتبر بودن،
        ثبت آن در Database.

        داده نامعتبر در Database به عنوان reading معتبر
        ذخیره نمی‌شود.
        """

        reading = self.read()

        if not reading.get("is_valid"):
            return {
                **reading,
                "stored": False,
                "database_error": None,
            }

        if self.database is None:
            return {
                **reading,
                "stored": False,
                "database_error": "database_not_configured",
            }

        try:
            record_id = self.database.add_sensor_reading(
                timestamp=reading["timestamp"],
                temperature=reading["temperature"],
                humidity=reading["humidity"],
                smoke=reading["smoke"],
                flame=reading["flame"],
                source=reading["source"],
            )

            return {
                **reading,
                "stored": True,
                "database_id": record_id,
                "database_error": None,
            }

        except Exception as exc:
            # قطع یا خطای Database نباید باعث crash سیستم شود.
            return {
                **reading,
                "stored": False,
                "database_id": None,
                "database_error": str(exc),
            }