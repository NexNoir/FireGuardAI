from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = (
    BASE_DIR
    / "data"
    / "fireguard_history.db"
)


class AlertStore:

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        self.db_path = Path(
            db_path or DB_PATH
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._ensure_table()

    def _connect(self):
        return sqlite3.connect(
            str(self.db_path),
            timeout=10,
        )

    def _ensure_table(self):

        with self._connect() as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sms_alert_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_sms_at TEXT,
                    last_status TEXT,
                    last_risk_level TEXT,
                    last_probability REAL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.commit()

    @staticmethod
    def _utc(
        timestamp: datetime,
    ) -> datetime:

        if timestamp.tzinfo is None:
            return timestamp.replace(
                tzinfo=timezone.utc
            )

        return timestamp.astimezone(
            timezone.utc
        )

    def get_last_sms_at(
        self,
    ) -> Optional[datetime]:

        with self._connect() as conn:

            row = conn.execute(
                """
                SELECT last_sms_at
                FROM sms_alert_state
                WHERE id = 1
                """
            ).fetchone()

        if not row or not row[0]:
            return None

        try:

            value = row[0]

            if value.endswith("Z"):
                value = (
                    value[:-1]
                    + "+00:00"
                )

            result = datetime.fromisoformat(
                value
            )

            if result.tzinfo is None:
                result = result.replace(
                    tzinfo=timezone.utc
                )

            return result.astimezone(
                timezone.utc
            )

        except Exception:
            return None

    def save_sms_success(
        self,
        timestamp: datetime,
        risk_level: str,
        probability: float,
    ):

        timestamp = self._utc(
            timestamp
        )

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO sms_alert_state
                (
                    id,
                    last_sms_at,
                    last_status,
                    last_risk_level,
                    last_probability,
                    updated_at
                )
                VALUES
                (
                    1, ?, ?, ?, ?, ?
                )
                ON CONFLICT(id)
                DO UPDATE SET
                    last_sms_at =
                        excluded.last_sms_at,
                    last_status =
                        excluded.last_status,
                    last_risk_level =
                        excluded.last_risk_level,
                    last_probability =
                        excluded.last_probability,
                    updated_at =
                        excluded.updated_at
                """,
                (
                    timestamp.isoformat(),
                    "SMS_SENT",
                    risk_level,
                    float(probability),
                    timestamp.isoformat(),
                ),
            )

            conn.commit()

    def save_sms_failure(
        self,
        timestamp: datetime,
        risk_level: str,
        probability: float,
    ):

        timestamp = self._utc(
            timestamp
        )

        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO sms_alert_state
                (
                    id,
                    last_sms_at,
                    last_status,
                    last_risk_level,
                    last_probability,
                    updated_at
                )
                VALUES
                (
                    1, NULL, ?, ?, ?, ?
                )
                ON CONFLICT(id)
                DO UPDATE SET
                    last_status =
                        excluded.last_status,
                    last_risk_level =
                        excluded.last_risk_level,
                    last_probability =
                        excluded.last_probability,
                    updated_at =
                        excluded.updated_at
                """,
                (
                    "SMS_FAILED",
                    risk_level,
                    float(probability),
                    timestamp.isoformat(),
                ),
            )

            conn.commit()