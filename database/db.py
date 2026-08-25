from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .schema import SCHEMA_SQL, TABLES


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_FILE = DATA_DIR / "fireguard_history.db"


class FireGuardDatabase:

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_FILE
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(self):

        with self.connect() as conn:

            conn.executescript(SCHEMA_SQL)

            # Compatibility for an older database
            self._ensure_column(
                conn,
                "model",
                "experiment",
                "TEXT",
            )

            self._ensure_column(
                conn,
                "model",
                "notes",
                "TEXT",
            )

    @staticmethod
    def _ensure_column(
        conn,
        table,
        column,
        column_type,
    ):

        rows = conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()

        existing = {
            row["name"]
            for row in rows
        }

        if column not in existing:

            conn.execute(
                f"""
                ALTER TABLE {table}
                ADD COLUMN {column} {column_type}
                """
            )

    # =========================================================
    # SENSOR READING
    # =========================================================

    def add_sensor_reading(
        self,
        timestamp,
        temperature=None,
        humidity=None,
        smoke=None,
        flame=None,
        source=None,
    ):

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO sensor_reading
                (
                    timestamp,
                    temperature,
                    humidity,
                    smoke,
                    flame,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    temperature,
                    humidity,
                    smoke,
                    flame,
                    source,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # PREDICTION
    # =========================================================

    def add_prediction(
        self,
        timestamp,
        model_version,
        feature_version,
        probability,
        uncertainty,
        horizon,
    ):

        probability = float(probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between 0 and 1"
            )

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO prediction
                (
                    timestamp,
                    model_version,
                    feature_version,
                    probability,
                    uncertainty,
                    horizon
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    model_version,
                    feature_version,
                    probability,
                    uncertainty,
                    horizon,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # FIRE EVENT
    # =========================================================

    def add_fire_event(
        self,
        event_id,
        timestamp,
        event_type=None,
        status="open",
        description=None,
    ):

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO fire_event
                (
                    event_id,
                    timestamp,
                    event_type,
                    status,
                    description
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    timestamp,
                    event_type,
                    status,
                    description,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # VERIFICATION
    # =========================================================

    def add_verification(
        self,
        event_id,
        label,
        verified_at,
        verified_by,
        source,
    ):

        allowed_labels = {
            "unverified",
            "confirmed_fire",
            "confirmed_no_fire",
        }

        if label not in allowed_labels:
            raise ValueError(
                f"Invalid label: {label}"
            )

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO verification
                (
                    event_id,
                    label,
                    verified_at,
                    verified_by,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    label,
                    verified_at,
                    verified_by,
                    source,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # ALERT
    # =========================================================

    def add_alert(
        self,
        event_id,
        alert_level,
        reason=None,
        status="active",
    ):

        allowed_levels = {
            "INFO",
            "WATCH",
            "WARNING",
            "HIGH",
            "CRITICAL",
        }

        if alert_level not in allowed_levels:
            raise ValueError(
                f"Invalid alert level: {alert_level}"
            )

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO alert
                (
                    event_id,
                    alert_level,
                    reason,
                    status
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    alert_level,
                    reason,
                    status,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # MODEL
    # =========================================================

    def add_model(
        self,
        model_version,
        model_name=None,
        experiment=None,
        feature_version=None,
        horizon=None,
        status=None,
        path=None,
        notes=None,
    ):

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO model
                (
                    model_version,
                    model_name,
                    experiment,
                    feature_version,
                    horizon,
                    status,
                    path,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_version,
                    model_name,
                    experiment,
                    feature_version,
                    horizon,
                    status,
                    path,
                    notes,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # TRAINING RUN
    # =========================================================

    def add_training_run(
        self,
        run_id,
        model_version=None,
        started_at=None,
        completed_at=None,
        dataset_version=None,
        samples=None,
        validation_score=None,
        status=None,
        notes=None,
    ):

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO training_run
                (
                    run_id,
                    model_version,
                    started_at,
                    completed_at,
                    dataset_version,
                    samples,
                    validation_score,
                    status,
                    notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    model_version,
                    started_at,
                    completed_at,
                    dataset_version,
                    samples,
                    validation_score,
                    status,
                    notes,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # EXTERNAL OBSERVATION
    # =========================================================

    def add_external_observation(
        self,
        timestamp,
        source,
        observation_type=None,
        value=None,
        confidence=None,
    ):

        with self.connect() as conn:

            cursor = conn.execute(
                """
                INSERT INTO external_observation
                (
                    timestamp,
                    source,
                    observation_type,
                    value,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    source,
                    observation_type,
                    value,
                    confidence,
                ),
            )

            return cursor.lastrowid

    # =========================================================
    # COUNT
    # =========================================================

    def count_records(self, table_name):

        if table_name not in TABLES:
            raise ValueError(
                f"Invalid table name: {table_name}"
            )

        with self.connect() as conn:

            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM {table_name}
                """
            ).fetchone()

            return int(row["count"])

    # =========================================================
    # RECENT RECORDS
    # =========================================================

    def get_recent(
        self,
        table_name,
        limit=10,
    ):

        if table_name not in TABLES:
            raise ValueError(
                f"Invalid table name: {table_name}"
            )

        limit = max(1, int(limit))

        with self.connect() as conn:

            return conn.execute(
                f"""
                SELECT *
                FROM {table_name}
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()


if __name__ == "__main__":

    db = FireGuardDatabase()

    print("=" * 60)
    print("🔥 FireGuard Database")
    print("=" * 60)
    print(f"Database: {db.db_path}")
    print("Database initialized successfully.")