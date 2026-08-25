"""
FireGuard - Safe import of the 1500-record sensor history.

Run from the project root:
    python import_sensor_history.py

This script:
1) Creates/repairs sensor_readings without deleting existing data.
2) Adds the required columns if they are missing.
3) Imports fireguard_sensor_history_1500.csv.
4) Prevents duplicate imports using timestamp + sensor values.
"""

import sqlite3
from pathlib import Path
import csv

import config


REQUIRED_COLUMNS = {
    "timestamp": "TEXT",
    "temperature": "REAL",
    "humidity": "REAL",
    "smoke": "INTEGER",
    "flame": "INTEGER",
    "label": "TEXT",
    "notes": "TEXT",
    "created_at": "TEXT",
}


def get_connection():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn):
    # Create the complete table if it does not exist.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            humidity REAL,
            smoke INTEGER,
            flame INTEGER,
            label TEXT DEFAULT 'unverified',
            notes TEXT,
            created_at TEXT
        )
    """)

    columns = {
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(sensor_readings)"
        ).fetchall()
    }

    for column, data_type in REQUIRED_COLUMNS.items():
        if column not in columns:
            conn.execute(
                f"ALTER TABLE sensor_readings ADD COLUMN {column} {data_type}"
            )
            print(f"  + ستون اضافه شد: {column}")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON sensor_readings(timestamp)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_label
        ON sensor_readings(label)
    """)


def import_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"فایل پیدا نشد: {csv_path}\n"
            "فایل fireguard_sensor_history_1500.csv را کنار kj.py قرار بده."
        )

    with get_connection() as conn:
        ensure_schema(conn)

        before = conn.execute(
            "SELECT COUNT(*) FROM sensor_readings"
        ).fetchone()[0]

        inserted = 0
        skipped = 0

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            required = {
                "timestamp",
                "temperature",
                "humidity",
                "smoke",
                "flame",
                "label",
                "notes",
                "created_at",
            }

            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"ستون‌های زیر در CSV وجود ندارند: {sorted(missing)}"
                )

            for row in reader:
                # Duplicate protection:
                # the imported dataset can be run more than once safely.
                exists = conn.execute("""
                    SELECT 1
                    FROM sensor_readings
                    WHERE timestamp = ?
                      AND temperature = ?
                      AND humidity = ?
                      AND smoke = ?
                      AND flame = ?
                    LIMIT 1
                """, (
                    row["timestamp"],
                    float(row["temperature"]),
                    float(row["humidity"]),
                    int(row["smoke"]),
                    int(row["flame"]),
                )).fetchone()

                if exists:
                    skipped += 1
                    continue

                conn.execute("""
                    INSERT INTO sensor_readings
                    (
                        timestamp,
                        temperature,
                        humidity,
                        smoke,
                        flame,
                        label,
                        notes,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["timestamp"],
                    float(row["temperature"]),
                    float(row["humidity"]),
                    int(row["smoke"]),
                    int(row["flame"]),
                    row["label"] or "unverified",
                    row["notes"] or "",
                    row["created_at"] or row["timestamp"],
                ))

                inserted += 1

        after = conn.execute(
            "SELECT COUNT(*) FROM sensor_readings"
        ).fetchone()[0]

    print("\n=== نتیجه Import ===")
    print(f"قبل از import : {before}")
    print(f"رکورد جدید    : {inserted}")
    print(f"تکراری/ردشده  : {skipped}")
    print(f"بعد از import  : {after}")
    print(f"Database       : {config.DB_PATH}")

    if inserted:
        print("\n✅ داده‌های 1500 رکوردی با موفقیت وارد شدند.")
    else:
        print("\nℹ️ رکورد جدیدی اضافه نشد؛ احتمالاً قبلاً import شده‌اند.")


if __name__ == "__main__":
    print("=== FireGuard: Safe Sensor History Import ===")
    print(f"Database: {config.DB_PATH}")

    # The converted CSV should be next to this script.
    csv_file = Path(__file__).resolve().parent / "fireguard_sensor_history_1500.csv"

    import_csv(csv_file)
