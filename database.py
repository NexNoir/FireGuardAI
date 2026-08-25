import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime
import config


def get_connection():
    """اتصال به دیتابیس"""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    with get_connection() as conn:

        conn.executescript("""
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
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON sensor_readings(timestamp);

            CREATE INDEX IF NOT EXISTS idx_label
            ON sensor_readings(label);
        """)

    print("✅ دیتابیس آماده شد.")

def rebuild_database():
    """
    بررسی و اصلاح ساختار جدول sensor_readings
    بدون حذف اطلاعات موجود.
    """

    with get_connection() as conn:

        # اگر جدول اصلاً وجود ندارد، ایجادش کن
        table_exists = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name='sensor_readings'
        """).fetchone()

        if not table_exists:
            conn.executescript("""
                CREATE TABLE sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    temperature REAL,
                    humidity REAL,
                    smoke INTEGER,
                    flame INTEGER,
                    label TEXT DEFAULT 'unverified',
                    notes TEXT,
                    created_at TEXT
                );

                CREATE INDEX idx_timestamp
                ON sensor_readings(timestamp);

                CREATE INDEX idx_label
                ON sensor_readings(label);
            """)

            print("✅ جدول sensor_readings ایجاد شد.")
            return

        # ساختار فعلی جدول
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(sensor_readings)"
            ).fetchall()
        }

        required_columns = {
            "timestamp": "TEXT",
            "temperature": "REAL",
            "humidity": "REAL",
            "smoke": "INTEGER",
            "flame": "INTEGER",
            "label": "TEXT",
            "notes": "TEXT",
            "created_at": "TEXT"
        }

        # اضافه کردن ستون‌های مفقود
        for column, data_type in required_columns.items():

            if column not in columns:

                conn.execute(
                    f"""
                    ALTER TABLE sensor_readings
                    ADD COLUMN {column} {data_type}
                    """
                )

                print(f"➕ ستون {column} اضافه شد.")

        # مقدار پیش‌فرض label برای رکوردهای قبلی
        conn.execute("""
            UPDATE sensor_readings
            SET label = 'unverified'
            WHERE label IS NULL
        """)

        # created_at برای رکوردهای قبلی
        conn.execute("""
            UPDATE sensor_readings
            SET created_at = timestamp
            WHERE created_at IS NULL
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON sensor_readings(timestamp)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_label
            ON sensor_readings(label)
        """)

    print("✅ ساختار دیتابیس بررسی و اصلاح شد.")

def insert_reading(
    timestamp: str,
    temperature: float,
    humidity: float,
    smoke: int,
    flame: int,
    label: str = "unverified"
) -> int:

    if label not in (
        "confirmed_fire",
        "confirmed_no_fire",
        "unverified"
    ):
        raise ValueError("برچسب نامعتبر")

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:

        cur = conn.execute("""
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
            timestamp,
            temperature,
            humidity,
            smoke,
            flame,
            label,
            "",
            created_at
        ))

        return cur.lastrowid


def get_recent_readings(limit: int = 50):

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return [dict(r) for r in rows]


def update_label(
    row_id: int,
    label: str,
    notes: str = ""
):

    if label not in (
        "confirmed_fire",
        "confirmed_no_fire",
        "unverified"
    ):
        raise ValueError("برچسب نامعتبر")

    with get_connection() as conn:

        conn.execute("""
            UPDATE sensor_readings
            SET label = ?,
                notes = ?
            WHERE id = ?
        """, (
            label,
            notes,
            row_id
        ))


def count_by_label():

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT
                label,
                COUNT(*) AS cnt
            FROM sensor_readings
            GROUP BY label
        """).fetchall()

    return {
        r["label"]: r["cnt"]
        for r in rows
    }


def get_all_labeled():

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT *
            FROM sensor_readings
            WHERE label IN (
                'confirmed_fire',
                'confirmed_no_fire'
            )
            ORDER BY id
        """).fetchall()

    return [dict(r) for r in rows]


def save_date_normalized(df: pd.DataFrame):

    if df.empty:
        print("⚠️ دیتایی برای ذخیره وجود ندارد.")
        return

    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError(
            "ستون timestamp در داده‌ها وجود ندارد."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    ).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        print("⚠️ هیچ timestamp معتبری پیدا نشد.")
        return

    # ستون‌های ضروری
    if "label" not in df.columns:
        df["label"] = "unverified"

    if "notes" not in df.columns:
        df["notes"] = ""

    df["created_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    columns = [
        "timestamp",
        "temperature",
        "humidity",
        "smoke",
        "flame",
        "label",
        "notes",
        "created_at"
    ]

    # فقط ستون‌های موجود
    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    with get_connection() as conn:

        df[columns].to_sql(
            "sensor_readings",
            conn,
            if_exists="append",
            index=False
        )

    print(
        f"✅ داده‌های تاریخ‌دار ذخیره شدند! "
        f"(تعداد: {len(df)})"
    )