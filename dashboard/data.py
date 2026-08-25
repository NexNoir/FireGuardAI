from pathlib import Path
import sqlite3

import pandas as pd

from sensors.live_sensor import LiveSensorService


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "fireguard_history.db"


# ============================================================
# DATABASE
# ============================================================

def connect():
    """
    اتصال امن به SQLite.

    اگر Database وجود نداشته باشد، None برمی‌گرداند
    و Dashboard نباید crash کند.
    """

    if not DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
        )

        conn.row_factory = sqlite3.Row

        return conn

    except Exception:
        return None


def read_table(
    table,
    limit=500,
):
    """
    خواندن یکی از جداول مجاز Database.
    """

    allowed = {
        "sensor_reading",
        "prediction",
        "fire_event",
        "verification",
        "alert",
        "model",
        "training_run",
        "external_observation",
    }

    if table not in allowed:
        return pd.DataFrame()

    conn = connect()

    if conn is None:
        return pd.DataFrame()

    try:

        return pd.read_sql_query(
            f"""
            SELECT *
            FROM {table}
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )

    except Exception:
        return pd.DataFrame()

    finally:
        conn.close()


# ============================================================
# LIVE SENSOR
# ============================================================

def read_live_sensor(
    database=None,
):
    """
    دریافت یک خوانش واقعی از ESP32.

    مسیر:

        ESP32
          ↓
        Esp32Reader
          ↓
        LiveSensorService
          ↓
        Validator
          ↓
        Dashboard

    هیچ داده جعلی تولید نمی‌شود.

    اگر سنسور قطع باشد:
        is_live = False
        is_stale = True

    و Dashboard نباید آخرین رکورد قدیمی را
    به عنوان داده زنده نمایش دهد.
    """

    try:

        service = LiveSensorService(
            database=database
        )

        result = service.read_and_store()

        if not isinstance(result, dict):
            return {
                "is_valid": False,
                "is_live": False,
                "is_stale": True,
                "source": "esp32",
                "errors": [
                    "invalid_sensor_response"
                ],
                "warnings": [],
            }

        return result

    except Exception as exc:

        return {
            "is_valid": False,
            "is_live": False,
            "is_stale": True,
            "source": "esp32",
            "errors": [
                f"live_sensor_error: {exc}"
            ],
            "warnings": [],
        }


# ============================================================
# LIVE SENSOR — DATABASE INTEGRATION
# ============================================================

def read_live_sensor_and_store():
    """
    خواندن سنسور واقعی و ذخیره خوانش معتبر در Database.

    این تابع برای Dashboard مناسب است.
    """

    try:

        # Lazy import برای جلوگیری از وابستگی سخت
        # در زمان import شدن Dashboard
        from database.db import FireGuardDatabase

        db = FireGuardDatabase()

        return read_live_sensor(
            database=db
        )

    except Exception as exc:

        # اگر Database در دسترس نبود،
        # تلاش می‌کنیم حداقل سنسور را بخوانیم.
        try:

            result = read_live_sensor(
                database=None
            )

            result["stored"] = False
            result["database_error"] = str(exc)

            return result

        except Exception as sensor_exc:

            return {
                "is_valid": False,
                "is_live": False,
                "is_stale": True,
                "source": "esp32",
                "errors": [
                    f"database_error: {exc}",
                    f"sensor_error: {sensor_exc}",
                ],
                "warnings": [],
            }


# ============================================================
# SENSOR HISTORY
# ============================================================

def sensors():
    """
    تاریخچه Database.

    توجه:
    این تابع داده زنده نمی‌خواند.
    فقط تاریخچه را برمی‌گرداند.
    """

    return read_table(
        "sensor_reading",
        1000,
    )


# ============================================================
# OTHER DATABASE TABLES
# ============================================================

def predictions():

    return read_table(
        "prediction",
        500,
    )


def fire_events():

    return read_table(
        "fire_event",
        500,
    )


def verifications():

    return read_table(
        "verification",
        500,
    )


def alerts():

    return read_table(
        "alert",
        500,
    )


def models():

    return read_table(
        "model",
        100,
    )


def training_runs():

    return read_table(
        "training_run",
        100,
    )


def external_observations():

    return read_table(
        "external_observation",
        500,
    )


# ============================================================
# DASHBOARD HELPERS
# ============================================================

def get_sensor_status():
    """
    وضعیت فعلی سنسور را برای Dashboard برمی‌گرداند.

    خروجی استاندارد:

        {
            "online": bool,
            "stale": bool,
            "valid": bool,
            "reading": dict
        }
    """

    reading = read_live_sensor_and_store()

    return {
        "online": bool(
            reading.get("is_live", False)
        ),
        "stale": bool(
            reading.get("is_stale", True)
        ),
        "valid": bool(
            reading.get("is_valid", False)
        ),
        "reading": reading,
    }


def get_live_sensor_values():
    """
    فقط در صورت دریافت داده معتبر و زنده،
    مقادیر سنسور را برمی‌گرداند.

    اگر سنسور offline/stale باشد،
    None برمی‌گرداند.

    این تابع عمداً اجازه نمی‌دهد
    آخرین داده Database به عنوان Live نمایش داده شود.
    """

    result = read_live_sensor_and_store()

    if not result.get("is_valid"):
        return None

    if not result.get("is_live"):
        return None

    if result.get("is_stale"):
        return None

    return {
        "temperature": result.get(
            "temperature"
        ),
        "humidity": result.get(
            "humidity"
        ),
        "smoke": result.get(
            "smoke"
        ),
        "flame": result.get(
            "flame"
        ),
        "brightness": result.get(
            "brightness"
        ),
        "wind": result.get(
            "wind"
        ),
        "timestamp": result.get(
            "timestamp"
        ),
        "source": result.get(
            "source"
        ),
        "source_url": result.get(
            "source_url"
        ),
    }