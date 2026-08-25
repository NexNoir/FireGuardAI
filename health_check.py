from pathlib import Path
from datetime import datetime, timezone
import shutil
import sqlite3

from production_config.settings import (
    PROJECT_ROOT,
    ENV_FILE,
    DATABASE_FILE,
    FIREGUARD_ENV,
    SMS_DRY_RUN,
    has_nasa_key,
    has_weather_key,
    has_sms_credentials,
    get_sms_status,
)


def print_result(name, status, details):
    icons = {
        "PASS": "PASS",
        "WARN": "WARN",
        "FAIL": "FAIL",
        "INFO": "INFO",
    }

    print(f"{icons.get(status, status):<5} | {name:<22} | {details}")


def check_env():
    if ENV_FILE.exists():
        return "PASS", ".env found"

    return "FAIL", ".env NOT found"


def check_database():
    if not DATABASE_FILE.exists():
        return "WARN", f"Database not found: {DATABASE_FILE}"

    try:
        conn = sqlite3.connect(str(DATABASE_FILE), timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        return "PASS", f"Available: {DATABASE_FILE.name}"

    except Exception as e:
        return "FAIL", f"Database unavailable: {type(e).__name__}: {e}"


def check_sensor_data():
    """
    فقط freshness را بررسی می‌کند.
    اگر داده‌ای وجود ندارد یا قدیمی است، آن را Live اعلام نمی‌کند.
    """

    candidates = [
        PROJECT_ROOT / "data" / "sensor_history.csv",
        PROJECT_ROOT / "data" / "sensor_readings.csv",
        PROJECT_ROOT / "fireguard_sensor_history_1500.csv",
    ]

    found = None

    for path in candidates:
        if path.exists():
            found = path
            break

    if found is None:
        return "WARN", "No sensor history file found"

    try:
        modified = datetime.fromtimestamp(
            found.stat().st_mtime,
            tz=timezone.utc,
        )

        now = datetime.now(timezone.utc)
        age_minutes = (now - modified).total_seconds() / 60

        if age_minutes <= 5:
            return "PASS", f"LIVE candidate, file age: {age_minutes:.1f} min"

        return "WARN", (
            f"STALE DATA — file age: {age_minutes:.1f} min "
            f"({found.name})"
        )

    except Exception as e:
        return "FAIL", f"Sensor freshness check failed: {type(e).__name__}: {e}"


def check_models():
    candidates = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "saved_models",
    ]

    model_files = []

    for folder in candidates:
        if folder.exists():
            model_files.extend(
                list(folder.glob("*.joblib"))
                + list(folder.glob("*.pkl"))
                + list(folder.glob("*.pickle"))
            )

    if not model_files:
        return "WARN", "No recognized model file found"

    newest = max(model_files, key=lambda p: p.stat().st_mtime)

    return "PASS", f"Model file found: {newest.name}"


def check_nasa():
    if has_nasa_key():
        return "PASS", "NASA API key configured"

    return "WARN", "NASA API key unavailable — external NASA calls may be unavailable"


def check_weather():
    if has_weather_key():
        return "PASS", "Weather API key configured"

    return "WARN", "Weather API key unavailable — fake weather must NOT be generated"


def check_sms():
    status = get_sms_status()

    if status == "DRY_RUN":
        return "PASS", (
            "DRY_RUN ENABLED — real SMS blocked"
        )

    if status == "READY":
        return "WARN", (
            "Real SMS credentials configured and DRY_RUN is disabled"
        )

    if status == "UNAVAILABLE":
        return "PASS", (
            "SMS unavailable — no fake success and no crash"
        )

    return "WARN", f"Unknown SMS state: {status}"


def check_disk():
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)

        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        free_percent = usage.free / usage.total * 100

        if free_percent < 5:
            return "FAIL", (
                f"Low disk space: {free_gb:.2f} GB free "
                f"of {total_gb:.2f} GB"
            )

        if free_percent < 15:
            return "WARN", (
                f"Disk space getting low: {free_gb:.2f} GB free"
            )

        return "PASS", (
            f"{free_gb:.2f} GB free "
            f"({free_percent:.1f}% available)"
        )

    except Exception as e:
        return "FAIL", f"Disk check failed: {type(e).__name__}: {e}"


def main():
    print("=" * 70)
    print("FireGuard — Stage 14 Production Health Check")
    print("=" * 70)
    print(f"Environment : {FIREGUARD_ENV}")
    print(f"Project     : {PROJECT_ROOT}")
    print(f"Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    checks = [
        ("Environment", check_env),
        ("Database", check_database),
        ("Sensor Freshness", check_sensor_data),
        ("Model", check_models),
        ("NASA", check_nasa),
        ("Weather", check_weather),
        ("SMS Safety", check_sms),
        ("Disk Space", check_disk),
    ]

    results = []

    for name, check_function in checks:
        try:
            status, details = check_function()
        except Exception as e:
            status = "FAIL"
            details = f"Health check crashed safely: {type(e).__name__}: {e}"

        results.append(status)
        print_result(name, status, details)

    print("-" * 70)

    passed = results.count("PASS")
    warned = results.count("WARN")
    failed = results.count("FAIL")

    print(
        f"SUMMARY | PASS={passed} | WARN={warned} | FAIL={failed}"
    )

    if failed > 0:
        print("STATUS: RED — Critical health checks failed")
        return 1

    if warned > 0:
        print("STATUS: YELLOW — System running with unavailable/degraded services")
        return 0

    print("STATUS: GREEN — All production health checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())