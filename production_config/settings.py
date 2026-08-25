import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# FireGuard — Secure Production Settings
# ============================================================

# مسیر اصلی پروژه
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# خواندن فایل .env
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def get_env(name: str, default=None):
    """Read a string value safely from environment variables."""
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    if value == "":
        return default

    return value


def get_bool(name: str, default=False):
    """Read a boolean value safely."""
    value = get_env(name)

    if value is None:
        return default

    return value.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ============================================================
# Environment
# ============================================================

FIREGUARD_ENV = get_env(
    "FIREGUARD_ENV",
    "development",
)


# ============================================================
# API Secrets
# ============================================================

NASA_FIRMS_API_KEY = get_env(
    "NASA_FIRMS_API_KEY",
    None,
)

WEATHER_API_KEY = get_env(
    "WEATHER_API_KEY",
    None,
)

KAVENEGAR_API_KEY = get_env(
    "KAVENEGAR_API_KEY",
    None,
)

SMS_SENDER = get_env(
    "SMS_SENDER",
    None,
)

SMS_RECEIVER = get_env(
    "SMS_RECEIVER",
    None,
)


# ============================================================
# SMS Safety
# ============================================================

SMS_DRY_RUN = get_bool(
    "SMS_DRY_RUN",
    True,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

DATABASE_FILE = (
    DATA_DIR / "fireguard_history.db"
)


# ============================================================
# Helper functions
# ============================================================

def has_nasa_key():
    return bool(NASA_FIRMS_API_KEY)


def has_weather_key():
    return bool(WEATHER_API_KEY)


def has_sms_credentials():
    return bool(
        KAVENEGAR_API_KEY
        and SMS_SENDER
        and SMS_RECEIVER
    )


def get_sms_status():
    """
    Returns the real SMS safety state.
    Never reports SMS as ready unless all credentials exist.
    """

    if SMS_DRY_RUN:
        return "DRY_RUN"

    if not has_sms_credentials():
        return "UNAVAILABLE"

    return "READY"