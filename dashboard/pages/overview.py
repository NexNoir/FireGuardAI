from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from database.db import FireGuardDatabase
from sensors.live_sensor import LiveSensorService

from ..data import (
    predictions,
    fire_events,
    alerts,
)
from ..components import probability, confidence


def _read_real_sensor():
    """
    فقط داده واقعی ESP32 را می‌خواند.

    اگر ESP32 قطع باشد:
    - داده جدید ساخته نمی‌شود
    - داده قدیمی LIVE نمایش داده نمی‌شود
    """
    try:
        db = FireGuardDatabase()
        service = LiveSensorService(database=db)

        return service.read_and_store()

    except Exception as exc:
        return {
            "is_valid": False,
            "is_live": False,
            "is_stale": True,
            "stored": False,
            "errors": [str(exc)],
            "source": "esp32",
        }


def _safe_metric_number(value, digits=1, suffix=""):
    try:
        value = float(value)

        if pd.isna(value):
            return "N/A"

        return f"{value:.{digits}f}{suffix}"

    except Exception:
        return "N/A"


def _safe_flame(value):
    try:
        return int(float(value)) == 1
    except Exception:
        return False


def _sensor_age_seconds(timestamp):
    if timestamp is None:
        return None

    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        return max(
            0.0,
            (now - timestamp.astimezone(timezone.utc)).total_seconds(),
        )

    except Exception:
        return None


def render():
    st.title("🔥 FireGuard Overview")
    st.caption(
        "Real Sensor • Real External Data • Explicit Failure States"
    )

    # ============================================================
    # REAL ESP32 READ
    # ============================================================

    sensor = _read_real_sensor()

    valid = sensor.get("is_valid", False)
    live = sensor.get("is_live", False)
    stale = sensor.get("is_stale", True)

    # ============================================================
    # SYSTEM STATUS
    # ============================================================

    st.subheader("System Status")

    s1, s2, s3, s4 = st.columns(4)

    if valid and live and not stale:
        s1.success("🟢 SENSOR LIVE")
    elif stale:
        s1.warning("🟡 STALE DATA")
    else:
        s1.error("🔴 SENSOR UNAVAILABLE")

    source = sensor.get("source", "unknown")
    s2.metric("Sensor Source", str(source).upper())

    if valid and live:
        s3.metric(
            "Database",
            "STORED"
            if sensor.get("stored", False)
            else "NOT STORED",
        )
    else:
        s3.metric("Database", "NO NEW DATA")

    timestamp = sensor.get("timestamp")
    age = _sensor_age_seconds(timestamp)

    s4.metric(
        "Data Age",
        f"{age:.0f} sec"
        if age is not None
        else "UNKNOWN",
    )

    st.divider()

    # ============================================================
    # LIVE SENSOR
    # ============================================================

    st.subheader("📡 Live Sensor")

    if not valid or not live or stale:

        if stale:
            st.warning("⚠️ STALE DATA")
        else:
            st.error("🔴 SENSOR UNAVAILABLE")

        errors = sensor.get("errors", [])

        if errors:
            st.caption(errors[0])

        st.info(
            "Old database readings are not displayed here as LIVE data."
        )

    else:
        st.success("🟢 REAL ESP32 DATA")

        st.caption(
            f"Timestamp: {timestamp} | "
            f"DB Record ID: {sensor.get('database_id', 'N/A')}"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🌡 Temperature",
            _safe_metric_number(
                sensor.get("temperature"),
                digits=1,
                suffix=" °C",
            ),
        )

        c2.metric(
            "💧 Humidity",
            _safe_metric_number(
                sensor.get("humidity"),
                digits=1,
                suffix=" %",
            ),
        )

        c3.metric(
            "☁ Smoke",
            _safe_metric_number(
                sensor.get("smoke"),
                digits=0,
            ),
        )

        if _safe_flame(sensor.get("flame")):
            c4.error("🔥 FLAME DETECTED")
        else:
            c4.success("🛡 NO FLAME")

    st.divider()

    # ============================================================
    # LATEST PREDICTION
    # ============================================================

    st.subheader("🧠 Latest Prediction")

    pred_df = predictions()

    if pred_df.empty:
        st.warning("FORECAST UNAVAILABLE")
    else:
        pred = pred_df.iloc[0]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Probability",
            probability(pred.get("probability")),
        )

        p2.metric(
            "Confidence",
            confidence(pred.get("confidence")),
        )

        uncertainty = pred.get("uncertainty")

        p3.metric(
            "Uncertainty",
            (
                f"{float(uncertainty):.1%}"
                if pd.notna(uncertainty)
                else "NOT AVAILABLE"
            ),
        )

        horizon = pred.get("horizon")

        p4.metric(
            "Horizon",
            f"{horizon}h"
            if pd.notna(horizon)
            else "N/A",
        )

        st.caption(
            f"Model: {pred.get('model_version', 'N/A')} | "
            f"Features: {pred.get('feature_version', 'N/A')}"
        )

    st.divider()

    # ============================================================
    # EVENTS + ALERTS
    # ============================================================

    events_df = fire_events()
    alerts_df = alerts()

    e1, e2, e3 = st.columns(3)

    e1.metric(
        "Fire Events",
        len(events_df),
    )

    open_events = 0

    if not events_df.empty and "status" in events_df.columns:
        open_events = int(
            events_df["status"]
            .astype(str)
            .str.lower()
            .eq("open")
            .sum()
        )

    e2.metric(
        "Open Events",
        open_events,
    )

    e3.metric(
        "Alerts",
        len(alerts_df),
    )

    st.divider()

    # ============================================================
    # RAW LIVE RECORD
    # ============================================================

    with st.expander("🔍 Current Live Sensor Record"):

        if valid and live and not stale:
            st.json(
                {
                    key: str(value)
                    for key, value in sensor.items()
                    if key != "errors"
                }
            )
        else:
            st.write("No valid live sensor record.")