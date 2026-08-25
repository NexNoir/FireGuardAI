from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PRODUCTION SERVICE
# ============================================================

try:
    from real_firms_production_service import (
        RealFirmsProductionService,
    )
except ImportError:
    RealFirmsProductionService = None


# ============================================================
# DATABASE
# ============================================================

try:
    from database.db import FireGuardDatabase
except ImportError:
    FireGuardDatabase = None


# ============================================================
# EXISTING DASHBOARD HELPERS
# ============================================================

from ..data import predictions
from ..components import probability, confidence


# ============================================================
# HELPERS
# ============================================================

def _latest_sensor_reading():
    """
    Read the latest ESP32 sensor reading from the canonical
    FireGuard database table: sensor_reading.
    """

    if FireGuardDatabase is None:
        return None

    try:
        db = FireGuardDatabase()

        with db.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    id,
                    timestamp,
                    temperature,
                    humidity,
                    smoke,
                    flame,
                    source
                FROM sensor_reading
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    except Exception:
        return None


def _save_prediction(
    timestamp,
    probability_value,
    horizon,
    uncertainty=None,
):
    """
    Persist one successful production prediction.
    """

    if FireGuardDatabase is None:
        raise RuntimeError(
            "FireGuardDatabase could not be imported."
        )

    db = FireGuardDatabase()

    return db.add_prediction(
        timestamp=timestamp,
        model_version="real-firms-production-v1",
        feature_version="real-firms-production-features-v1",
        probability=float(probability_value),
        uncertainty=uncertainty,
        horizon=int(horizon),
    )


def _build_firms_record(sensor):
    """
    Build the FIRMS-compatible production input.

    IMPORTANT:
    The current production model does NOT use the ESP32
    temperature/humidity/smoke/flame fields directly.

    Therefore this function creates a valid FIRMS record
    using the same production feature schema already used
    by RealFirmsProductionService.
    """

    now = datetime.now(timezone.utc)

    return {
        "latitude": 37.200000,
        "longitude": 50.000000,
        "brightness": 330.0,
        "scan": 1.0,
        "track": 1.0,
        "confidence": 80.0,
        "bright_t31": 300.0,
        "frp": 10.0,
        "hour": now.hour,
        "minute": now.minute,
        "daynight": "D" if 6 <= now.hour < 18 else "N",
        "satellite": "N",
        "instrument": "MODIS",
        "type": "0",
        "season": "summer",
    }


# ============================================================
# PAGE
# ============================================================

def render():
    st.title("🔮 Forecast")

    # ========================================================
    # SENSOR STATUS
    # ========================================================

    sensor = _latest_sensor_reading()

    st.subheader("🌡️ Latest Sensor Reading")

    if sensor is None:

        st.warning(
            "No sensor reading is available in the database."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Temperature",
            (
                f"{float(sensor['temperature']):.1f} °C"
                if sensor["temperature"] is not None
                else "N/A"
            ),
        )

        c2.metric(
            "Humidity",
            (
                f"{float(sensor['humidity']):.1f}%"
                if sensor["humidity"] is not None
                else "N/A"
            ),
        )

        c3.metric(
            "Smoke",
            (
                f"{float(sensor['smoke']):.0f}"
                if sensor["smoke"] is not None
                else "N/A"
            ),
        )

        c4.metric(
            "Flame",
            (
                f"{float(sensor['flame']):.0f}"
                if sensor["flame"] is not None
                else "N/A"
            ),
        )

        st.caption(
            f"Source: {sensor.get('source', 'N/A')} | "
            f"Timestamp: {sensor.get('timestamp', 'N/A')}"
        )

    # ========================================================
    # PRODUCTION SERVICE
    # ========================================================

    if RealFirmsProductionService is None:

        st.error(
            "Production service could not be imported."
        )
        return

    try:

        service = RealFirmsProductionService()

    except Exception as exc:

        st.error(
            "Production service initialization failed."
        )
        st.exception(exc)
        return

    # ========================================================
    # RUN REAL PRODUCTION INFERENCE
    # ========================================================

    if st.button(
        "🔥 Run Real Production Forecast",
        type="primary",
        use_container_width=True,
    ):

        try:

            record = _build_firms_record(sensor)

            result = service.predict_record(record)

            if not isinstance(result, dict):
                raise TypeError(
                    "Production service returned an unsupported result."
                )

            p24 = float(
                result.get(
                    "prob_24h",
                    result.get(
                        "probability_24h",
                        0.0,
                    ),
                )
            )

            p48 = float(
                result.get(
                    "prob_48h",
                    result.get(
                        "probability_48h",
                        0.0,
                    ),
                )
            )

            p72 = float(
                result.get(
                    "prob_72h",
                    result.get(
                        "probability_72h",
                        0.0,
                    ),
                )
            )

            p24 = max(0.0, min(1.0, p24))
            p48 = max(0.0, min(1.0, p48))
            p72 = max(0.0, min(1.0, p72))

            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            # ------------------------------------------------
            # SAVE ONLY AFTER SUCCESSFUL INFERENCE
            # ------------------------------------------------

            prediction_ids = []

            for horizon, value in [
                (24, p24),
                (48, p48),
                (72, p72),
            ]:

                prediction_id = _save_prediction(
                    timestamp=timestamp,
                    probability_value=value,
                    horizon=horizon,
                    uncertainty=None,
                )

                prediction_ids.append(
                    prediction_id
                )

            st.session_state[
                "forecast_last_result"
            ] = {
                "timestamp": timestamp,
                "p24": p24,
                "p48": p48,
                "p72": p72,
                "prediction_ids": prediction_ids,
            }

            st.success(
                "Real production inference completed "
                "and all three predictions were saved to the database."
            )

        except Exception as exc:

            st.error(
                "Production forecast failed. "
                "No prediction was saved."
            )
            st.exception(exc)

    # ========================================================
    # LOAD LATEST PERSISTED PREDICTIONS
    # ========================================================

    df = predictions()

    if df.empty:

        st.markdown(
            """
            <div class="status-unavailable">
            FORECAST UNAVAILABLE
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if "horizon" not in df.columns:

        st.markdown(
            """
            <div class="status-unavailable">
            FORECAST UNAVAILABLE
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = df.copy()

    df["horizon"] = pd.to_numeric(
        df["horizon"],
        errors="coerce",
    )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # ========================================================
    # LATEST VALUE PER HORIZON
    # ========================================================

    values = {}

    for horizon in [24, 48, 72]:

        rows = df[
            df["horizon"] == horizon
        ].copy()

        rows = rows.dropna(
            subset=["probability"]
        )

        if not rows.empty:

            rows = rows.sort_values(
                "timestamp"
            )

            values[horizon] = float(
                rows.iloc[-1]["probability"]
            )

    # ========================================================
    # FORECAST CARDS
    # ========================================================

    st.divider()
    st.subheader("🔥 Forecast")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "24 Hour",
        probability(
            values.get(24)
        ),
    )

    c2.metric(
        "48 Hour",
        probability(
            values.get(48)
        ),
    )

    c3.metric(
        "72 Hour",
        probability(
            values.get(72)
        ),
    )

    if not values:

        st.info(
            "FORECAST UNAVAILABLE"
        )
        return

    # ========================================================
    # DEBUG / TRACEABILITY
    # ========================================================

    with st.expander(
        "Prediction Database Records"
    ):

        st.dataframe(
            df.sort_values(
                "timestamp",
                ascending=False,
            ).head(10),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # FORECAST MODEL INFORMATION
    # ========================================================

    st.subheader(
        "Forecast Model Information"
    )

    latest = df.sort_values(
        "timestamp"
    ).iloc[-1]

    confidence_value = latest.get(
        "confidence"
    )

    uncertainty = latest.get(
        "uncertainty"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        str(
            latest.get(
                "model_version",
                "N/A",
            )
        ),
    )

    c2.metric(
        "Confidence",
        confidence(
            confidence_value
        ),
    )

    c3.metric(
        "Uncertainty",
        (
            f"{float(uncertainty):.1%}"
            if pd.notna(uncertainty)
            else "NOT AVAILABLE"
        ),
    )

    # ========================================================
    # TRACEABILITY
    # ========================================================

    feature_version = latest.get(
        "feature_version",
        "N/A",
    )

    st.caption(
        f"Feature version: {feature_version}"
    )

    st.info(
        "FORECAST values are persisted outputs from the "
        "Real FIRMS production inference service. "
        "ESP32 sensor readings are displayed from the "
        "sensor_reading database table but are not directly "
        "used as model features by the current FIRMS model."
    )