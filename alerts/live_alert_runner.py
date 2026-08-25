from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data import sensors, predictions
from ..components import (
    freshness,
    probability,
    confidence,
)

from alerts.live_alert_runner import LiveAlertRunner


def safe_flame(value):
    try:
        return int(float(value)) == 1
    except Exception:
        return False


def _safe_probability(value):
    try:
        value = float(value)

        if pd.isna(value):
            return None

        return value

    except Exception:
        return None


def render():
    st.title("📡 Live Monitoring")

    # ============================================================
    # LIVE SENSOR DATA
    # ============================================================

    sensor_df = sensors()

    if sensor_df.empty:
        st.error("NO VALID SENSOR DATA")
        return

    latest = sensor_df.iloc[0]

    timestamp = latest.get("timestamp")

    freshness(timestamp)

    st.caption(
        f"Last reading: {timestamp}"
    )

    c1, c2, c3, c4 = st.columns(4)

    temperature = latest.get("temperature")
    humidity = latest.get("humidity")
    smoke = latest.get("smoke")
    flame = latest.get("flame", 0)

    c1.metric(
        "🌡 Temperature",
        (
            f"{float(temperature):.1f} °C"
            if pd.notna(temperature)
            else "N/A"
        ),
    )

    c2.metric(
        "💧 Humidity",
        (
            f"{float(humidity):.1f} %"
            if pd.notna(humidity)
            else "N/A"
        ),
    )

    c3.metric(
        "☁ Smoke",
        (
            f"{float(smoke):.1f}"
            if pd.notna(smoke)
            else "N/A"
        ),
    )

    if safe_flame(flame):
        c4.error("🔥 FLAME DETECTED")
    else:
        c4.success("🛡 NO FLAME")

    # ============================================================
    # ML PREDICTION
    # ============================================================

    st.divider()
    st.subheader("🧠 ML Prediction")

    pred_df = predictions()

    if pred_df.empty:
        st.warning("PREDICTION UNAVAILABLE")
    else:

        pred = pred_df.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Probability",
            probability(
                pred.get("probability")
            ),
        )

        c2.metric(
            "Confidence",
            confidence(
                pred.get("confidence")
            ),
        )

        uncertainty = pred.get("uncertainty")

        c3.metric(
            "Uncertainty",
            (
                f"{float(uncertainty):.1%}"
                if pd.notna(uncertainty)
                else "NOT AVAILABLE"
            ),
        )

        c4.metric(
            "Horizon",
            (
                f"{pred.get('horizon')}h"
                if pd.notna(pred.get("horizon"))
                else "N/A"
            ),
        )

        st.caption(
            f"Model version: "
            f"{pred.get('model_version', 'N/A')} | "
            f"Feature version: "
            f"{pred.get('feature_version', 'N/A')}"
        )

    # ============================================================
    # FIREGUARD ALERT ENGINE
    # ============================================================

    st.divider()
    st.subheader("🚨 FireGuard Alert Engine")

    try:

        runner = LiveAlertRunner()

        alert_result = runner.evaluate(
            history_limit=120,
            horizon="72h",
            sensor_quality=1.0,
            nasa_evidence=None,
            weather_risk=None,
            forecast_risk=None,
            uncertainty=None,
            persist_alert=True,
        )

        # --------------------------------------------------------
        # Forecast / history unavailable
        # --------------------------------------------------------

        if not alert_result.get(
            "success",
            False,
        ):

            reason = alert_result.get(
                "reason"
            )

            message = alert_result.get(
                "message"
            ) or alert_result.get(
                "error"
            ) or "Alert Engine unavailable"

            if reason == "insufficient_sensor_history":

                st.info(
                    "⏳ Forecast انتظار می‌کشد تا "
                    "تاریخچه واقعی کافی از ESP32 جمع‌آوری شود."
                )

                st.caption(
                    f"Sensor history: "
                    f"{alert_result.get('history_count', 0)} "
                    "records"
                )

            else:

                st.warning(
                    f"⚠️ {message}"
                )

        # --------------------------------------------------------
        # Successful evaluation
        # --------------------------------------------------------

        else:

            alert_block = (
                alert_result.get(
                    "alert",
                    {},
                )
            )

            alert = (
                alert_block.get(
                    "alert_result",
                    {},
                )
            )

            level = alert.get(
                "level",
                "INFO",
            )

            probability_value = (
                _safe_probability(
                    alert.get(
                        "fire_probability"
                    )
                )
            )

            reasons = alert.get(
                "reasons",
                [],
            )

            sms_sent = bool(
                alert_block.get(
                    "sms_sent",
                    False,
                )
            )

            sms_cooldown = bool(
                alert_block.get(
                    "sms_cooldown",
                    False,
                )
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Alert Level",
                level,
            )

            c2.metric(
                "Alert Probability",
                (
                    f"{probability_value:.1%}"
                    if probability_value is not None
                    else "N/A"
                ),
            )

            if sms_sent:

                c3.success(
                    "📱 SMS SENT"
                )

            elif sms_cooldown:

                c3.warning(
                    "⏱ SMS COOLDOWN"
                )

            else:

                c3.info(
                    "SMS NOT REQUIRED"
                )

            # ----------------------------------------------------
            # Alert reasons
            # ----------------------------------------------------

            if reasons:

                st.caption(
                    " | ".join(
                        str(x)
                        for x in reasons
                    )
                )

            # ----------------------------------------------------
            # Forecast status
            # ----------------------------------------------------

            forecast = alert_result.get(
                "forecast",
                {},
            )

            if isinstance(
                forecast,
                dict,
            ):

                selected = (
                    forecast.get(
                        "72h"
                    )
                )

                if selected:

                    f1, f2, f3 = st.columns(3)

                    f1.metric(
                        "V4 72h Probability",
                        (
                            f"{float(selected.get('fire_probability')):.1%}"
                            if selected.get(
                                "fire_probability"
                            ) is not None
                            else "N/A"
                        ),
                    )

                    f2.metric(
                        "V4 Threshold",
                        (
                            f"{float(selected.get('threshold')):.2f}"
                            if selected.get(
                                "threshold"
                            ) is not None
                            else "N/A"
                        ),
                    )

                    f3.metric(
                        "V4 Risk",
                        selected.get(
                            "risk_level",
                            "N/A",
                        ),
                    )

                    st.caption(
                        f"V4 model: "
                        f"{selected.get('model_version', 'v4')} | "
                        f"Calibration: "
                        f"{selected.get('calibration_method', 'N/A')}"
                    )

            # ----------------------------------------------------
            # Persistent alert information
            # ----------------------------------------------------

            if alert_block.get(
                "alert_recorded",
                False,
            ):

                event_id = alert.get(
                    "event_id",
                    "N/A",
                )

                st.caption(
                    f"Alert event ID: {event_id}"
                )

            elif alert_block.get(
                "alert_record_error"
            ):

                st.warning(
                    "Alert database record failed: "
                    + str(
                        alert_block.get(
                            "alert_record_error"
                        )
                    )
                )

    except Exception as exc:

        st.error(
            "ALERT ENGINE ERROR"
        )

        st.exception(exc)

    # ============================================================
    # LIVE SOURCE INFORMATION
    # ============================================================

    st.divider()
    st.subheader("📡 Live Source")

    source = latest.get(
        "source",
        "N/A",
    )

    source_url = latest.get(
        "source_url",
        "N/A",
    )

    sc1, sc2, sc3 = st.columns(3)

    sc1.metric(
        "Source",
        str(source),
    )

    sc2.metric(
        "Live",
        "YES"
        if latest.get("is_live", False)
        else "NO",
    )

    sc3.metric(
        "Stale",
        "YES"
        if latest.get("is_stale", False)
        else "NO",
    )

    st.caption(
        f"Source URL: {source_url}"
    )
