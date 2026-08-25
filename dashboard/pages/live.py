from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

from database.db import FireGuardDatabase
from sensors.live_sensor import LiveSensorService

from alert_engine.alert_engine import (
    AlertEngine,
    AlertInput,
)

from alert_engine.alert_store import AlertStore
from alerts.sms_service import SmsService

from ..data import predictions
from ..components import (
    freshness,
    probability,
    confidence,
)


# ============================================================
# CONFIGURATION
# ============================================================

try:
    SMS_COOLDOWN_MINUTES = max(
        1,
        int(
            os.getenv(
                "SMS_COOLDOWN_MINUTES",
                "15",
            )
        ),
    )
except ValueError:
    SMS_COOLDOWN_MINUTES = 15


# ============================================================
# REAL ESP32 SENSOR
# ============================================================

def read_live_sensor() -> dict[str, Any]:
    """
    Read the real ESP32 sensor and store the accepted reading.

    No synthetic data is generated.
    """

    try:
        db = FireGuardDatabase()

        service = LiveSensorService(
            database=db
        )

        return service.read_and_store()

    except Exception as exc:

        return {
            "is_valid": False,
            "is_live": False,
            "is_stale": True,
            "source": "esp32",
            "errors": [str(exc)],
        }


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_flame(value: Any) -> bool:
    try:
        return int(float(value)) == 1
    except Exception:
        return False


def safe_float(
    value: Any,
) -> Optional[float]:

    try:

        if value is None:
            return None

        value = float(value)

        if pd.isna(value):
            return None

        return value

    except Exception:
        return None


def safe_timestamp(
    value: Any,
) -> Optional[datetime]:

    if value is None:
        return None

    try:

        timestamp = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(timestamp):
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                "UTC"
            )

        return timestamp.to_pydatetime()

    except Exception:
        return None


# ============================================================
# LATEST PREDICTION
# ============================================================

def get_latest_prediction():

    try:

        df = predictions()

        if df.empty:
            return None

        return df.iloc[0]

    except Exception:

        return None


# ============================================================
# PERSIST ALERT
# ============================================================

def persist_alert_result(
    result,
) -> dict[str, Any]:
    """
    Store fire_event + alert using the existing database API.

    Database failure must not crash Live Monitoring.
    """

    try:

        db = FireGuardDatabase()

        reason = " | ".join(
            result.reasons
        )

        event_row_id = (
            db.add_fire_event(
                event_id=result.event_id,
                timestamp=result.created_at,
                event_type="fire_risk",
                status="open",
                description=reason,
            )
        )

        alert_row_id = (
            db.add_alert(
                event_id=result.event_id,
                alert_level=result.level,
                reason=reason,
                status="active",
            )
        )

        return {
            "success": True,
            "event_row_id": event_row_id,
            "alert_row_id": alert_row_id,
            "error": None,
        }

    except Exception as exc:

        return {
            "success": False,
            "event_row_id": None,
            "alert_row_id": None,
            "error": str(exc),
        }


# ============================================================
# SMS COOLDOWN
# ============================================================

def sms_cooldown_state(
    store: AlertStore,
) -> tuple[bool, int]:

    last_sms = (
        store.get_last_sms_at()
    )

    if last_sms is None:
        return False, 0

    now = datetime.now(
        timezone.utc
    )

    if last_sms.tzinfo is None:
        last_sms = last_sms.replace(
            tzinfo=timezone.utc
        )

    elapsed = (
        now - last_sms
    ).total_seconds()

    cooldown_seconds = (
        SMS_COOLDOWN_MINUTES * 60
    )

    remaining = (
        cooldown_seconds - elapsed
    )

    if remaining <= 0:
        return False, 0

    return True, int(
        remaining
    )


# ============================================================
# SMS SAFETY GATE
# ============================================================

def authorize_sms(
    *,
    level: str,
    probability_value: Optional[float],
    flame: int,
    sensor_quality: float,
    smoke: Optional[float],
) -> tuple[bool, str]:

    # --------------------------------------------------------
    # Sensor must be trustworthy
    # --------------------------------------------------------

    if sensor_quality < 1.0:

        return (
            False,
            "SMS blocked: sensor data is incomplete",
        )

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    if level == "INFO":

        return (
            False,
            "SMS not required: INFO level",
        )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    if level == "WATCH":

        return (
            False,
            "SMS not sent: WATCH is dashboard-only",
        )

    # --------------------------------------------------------
    # Flame evidence
    # --------------------------------------------------------

    if flame == 1:

        return (
            True,
            "SMS authorized: flame detected",
        )

    # --------------------------------------------------------
    # ML evidence
    # --------------------------------------------------------

    if (
        probability_value is not None
        and probability_value >= 0.85
    ):

        return (
            True,
            "SMS authorized: ML probability >= 85%",
        )

    # --------------------------------------------------------
    # HIGH / CRITICAL without local evidence
    # --------------------------------------------------------

    if level in {
        "HIGH",
        "CRITICAL",
    }:

        return (
            False,
            (
                "SMS blocked: alert level is high "
                "but there is no flame or sufficient "
                "local ML evidence"
            ),
        )

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    if level == "WARNING":

        return (
            False,
            (
                "SMS blocked: WARNING requires "
                "credible local fire evidence"
            ),
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return (
        False,
        "SMS blocked: no credible fire evidence",
    )


# ============================================================
# ALERT + SMS PIPELINE
# ============================================================

def run_alert_pipeline(
    live_result: dict[str, Any],
    prediction_row,
) -> dict[str, Any]:
    """
    Complete live alert pipeline.

    IMPORTANT:
    SMS is deliberately NOT delegated to AlertService here.
    The Safety Gate runs BEFORE SmsService.send_alert().

        Sensor
          ↓
        AlertEngine
          ↓
        Safety Gate
          ↓
        Cooldown
          ↓
        Kavenegar
    """

    temperature = safe_float(
        live_result.get(
            "temperature"
        )
    )

    humidity = safe_float(
        live_result.get(
            "humidity"
        )
    )

    smoke = safe_float(
        live_result.get(
            "smoke"
        )
    )

    flame = (
        1
        if safe_flame(
            live_result.get(
                "flame",
                0,
            )
        )
        else 0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    fire_probability = None
    uncertainty = None

    if prediction_row is not None:

        fire_probability = safe_float(
            prediction_row.get(
                "probability"
            )
        )

        uncertainty = safe_float(
            prediction_row.get(
                "uncertainty"
            )
        )

    # --------------------------------------------------------
    # Sensor quality
    # --------------------------------------------------------

    sensor_quality = 1.0

    if (
        temperature is None
        or humidity is None
        or smoke is None
    ):
        sensor_quality = 0.0

    # --------------------------------------------------------
    # Alert Input
    #
    # Do NOT fabricate:
    # - smoke trend
    # - NASA evidence
    # - weather risk
    # - forecast risk
    # --------------------------------------------------------

    alert_input = AlertInput(
        fire_probability=fire_probability,
        flame=flame,
        smoke_trend=None,
        sensor_quality=sensor_quality,
        nasa_evidence=None,
        weather_risk=None,
        forecast_risk=None,
        uncertainty=uncertainty,
        temperature=temperature,
        humidity=humidity,
        smoke=smoke,
        timestamp=str(
            live_result.get(
                "timestamp"
            )
        ),
        source="ESP32 + FireGuard",
    )

    # --------------------------------------------------------
    # AlertEngine
    # --------------------------------------------------------

    engine = AlertEngine()

    result = engine.evaluate(
        alert_input
    )

    output = {
        "alert_result": result.to_dict(),
        "alert_recorded": False,
        "alert_record_error": None,
        "sms_authorized": False,
        "sms_attempted": False,
        "sms_sent": False,
        "sms_cooldown": False,
        "sms_message": None,
        "sms_provider": None,
    }

    level = result.level

    # --------------------------------------------------------
    # INFO = no alert
    # --------------------------------------------------------

    if result.level_value <= 0:

        output["sms_message"] = (
            "SMS not required: normal / INFO condition"
        )

        return output

    # --------------------------------------------------------
    # Store Alert
    # --------------------------------------------------------

    persistence = (
        persist_alert_result(
            result
        )
    )

    output[
        "alert_recorded"
    ] = persistence[
        "success"
    ]

    output[
        "alert_record_error"
    ] = persistence[
        "error"
    ]

    # --------------------------------------------------------
    # SMS Safety Gate
    # --------------------------------------------------------

    authorized, reason = (
        authorize_sms(
            level=level,
            probability_value=(
                result.fire_probability
            ),
            flame=flame,
            sensor_quality=sensor_quality,
            smoke=smoke,
        )
    )

    output[
        "sms_authorized"
    ] = authorized

    if not authorized:

        output[
            "sms_message"
        ] = reason

        return output

    # --------------------------------------------------------
    # Persistent Cooldown
    # --------------------------------------------------------

    store = AlertStore()

    cooldown_active, remaining = (
        sms_cooldown_state(
            store
        )
    )

    if cooldown_active:

        minutes = remaining // 60
        seconds = remaining % 60

        output[
            "sms_cooldown"
        ] = True

        output[
            "sms_message"
        ] = (
            "پیامک در حالت محدودیت ارسال است؛ "
            f"{minutes:02d}:{seconds:02d} باقی مانده"
        )

        return output

    # --------------------------------------------------------
    # SMS Configuration
    # --------------------------------------------------------

    sms_service = SmsService()

    configuration = (
        sms_service.configuration_status()
    )

    if not configuration.get(
        "configured",
        False,
    ):

        output[
            "sms_message"
        ] = (
            "SMS unavailable: "
            + "; ".join(
                configuration.get(
                    "errors",
                    [],
                )
            )
        )

        return output

    # --------------------------------------------------------
    # ACTUAL SMS SEND
    # --------------------------------------------------------

    output[
        "sms_attempted"
    ] = True

    sms_result = (
        sms_service.send_alert(
            risk_level=level,
            probability=(
                result.fire_probability
            ),
            temperature=temperature,
            humidity=humidity,
            smoke=smoke,
            flame=flame,
            source="ESP32 + FireGuard",
        )
    )

    output[
        "sms_provider"
    ] = sms_result.get(
        "provider",
        "kavenegar",
    )

    # --------------------------------------------------------
    # SMS SUCCESS
    # --------------------------------------------------------

    if sms_result.get(
        "success",
        False,
    ):

        now = datetime.now(
            timezone.utc
        )

        store.save_sms_success(
            timestamp=now,
            risk_level=level,
            probability=float(
                result.fire_probability
                if result.fire_probability
                is not None
                else 0.0
            ),
        )

        output[
            "sms_sent"
        ] = True

        output[
            "sms_message"
        ] = (
            "پیامک هشدار با موفقیت "
            "به Kavenegar تحویل شد"
        )

    # --------------------------------------------------------
    # SMS FAILURE
    # --------------------------------------------------------

    else:

        store.save_sms_failure(
            timestamp=datetime.now(
                timezone.utc
            ),
            risk_level=level,
            probability=float(
                result.fire_probability
                if result.fire_probability
                is not None
                else 0.0
            ),
        )

        output[
            "sms_message"
        ] = (
            sms_result.get(
                "error"
            )
            or "ارسال پیامک ناموفق بود"
        )

    return output


# ============================================================
# RENDER
# ============================================================

def render():

    st.title(
        "📡 Live Monitoring"
    )

    # ========================================================
    # REAL ESP32
    # ========================================================

    live_result = read_live_sensor()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not live_result.get(
        "is_valid",
        False,
    ):

        st.error(
            "🔴 SENSOR UNAVAILABLE"
        )

        st.warning(
            "STALE DATA"
        )

        errors = live_result.get(
            "errors",
            [],
        )

        if errors:

            st.caption(
                f"Sensor error: {errors[0]}"
            )

        st.info(
            "No new sensor reading was accepted."
        )

        return

    if not live_result.get(
        "is_live",
        False,
    ):

        st.warning(
            "STALE DATA"
        )

        return

    if live_result.get(
        "is_stale",
        True,
    ):

        st.warning(
            "STALE DATA"
        )

        return

    # ========================================================
    # LIVE STATUS
    # ========================================================

    st.success(
        "🟢 LIVE SENSOR DATA"
    )

    timestamp = live_result.get(
        "timestamp"
    )

    st.caption(
        f"Source: ESP32 | "
        f"Last reading: {timestamp}"
    )

    # ========================================================
    # SENSOR VALUES
    # ========================================================

    temperature = safe_float(
        live_result.get(
            "temperature"
        )
    )

    humidity = safe_float(
        live_result.get(
            "humidity"
        )
    )

    smoke = safe_float(
        live_result.get(
            "smoke"
        )
    )

    flame = live_result.get(
        "flame",
        0,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌡 Temperature",
        (
            f"{temperature:.1f} °C"
            if temperature is not None
            else "N/A"
        ),
    )

    c2.metric(
        "💧 Humidity",
        (
            f"{humidity:.1f} %"
            if humidity is not None
            else "N/A"
        ),
    )

    c3.metric(
        "☁ Smoke",
        (
            f"{smoke:.0f}"
            if smoke is not None
            else "N/A"
        ),
    )

    if safe_flame(flame):

        c4.error(
            "🔥 FLAME DETECTED"
        )

    else:

        c4.success(
            "🛡 NO FLAME"
        )

    freshness(timestamp)

    # ========================================================
    # DATABASE STORAGE
    # ========================================================

    if live_result.get(
        "stored",
        False,
    ):

        st.caption(
            "✓ Stored in database | "
            f"Record ID: "
            f"{live_result.get('database_id')}"
        )

    else:

        db_error = live_result.get(
            "database_error"
        )

        if db_error:

            st.warning(
                "Sensor is live but database "
                f"storage failed: {db_error}"
            )

        else:

            st.warning(
                "Sensor is live but this reading "
                "was not stored."
            )

    # ========================================================
    # ML PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 ML Prediction"
    )

    pred = get_latest_prediction()

    if pred is None:

        st.warning(
            "FORECAST UNAVAILABLE"
        )

        st.info(
            "No valid prediction exists in the database yet."
        )

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Probability",
            probability(
                pred.get(
                    "probability"
                )
            ),
        )

        c2.metric(
            "Confidence",
            confidence(
                pred.get(
                    "confidence"
                )
            ),
        )

        uncertainty = pred.get(
            "uncertainty"
        )

        c3.metric(
            "Uncertainty",
            (
                f"{float(uncertainty):.1%}"
                if pd.notna(
                    uncertainty
                )
                else "NOT AVAILABLE"
            ),
        )

        horizon = pred.get(
            "horizon"
        )

        c4.metric(
            "Horizon",
            (
                f"{int(horizon)}h"
                if pd.notna(
                    horizon
                )
                else "N/A"
            ),
        )

        st.caption(
            f"Model version: "
            f"{pred.get('model_version', 'N/A')} | "
            f"Feature version: "
            f"{pred.get('feature_version', 'N/A')}"
        )

    # ========================================================
    # ALERT ENGINE
    # ========================================================

    st.divider()

    st.subheader(
        "🚨 FireGuard Alert Engine"
    )

    try:

        alert_output = run_alert_pipeline(
            live_result=live_result,
            prediction_row=pred,
        )

        alert_result = (
            alert_output.get(
                "alert_result",
                {},
            )
        )

        level = alert_result.get(
            "level",
            "INFO",
        )

        level_value = alert_result.get(
            "level_value",
            0,
        )

        reasons = alert_result.get(
            "reasons",
            [],
        )

        fire_probability = (
            alert_result.get(
                "fire_probability"
            )
        )

        # ----------------------------------------------------
        # Alert level
        # ----------------------------------------------------

        if level == "CRITICAL":

            st.error(
                "🔴 CRITICAL ALERT"
            )

        elif level == "HIGH":

            st.error(
                "🟠 HIGH ALERT"
            )

        elif level == "WARNING":

            st.warning(
                "🟡 WARNING"
            )

        elif level == "WATCH":

            st.info(
                "🔵 WATCH"
            )

        else:

            st.success(
                "🟢 INFO / NORMAL"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Risk Level",
            level,
        )

        a2.metric(
            "ML Probability",
            (
                f"{float(fire_probability):.1%}"
                if fire_probability is not None
                else "N/A"
            ),
        )

        a3.metric(
            "Alert Value",
            str(level_value),
        )

        # ----------------------------------------------------
        # Reasons
        # ----------------------------------------------------

        if reasons:

            st.markdown(
                "**Alert Evidence:**"
            )

            for reason in reasons:

                st.write(
                    f"• {reason}"
                )

        # ====================================================
        # SMS STATUS
        # ====================================================

        st.subheader(
            "📱 SMS Notification"
        )

        if alert_output.get(
            "sms_sent",
            False,
        ):

            st.success(
                "🟢 SMS SENT"
            )

            st.caption(
                alert_output.get(
                    "sms_message",
                    "پیامک هشدار با موفقیت ارسال شد",
                )
            )

        elif alert_output.get(
            "sms_cooldown",
            False,
        ):

            st.warning(
                "🟡 SMS COOLDOWN"
            )

            st.caption(
                alert_output.get(
                    "sms_message",
                    "پیامک به دلیل محدودیت زمانی ارسال نشد",
                )
            )

        elif alert_output.get(
            "sms_attempted",
            False,
        ):

            st.error(
                "🔴 SMS FAILED"
            )

            st.caption(
                alert_output.get(
                    "sms_message",
                    "ارسال پیامک ناموفق بود",
                )
            )

        elif alert_output.get(
            "sms_authorized",
            False,
        ):

            st.info(
                "ℹ️ SMS AUTHORIZED"
            )

        else:

            st.success(
                "🟢 SMS NOT SENT"
            )

            st.caption(
                alert_output.get(
                    "sms_message",
                    "شرایط لازم برای ارسال پیامک وجود ندارد.",
                )
            )

        # ----------------------------------------------------
        # Alert database
        # ----------------------------------------------------

        if alert_output.get(
            "alert_recorded",
            False,
        ):

            st.caption(
                "✓ Alert event recorded in database"
            )

        elif alert_output.get(
            "alert_record_error"
        ):

            st.warning(
                "Alert database error: "
                + str(
                    alert_output.get(
                        "alert_record_error"
                    )
                )
            )

    except Exception as exc:

        st.error(
            "⚠️ ALERT ENGINE ERROR"
        )

        st.exception(exc)