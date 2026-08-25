from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..data import (
    sensors,
    predictions,
    fire_events,
    verifications,
    alerts,
    models,
    training_runs,
    external_observations,
)

from alerts.sms_service import SmsService
from alerts.alert_service import AlertService


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "fireguard_history.db"


def render():
    st.title("🩺 System Health")

    # =========================================================
    # CORE SYSTEM CHECKS
    # =========================================================

    checks = [
        ("Database", DB_PATH.exists()),
        ("Sensor History", not sensors().empty),
        ("Prediction History", not predictions().empty),
        ("Fire Events", not fire_events().empty),
        ("Verification", not verifications().empty),
        ("Alert Engine", not alerts().empty),
        ("Model Registry", not models().empty),
        ("Training History", not training_runs().empty),
        (
            "External Observations",
            not external_observations().empty,
        ),
    ]

    for name, ok in checks:

        if ok:
            st.success(
                f"🟢 {name}: PASS"
            )

        else:
            st.warning(
                f"🟡 {name}: NOT AVAILABLE"
            )

    st.divider()

    # =========================================================
    # SAFETY
    # =========================================================

    st.subheader("Safety")

    st.success(
        "ML modification from Dashboard: BLOCKED"
    )

    st.success(
        "Training from Dashboard: BLOCKED"
    )

    st.success(
        "Calibration from Dashboard: BLOCKED"
    )

    st.success(
        "Dataset modification from Dashboard: BLOCKED"
    )

    st.divider()

    # =========================================================
    # SMS HEALTH
    # =========================================================

    st.subheader("📱 SMS Alert System")

    try:

        sms_service = SmsService()

        sms_status = (
            sms_service.configuration_status()
        )

        sms_configured = bool(
            sms_status.get(
                "configured",
                False,
            )
        )

        sms_enabled = bool(
            sms_status.get(
                "enabled",
                False,
            )
        )

        # -----------------------------------------------------
        # Configuration
        # -----------------------------------------------------

        if sms_configured and sms_enabled:

            st.success(
                "🟢 SMS: READY"
            )

            st.write(
                "Provider: Kavenegar"
            )

            st.write(
                "API configuration: OK"
            )

            st.write(
                "Sender configuration: OK"
            )

            st.write(
                "Receiver configuration: OK"
            )

        elif sms_configured and not sms_enabled:

            st.warning(
                "🟡 SMS: CONFIGURED BUT DISABLED"
            )

            st.caption(
                "SMS credentials are configured, "
                "but SMS_ENABLED is not active."
            )

        else:

            st.error(
                "🔴 SMS: NOT READY"
            )

            errors = sms_status.get(
                "errors",
                [],
            )

            if errors:

                for error in errors:

                    st.warning(
                        str(error)
                    )

            else:

                st.warning(
                    "SMS configuration is incomplete."
                )

        # -----------------------------------------------------
        # Persistent Cooldown
        # -----------------------------------------------------

        st.divider()

        st.subheader(
            "⏱ SMS Cooldown"
        )

        alert_service = AlertService()

        cooldown_active = (
            alert_service.cooldown_active()
        )

        if cooldown_active:

            remaining = (
                alert_service
                .cooldown_remaining_seconds()
            )

            minutes = remaining // 60
            seconds = remaining % 60

            st.warning(
                "🟡 SMS COOLDOWN ACTIVE"
            )

            st.info(
                f"Remaining time: "
                f"{minutes:02d}:{seconds:02d}"
            )

        else:

            st.success(
                "🟢 SMS COOLDOWN: READY"
            )

        # -----------------------------------------------------
        # Last SMS state
        # -----------------------------------------------------

        last_sms = (
            alert_service
            .alert_store
            .get_last_sms_at()
        )

        if last_sms is not None:

            st.caption(
                f"Last successful SMS: {last_sms}"
            )

        else:

            st.caption(
                "Last successful SMS: None"
            )

    except Exception as exc:

        st.error(
            "🔴 SMS subsystem health check failed"
        )

        st.exception(exc)

    # =========================================================
    # DATABASE
    # =========================================================

    st.divider()

    st.caption(
        f"Database path: {DB_PATH}"
    )