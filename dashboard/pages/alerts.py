
from __future__ import annotations

import streamlit as st

from ..data import alerts

try:
    from alerts.sms_service import SmsService
except ImportError:
    SmsService = None


LEVELS = [
    "CRITICAL",
    "HIGH",
    "WARNING",
    "WATCH",
    "INFO",
]


def render():
    st.title("Alerts")

    df = alerts()

    if df.empty:
        st.success("NO ALERTS RECORDED")
    else:
        level_column = None

        for name in ["alert_level", "level", "risk_level"]:
            if name in df.columns:
                level_column = name
                break

        if level_column:
            selected = st.multiselect(
                "Severity",
                LEVELS,
                default=LEVELS,
            )

            filtered = df[
                df[level_column]
                .astype(str)
                .str.upper()
                .isin(selected)
            ]
        else:
            filtered = df

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total",
            len(filtered),
        )

        if level_column:
            c2.metric(
                "Critical",
                int(
                    (
                        filtered[level_column]
                        .astype(str)
                        .str.upper()
                        == "CRITICAL"
                    ).sum()
                ),
            )

            c3.metric(
                "High",
                int(
                    (
                        filtered[level_column]
                        .astype(str)
                        .str.upper()
                        == "HIGH"
                    ).sum()
                ),
            )

        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    # ========================================================
    # SMS SERVICE STATUS
    # ========================================================

    st.subheader("SMS Service")

    if SmsService is None:
        st.error("SMS Service could not be imported.")
        return

    try:
        sms_service = SmsService()

        status = sms_service.configuration_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            if status.get("enabled"):
                st.success("SMS ENABLED")
            else:
                st.error("SMS DISABLED")

        with col2:
            if status.get("api_key_configured"):
                st.success("KAVENEGAR CONFIGURED")
            else:
                st.error("KAVENEGAR NOT CONFIGURED")

        with col3:
            if status.get("receiver_configured"):
                st.success("RECEIVER CONFIGURED")
            else:
                st.error("RECEIVER NOT CONFIGURED")

        st.divider()

        st.subheader("SMS Configuration")

        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write(
                "Provider: Kavenegar"
            )

            st.write(
                "Sender configured:",
                "YES"
                if status.get("sender_configured")
                else "NO",
            )

        with config_col2:
            st.write(
                "Receiver configured:",
                "YES"
                if status.get("receiver_configured")
                else "NO",
            )

            st.write(
                "SMS enabled:",
                "YES"
                if status.get("enabled")
                else "NO",
            )

        # ====================================================
        # CONFIGURATION ERRORS
        # ====================================================

        errors = status.get("errors", [])

        if errors:
            st.warning("SMS configuration requires attention.")

            for error in errors:
                st.error(str(error))
        else:
            st.success(
                "SMS service configuration is valid."
            )

    except Exception as exc:
        st.error("SMS SERVICE ERROR")
        st.exception(exc)