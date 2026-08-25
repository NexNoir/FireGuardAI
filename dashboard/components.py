from datetime import datetime, timezone

import streamlit as st
import pandas as pd


def stale(timestamp, minutes=10):

    if timestamp is None:
        return True

    ts = pd.to_datetime(
        timestamp,
        errors="coerce",
        utc=True,
    )

    if pd.isna(ts):
        return True

    age = (
        datetime.now(timezone.utc) - ts.to_pydatetime()
    ).total_seconds() / 60

    return age > minutes


def freshness(timestamp):

    if stale(timestamp):

        st.markdown(
            """
            <div class="status-stale">
            STALE DATA
            </div>
            """,
            unsafe_allow_html=True,
        )

        return False

    st.markdown(
        """
        <div class="status-live">
        ● LIVE DATA
        </div>
        """,
        unsafe_allow_html=True,
    )

    return True


def probability(value):

    if value is None:
        return "N/A"

    try:
        value = float(value)

        if not 0 <= value <= 1:
            return "N/A"

        return f"{value:.1%}"

    except Exception:
        return "N/A"


def confidence(value):

    if value is None:
        return "NOT AVAILABLE"

    try:
        value = float(value)

        if not 0 <= value <= 1:
            return "NOT AVAILABLE"

        return f"{value:.1%}"

    except Exception:
        return "NOT AVAILABLE"


def unavailable(label):

    st.markdown(
        f"""
        <div class="status-unavailable">
        {label}
        </div>
        """,
        unsafe_allow_html=True,
    )