from __future__ import annotations

import streamlit as st

from ..data import fire_events, verifications


def render():
    st.title("🔥 Fire Events")

    events = fire_events()
    verification = verifications()

    if events.empty:
        st.info("NO FIRE EVENTS")
        return

    st.metric(
        "Total Fire Events",
        len(events),
    )

    st.dataframe(
        events,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Verification")

    if verification.empty:
        st.info("NO VERIFICATION RECORDS")
    else:
        st.dataframe(
            verification,
            width="stretch",
            hide_index=True,
        )