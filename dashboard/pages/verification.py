from __future__ import annotations

import streamlit as st

from ..data import verifications


def render():
    st.title("✅ Verification")

    df = verifications()

    if df.empty:
        st.info("NO VERIFICATION RECORDS")
        return

    if "label" in df.columns:

        labels = (
            df["label"]
            .astype(str)
            .str.lower()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Unverified",
            int(
                (labels == "unverified").sum()
            ),
        )

        c2.metric(
            "Confirmed Fire",
            int(
                (labels == "confirmed_fire").sum()
            ),
        )

        c3.metric(
            "Confirmed No Fire",
            int(
                (labels == "confirmed_no_fire").sum()
            ),
        )

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Only human/trusted verification should be used "
        "as training truth."
    )