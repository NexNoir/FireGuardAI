from __future__ import annotations

import streamlit as st

from ..data import models, training_runs


def render():
    st.title("🤖 Model Status")

    df = models()

    if df.empty:
        st.warning(
            "NO MODEL REGISTRY RECORDS"
        )
    else:

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    st.subheader("Training History")

    runs = training_runs()

    if runs.empty:
        st.info(
            "NO TRAINING RUNS"
        )
    else:

        st.dataframe(
            runs,
            width="stretch",
            hide_index=True,
        )

    st.divider()

    st.info(
        "Dashboard never performs training or calibration. "
        "Model replacement must go through the controlled "
        "self-learning pipeline."
    )