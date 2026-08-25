from __future__ import annotations

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


def render():
    st.title("🗄️ Database History")

    tables = {
        "Sensor Readings": sensors,
        "Predictions": predictions,
        "Fire Events": fire_events,
        "Verification": verifications,
        "Alerts": alerts,
        "Models": models,
        "Training Runs": training_runs,
        "External Observations": external_observations,
    }

    for name, loader in tables.items():

        with st.expander(name):

            df = loader()

            if df.empty:
                st.info("No records.")
            else:
                st.dataframe(
                    df.head(100),
                    width="stretch",
                    hide_index=True,
                )