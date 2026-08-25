import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from ..data import (
    sensors,
    predictions,
    fire_events,
    alerts,
)


def render():

    st.title("📈 Analytics")

    sensor_df = sensors()
    pred_df = predictions()
    event_df = fire_events()
    alert_df = alerts()

    tabs = st.tabs(
        [
            "Sensors",
            "Forecast",
            "Events",
            "Alerts",
        ]
    )

    # ========================================================
    # SENSOR CHARTS
    # ========================================================

    with tabs[0]:

        if sensor_df.empty:
            st.info("No sensor history.")
        else:

            sensor_df["timestamp"] = pd.to_datetime(
                sensor_df["timestamp"],
                errors="coerce",
            )

            sensor_df = sensor_df.sort_values(
                "timestamp"
            )

            fig = go.Figure()

            if "temperature" in sensor_df.columns:

                fig.add_trace(
                    go.Scatter(
                        x=sensor_df["timestamp"],
                        y=sensor_df["temperature"],
                        name="Temperature",
                    )
                )

            if "humidity" in sensor_df.columns:

                fig.add_trace(
                    go.Scatter(
                        x=sensor_df["timestamp"],
                        y=sensor_df["humidity"],
                        name="Humidity",
                    )
                )

            fig.update_layout(
                title="Temperature / Humidity",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            if "smoke" in sensor_df.columns:

                fig_smoke = go.Figure()

                fig_smoke.add_trace(
                    go.Scatter(
                        x=sensor_df["timestamp"],
                        y=sensor_df["smoke"],
                        name="Smoke",
                    )
                )

                fig_smoke.update_layout(
                    title="Smoke Trend",
                    template="plotly_dark",
                    height=400,
                )

                st.plotly_chart(
                    fig_smoke,
                    width="stretch",
                )

    # ========================================================
    # FORECAST CHART
    # ========================================================

    with tabs[1]:

        if pred_df.empty:

            st.info(
                "FORECAST UNAVAILABLE"
            )

        else:

            if "horizon" not in pred_df.columns:
                st.info(
                    "FORECAST UNAVAILABLE"
                )
            else:

                latest = (
                    pred_df
                    .sort_values("timestamp")
                    .groupby("horizon")
                    .tail(1)
                )

                x = []
                y = []

                for horizon in [24, 48, 72]:

                    row = latest[
                        latest["horizon"] == horizon
                    ]

                    if not row.empty:

                        x.append(
                            f"{horizon}h"
                        )

                        y.append(
                            float(
                                row.iloc[0]["probability"]
                            )
                        )

                if not y:

                    st.info(
                        "FORECAST UNAVAILABLE"
                    )

                else:

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines+markers",
                            name="Probability",
                        )
                    )

                    fig.update_layout(
                        title="24h / 48h / 72h Forecast",
                        yaxis=dict(
                            range=[0, 1],
                            tickformat=".0%",
                        ),
                        template="plotly_dark",
                        height=450,
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

    # ========================================================
    # EVENTS
    # ========================================================

    with tabs[2]:

        if event_df.empty:

            st.info(
                "No fire events."
            )

        else:

            event_df["timestamp"] = pd.to_datetime(
                event_df["timestamp"],
                errors="coerce",
            )

            counts = (
                event_df
                .set_index("timestamp")
                .resample("D")
                .size()
            )

            st.bar_chart(
                counts,
                height=400,
            )

    # ========================================================
    # ALERTS
    # ========================================================

    with tabs[3]:

        if alert_df.empty:

            st.info(
                "No alerts."
            )

        else:

            level_col = (
                "alert_level"
                if "alert_level"
                in alert_df.columns
                else None
            )

            if level_col:

                counts = (
                    alert_df[level_col]
                    .value_counts()
                )

                st.bar_chart(
                    counts,
                    height=400,
                )