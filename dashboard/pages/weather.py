from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from weather.live_weather import LiveWeatherService


def _show_value(value, digits=1, suffix=""):
    try:
        value = float(value)

        if pd.isna(value):
            return "N/A"

        return f"{value:.{digits}f}{suffix}"

    except Exception:
        return "N/A"


def render():
    st.title("🌦 Weather")
    st.caption("Real external weather data only")

    service = LiveWeatherService()
    result = service.get_weather()

    if not result.get("available", False):

        st.error("WEATHER UNAVAILABLE")

        st.caption(
            result.get("error", "Unknown weather error")
        )

        st.info(
            "No fake weather values are generated when the external "
            "weather service is unavailable."
        )

        return

    data = result.get("data")

    if not isinstance(data, dict):

        st.error("WEATHER UNAVAILABLE")
        st.caption("No valid weather payload received.")

        return

    st.success("🟢 REAL WEATHER DATA")

    st.caption(
        f"Source: {result.get('source')} | "
        f"Observed: {data.get('observed_at')}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🌡 Temperature",
        _show_value(
            data.get("temperature"),
            1,
            " °C",
        ),
    )

    c2.metric(
        "💧 Humidity",
        _show_value(
            data.get("humidity"),
            0,
            " %",
        ),
    )

    c3.metric(
        "💨 Wind Speed",
        _show_value(
            data.get("wind_speed"),
            1,
            " m/s",
        ),
    )

    c4.metric(
        "🔽 Pressure",
        _show_value(
            data.get("pressure"),
            0,
            " hPa",
        ),
    )

    st.divider()

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Wind Direction",
        _show_value(
            data.get("wind_direction"),
            0,
            "°",
        ),
    )

    d2.metric(
        "Condition",
        data.get("description") or "N/A",
    )

    d3.metric(
        "Coordinates",
        (
            f"{data.get('latitude')}, "
            f"{data.get('longitude')}"
        ),
    )

    with st.expander("Raw Weather Record"):
        st.json(
            {
                key: str(value)
                for key, value in data.items()
            }
        )