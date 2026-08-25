from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from nasa.live_firms import LiveFirmsService
from ..data import external_observations


# ============================================================
# Hircanian Forest Study Area
# ============================================================

HIRCANIAN_CENTER = [37.2, 51.5]

HIRCANIAN_BOUNDS = [
    [35.8, 48.5],
    [38.5, 54.5],
]


# ============================================================
# Helpers
# ============================================================

def _safe_float(value):
    try:
        value = float(value)

        if pd.isna(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def _safe_text(value):
    if value is None:
        return "N/A"

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return "N/A"

    return text


def _parse_timestamp(row):
    """
    Build an observation timestamp only from actual NASA FIRMS
    acquisition fields.

    No timestamp is invented.
    """

    acq_date = row.get("acq_date")
    acq_time = row.get("acq_time")

    if pd.isna(acq_date) or pd.isna(acq_time):
        return None

    try:
        date_text = str(acq_date).strip()

        time_number = int(float(acq_time))
        time_text = f"{time_number:04d}"

        hour = int(time_text[:2])
        minute = int(time_text[2:])

        return datetime(
            year=int(date_text[0:4]),
            month=int(date_text[5:7]),
            day=int(date_text[8:10]),
            hour=hour,
            minute=minute,
            tzinfo=timezone.utc,
        )

    except Exception:
        return None


def _prepare_dataframe(observations):
    """
    Convert only real NASA FIRMS observations into a DataFrame.

    No scientific values are generated here.
    """

    if not observations:
        return pd.DataFrame()

    df = pd.DataFrame(observations)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        )

    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        )

    df = df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    # --------------------------------------------------------
    # Geographic validation
    # --------------------------------------------------------

    if "latitude" in df.columns and "longitude" in df.columns:
        df = df[
            df["latitude"].between(35.8, 38.5)
            & df["longitude"].between(48.5, 54.5)
        ].copy()

    # --------------------------------------------------------
    # Numeric NASA measurements
    # --------------------------------------------------------

    for column in (
        "frp",
        "brightness",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Acquisition timestamp
    # --------------------------------------------------------

    if "acq_date" in df.columns and "acq_time" in df.columns:

        timestamps = []

        for _, row in df.iterrows():
            timestamps.append(
                _parse_timestamp(row)
            )

        df["observation_timestamp"] = timestamps

    return df.reset_index(drop=True)


# ============================================================
# Map
# ============================================================

def _render_map(df):
    """
    Render the Hircanian study area.

    🔴 = real NASA FIRMS observation
    🔵 = configured FireGuard sensor location
    """

    try:
        import folium
        from streamlit_folium import st_folium

    except ImportError:

        st.error(
            "NASA map dependencies are missing."
        )

        st.code(
            "pip install folium streamlit-folium"
        )

        return

    st.subheader(
        "🌍 Hircanian Forest Live Fire Map"
    )

    st.caption(
        "🔴 Red points = real NASA FIRMS observations | "
        "🔵 Blue point = FireGuard sensor"
    )

    service = LiveFirmsService()

    try:
        sensor_lat = float(service.lat)
        sensor_lon = float(service.lon)

    except Exception:
        sensor_lat = HIRCANIAN_CENTER[0]
        sensor_lon = HIRCANIAN_CENTER[1]

    m = folium.Map(
        location=HIRCANIAN_CENTER,
        zoom_start=7,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # --------------------------------------------------------
    # Hircanian study area
    # --------------------------------------------------------

    folium.Rectangle(
        bounds=HIRCANIAN_BOUNDS,
        color="orange",
        weight=2,
        fill=True,
        fill_opacity=0.10,
        popup="Hircanian Forest Study Area",
        tooltip="Hircanian Forest Study Area",
    ).add_to(m)

    # --------------------------------------------------------
    # Sensor
    # --------------------------------------------------------

    folium.Marker(
        location=[
            sensor_lat,
            sensor_lon,
        ],
        popup=(
            "📍 FireGuard ESP32 Sensor"
        ),
        tooltip="FireGuard ESP32 Sensor",
        icon=folium.Icon(
            color="blue",
            icon="microchip",
            prefix="fa",
        ),
    ).add_to(m)

    # --------------------------------------------------------
    # NASA observations
    # --------------------------------------------------------

    if not df.empty:

        for _, row in df.head(500).iterrows():

            lat = _safe_float(
                row.get("latitude")
            )

            lon = _safe_float(
                row.get("longitude")
            )

            if lat is None or lon is None:
                continue

            frp = _safe_text(
                row.get("frp")
            )

            confidence = _safe_text(
                row.get("confidence")
            )

            brightness = _safe_text(
                row.get("brightness")
            )

            satellite = _safe_text(
                row.get("satellite")
            )

            instrument = _safe_text(
                row.get("instrument")
            )

            acq_date = _safe_text(
                row.get("acq_date")
            )

            acq_time = _safe_text(
                row.get("acq_time")
            )

            popup_html = f"""
            <b>🔥 NASA FIRMS Active Fire</b><br>
            Latitude: {lat:.5f}<br>
            Longitude: {lon:.5f}<br>
            FRP: {frp} MW<br>
            Confidence: {confidence}<br>
            Brightness: {brightness}<br>
            Satellite: {satellite}<br>
            Instrument: {instrument}<br>
            Acquisition: {acq_date} {acq_time}
            """

            folium.CircleMarker(
                location=[
                    lat,
                    lon,
                ],
                radius=7,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(
                    popup_html,
                    max_width=350,
                ),
                tooltip="🔥 NASA FIRMS detection",
            ).add_to(m)

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    st_folium(
        m,
        width=None,
        height=700,
        returned_objects=[],
    )


# ============================================================
# Main page
# ============================================================

def render():

    st.title("🛰 NASA FIRMS")

    st.caption(
        "Real NASA satellite observations only • "
        "No fake FRP • No fake confidence"
    )

    # --------------------------------------------------------
    # Refresh / manual fetch
    # --------------------------------------------------------

    refresh_col, status_col = st.columns(
        [1, 4]
    )

    with refresh_col:

        refresh_clicked = st.button(
            "🔄 Refresh NASA",
            width="stretch",
        )

    # Auto-refresh from dashboard/app.py causes the page
    # to execute again every REFRESH_SECONDS.
    #
    # The button is also available for an immediate manual
    # request.

    if refresh_clicked:
        st.rerun()

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    service = LiveFirmsService()

    with st.spinner(
        "در حال دریافت داده واقعی NASA FIRMS..."
    ):

        result = service.fetch()

    # --------------------------------------------------------
    # API unavailable
    # --------------------------------------------------------

    if not result.get("available", False):

        with status_col:
            st.error(
                "🔴 NASA FIRMS UNAVAILABLE"
            )

        st.warning(
            result.get(
                "error",
                "Unknown NASA FIRMS error",
            )
        )

        st.info(
            "هیچ داده جعلی نمایش داده نمی‌شود. "
            "FireGuard بدون NASA نیز می‌تواند ادامه دهد."
        )

        st.divider()

        st.subheader(
            "Hircanian Forest Map"
        )

        _render_map(
            pd.DataFrame()
        )

        return

    # --------------------------------------------------------
    # API connected
    # --------------------------------------------------------

    with status_col:
        st.success(
            "🟢 NASA FIRMS API CONNECTED"
        )

    observations = result.get(
        "observations",
        [],
    )

    df = _prepare_dataframe(
        observations
    )

    checked_at = result.get(
        "checked_at"
    )

    st.caption(
        f"Source: {result.get('source', 'NASA FIRMS')} | "
        f"Checked: {checked_at}"
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if df.empty:

        st.info(
            "در آخرین دریافت NASA FIRMS، "
            "هیچ مشاهده معتبر آتش در محدوده هیرکانی "
            "برگردانده نشد."
        )

        st.divider()

        _render_map(
            pd.DataFrame()
        )

        st.divider()

        st.subheader(
            "NASA Data Status"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "API",
            "CONNECTED",
        )

        c2.metric(
            "Real Observations",
            "0",
        )

        c3.metric(
            "Fake Data",
            "0",
        )

        return

    # --------------------------------------------------------
    # Real observations received
    # --------------------------------------------------------

    st.success(
        f"🔥 {len(df)} REAL NASA OBSERVATIONS RECEIVED"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Observations",
        len(df),
    )

    if "frp" in df.columns:

        frp = pd.to_numeric(
            df["frp"],
            errors="coerce",
        ).dropna()

    else:
        frp = pd.Series(
            dtype=float
        )

    c2.metric(
        "Maximum FRP",
        (
            f"{frp.max():.1f} MW"
            if not frp.empty
            else "N/A"
        ),
    )

    if "confidence" in df.columns:

        confidence_count = int(
            df["confidence"]
            .notna()
            .sum()
        )

    else:
        confidence_count = 0

    c3.metric(
        "Confidence Values",
        confidence_count,
    )

    c4.metric(
        "Data Source",
        "NASA FIRMS",
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    st.divider()

    _render_map(
        df
    )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🛰 Real Satellite Observations"
    )

    display_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "frp",
            "confidence",
            "brightness",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
        ]
        if column in df.columns
    ]

    if display_columns:

        st.dataframe(
            df[display_columns],
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Raw data
    # --------------------------------------------------------

    with st.expander(
        "Raw NASA FIRMS Observations"
    ):

        st.json(
            observations[:50]
        )

    # --------------------------------------------------------
    # Database history
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🗄 NASA Database History"
    )

    history = external_observations()

    if history.empty:

        st.caption(
            "No external observations have been "
            "recorded in the database yet."
        )

    else:

        # Only show records that actually contain NASA
        # in their source/type fields.
        nasa_history = history[
            history.astype(str)
            .apply(
                lambda row: row.str.contains(
                    "NASA",
                    case=False,
                    na=False,
                ).any(),
                axis=1,
            )
        ]

        if nasa_history.empty:

            st.caption(
                "Database contains external observations, "
                "but no NASA-labeled records were found."
            )

        else:

            st.dataframe(
                nasa_history.head(100),
                width="stretch",
                hide_index=True,
            )