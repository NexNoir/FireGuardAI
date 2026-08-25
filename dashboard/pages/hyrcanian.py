from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "hyrcanian"
    / "hyrcanian_firms_fire_history.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "hyrcanian"
    / "hyrcanian_firms_report.json"
)

BOUNDARY_PATH = (
    BASE_DIR
    / "data"
    / "historical"
    / "ecoregions"
    / "PA0407.geojson"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_report():
    if not REPORT_PATH.exists():
        return None

    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False,
    )

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(
            df["acq_date"],
            errors="coerce",
        )

    return df


# ============================================================
# RENDER
# ============================================================

def render():

    st.title(
        "🌲 Hyrcanian Historical Fire Evidence"
    )

    st.caption(
        "Historical regional fire evidence extracted from "
        "NASA FIRMS using the official Caspian Hyrcanian "
        "mixed forests ecoregion boundary."
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    if not REPORT_PATH.exists():

        st.error(
            "Hyrcanian report file is not available."
        )

        st.code(
            str(REPORT_PATH)
        )

        return

    if not DATASET_PATH.exists():

        st.error(
            "Hyrcanian historical dataset is not available."
        )

        st.code(
            str(DATASET_PATH)
        )

        return

    if not BOUNDARY_PATH.exists():

        st.warning(
            "Official ecoregion boundary file is not available."
        )

    # ========================================================
    # LOAD
    # ========================================================

    report = load_report()
    df = load_dataset()

    if report is None:

        st.error(
            "Could not load Hyrcanian report."
        )

        return

    if df.empty:

        st.warning(
            "Hyrcanian dataset is empty."
        )

        return

    spatial = report.get(
        "spatial_filter",
        {},
    )

    source = report.get(
        "source",
        {},
    )

    result = report.get(
        "result",
        {},
    )

    statistics = report.get(
        "statistics",
        {},
    )

    # ========================================================
    # STATUS
    # ========================================================

    st.success(
        "🟢 HISTORICAL HYRCANIAN DATA AVAILABLE"
    )

    # ========================================================
    # MAIN METRICS
    # ========================================================

    st.subheader(
        "📊 Dataset Summary"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Iran FIRMS Records",
        f"{source.get('total_iran_records', 0):,}",
    )

    c2.metric(
        "Hyrcanian Records",
        f"{result.get('hyrcanian_records', len(df)):,}",
    )

    c3.metric(
        "Source Retained",
        f"{result.get('percentage_of_source', 0):.4f}%",
    )

    c4.metric(
        "Ecoregion ID",
        str(
            spatial.get(
                "ecoregion_id",
                "N/A",
            )
        ),
    )

    # ========================================================
    # ECOREGION INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🗺️ Spatial Evidence"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Ecoregion:**",
            spatial.get(
                "ecoregion_name",
                "N/A",
            ),
        )

        st.write(
            "**Spatial filter:**",
            spatial.get(
                "method",
                "N/A",
            ),
        )

    with col2:

        st.write(
            "**Boundary file:**"
        )

        st.code(
            spatial.get(
                "boundary_file",
                str(BOUNDARY_PATH),
            )
        )

        st.write(
            "**Boundary available:**",
            "YES"
            if BOUNDARY_PATH.exists()
            else "NO",
        )

    # ========================================================
    # SOURCE INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🛰️ Source and Time Coverage"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Source",
        report.get(
            "dataset",
            "N/A",
        ),
    )

    c2.metric(
        "Start Date",
        result.get(
            "start_date",
            "N/A",
        ),
    )

    c3.metric(
        "End Date",
        result.get(
            "end_date",
            "N/A",
        ),
    )

    st.caption(
        "Source file: "
        + str(
            source.get(
                "file",
                "N/A",
            )
        )
    )

    # ========================================================
    # SCIENTIFIC NOTE
    # ========================================================

    st.info(
        "This page presents historical regional fire evidence. "
        "The extracted FIRMS records are spatially filtered "
        "inside the official Caspian Hyrcanian mixed forests "
        "ecoregion. These records are not automatically treated "
        "as direct training rows for the ESP32 sensor ML model."
    )

    # ========================================================
    # YEARLY ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "📅 Historical Fire Records by Year"
    )

    yearly_counts = statistics.get(
        "yearly_counts",
        {},
    )

    if yearly_counts:

        yearly_df = pd.DataFrame(
            {
                "year": [
                    int(year)
                    for year in yearly_counts.keys()
                ],
                "fire_records": [
                    int(count)
                    for count in yearly_counts.values()
                ],
            }
        ).sort_values(
            "year"
        )

        st.bar_chart(
            yearly_df.set_index(
                "year"
            )
        )

        st.dataframe(
            yearly_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.warning(
            "Yearly statistics are not available."
        )

    # ========================================================
    # MONTHLY ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "📆 Historical Fire Records by Month"
    )

    monthly_counts = statistics.get(
        "monthly_counts",
        {},
    )

    if monthly_counts:

        month_names = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December",
        }

        monthly_rows = []

        for month in range(1, 13):

            monthly_rows.append(
                {
                    "month_number": month,
                    "month": month_names[month],
                    "fire_records": int(
                        monthly_counts.get(
                            str(month),
                            0,
                        )
                    ),
                }
            )

        monthly_df = pd.DataFrame(
            monthly_rows
        )

        st.bar_chart(
            monthly_df.set_index(
                "month"
            )[["fire_records"]]
        )

        st.dataframe(
            monthly_df,
            width="stretch",
            hide_index=True,
        )

    else:

        st.warning(
            "Monthly statistics are not available."
        )

    # ========================================================
    # SEASONAL ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🌦️ Seasonal Fire Pattern"
    )

    seasonal_counts = statistics.get(
        "seasonal_counts",
        {},
    )

    if seasonal_counts:

        season_order = [
            "spring",
            "summer",
            "autumn",
            "winter",
        ]

        seasonal_rows = []

        total = sum(
            int(value)
            for value in seasonal_counts.values()
        )

        for season in season_order:

            count = int(
                seasonal_counts.get(
                    season,
                    0,
                )
            )

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            seasonal_rows.append(
                {
                    "season": season.title(),
                    "fire_records": count,
                    "percentage": round(
                        percentage,
                        2,
                    ),
                }
            )

        seasonal_df = pd.DataFrame(
            seasonal_rows
        )

        st.bar_chart(
            seasonal_df.set_index(
                "season"
            )[["fire_records"]]
        )

        st.dataframe(
            seasonal_df,
            width="stretch",
            hide_index=True,
        )

        summer_row = seasonal_df[
            seasonal_df["season"] == "Summer"
        ]

        if not summer_row.empty:

            summer_count = int(
                summer_row.iloc[0][
                    "fire_records"
                ]
            )

            summer_percentage = float(
                summer_row.iloc[0][
                    "percentage"
                ]
            )

            st.success(
                "Summer has the highest number of "
                f"historical records: {summer_count:,} "
                f"({summer_percentage:.2f}%)."
            )

    else:

        st.warning(
            "Seasonal statistics are not available."
        )

    # ========================================================
    # FIRMS SIGNAL STATISTICS
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 FIRMS Signal Statistics"
    )

    confidence_stats = statistics.get(
        "confidence",
        {},
    )

    frp_stats = statistics.get(
        "frp",
        {},
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Confidence Mean",
        (
            f"{confidence_stats.get('mean', 0):.2f}"
            if confidence_stats.get("mean") is not None
            else "N/A"
        ),
    )

    c2.metric(
        "Confidence Median",
        (
            f"{confidence_stats.get('median', 0):.2f}"
            if confidence_stats.get("median") is not None
            else "N/A"
        ),
    )

    c3.metric(
        "FRP Mean",
        (
            f"{frp_stats.get('mean', 0):.2f}"
            if frp_stats.get("mean") is not None
            else "N/A"
        ),
    )

    c4.metric(
        "FRP Median",
        (
            f"{frp_stats.get('median', 0):.2f}"
            if frp_stats.get("median") is not None
            else "N/A"
        ),
    )

    c5.metric(
        "FRP Maximum",
        (
            f"{frp_stats.get('max', 0):.2f}"
            if frp_stats.get("max") is not None
            else "N/A"
        ),
    )

    # ========================================================
    # RAW DATA PREVIEW
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Extracted Historical Records"
    )

    preview_columns = [
        column
        for column in [
            "acq_date",
            "latitude",
            "longitude",
            "confidence",
            "frp",
            "brightness",
            "bright_t31",
            "daynight",
            "year",
            "month",
            "season",
        ]
        if column in df.columns
    ]

    st.caption(
        f"Showing {len(df):,} extracted historical "
        "records from the Hyrcanian ecoregion."
    )

    st.dataframe(
        df[
            preview_columns
        ].sort_values(
            "acq_date",
            ascending=False,
        ),
        width="stretch",
        hide_index=True,
        height=500,
    )

    # ========================================================
    # AUDIT INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Reproducibility / Audit"
    )

    st.code(
        "\n".join(
            [
                "Source: NASA FIRMS MODIS C6.1",
                "Spatial method: Exact Point-in-Polygon",
                (
                    "Ecoregion: "
                    + str(
                        spatial.get(
                            "ecoregion_name",
                            "N/A",
                        )
                    )
                ),
                (
                    "Ecoregion ID: "
                    + str(
                        spatial.get(
                            "ecoregion_id",
                            "N/A",
                        )
                    )
                ),
                (
                    "Iran source records: "
                    + str(
                        source.get(
                            "total_iran_records",
                            "N/A",
                        )
                    )
                ),
                (
                    "Hyrcanian extracted records: "
                    + str(
                        result.get(
                            "hyrcanian_records",
                            "N/A",
                        )
                    )
                ),
                (
                    "Date range: "
                    + str(
                        result.get(
                            "start_date",
                            "N/A",
                        )
                    )
                    + " to "
                    + str(
                        result.get(
                            "end_date",
                            "N/A",
                        )
                    )
                ),
            ]
        )
    )
