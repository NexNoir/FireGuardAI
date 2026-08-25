from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATABASE
# ============================================================

try:
    from database.db import FireGuardDatabase
except ImportError:
    FireGuardDatabase = None


# ============================================================
# PRODUCTION SERVICE
# ============================================================

try:
    from real_firms_production_service import (
        RealFirmsProductionService,
    )
except ImportError:
    RealFirmsProductionService = None


# ============================================================
# PREDICTION METADATA
# ============================================================

MODEL_VERSION = "real-firms-production-v1"
FEATURE_VERSION = "real-firms-production-features-v1"


# ============================================================
# PAGE
# ============================================================

def render() -> None:
    st.title("🌲 Seasonal Forecast")
    st.caption(
        "Real FIRMS regional fire-risk prediction for the Hyrcanian Forest"
    )

    st.divider()

    # --------------------------------------------------------
    # PRODUCTION SERVICE CHECK
    # --------------------------------------------------------

    if RealFirmsProductionService is None:
        st.error(
            "Production service could not be imported: "
            "real_firms_production_service.py"
        )
        return

    # --------------------------------------------------------
    # DATABASE CHECK
    # --------------------------------------------------------

    if FireGuardDatabase is None:
        st.error(
            "Database module could not be imported: "
            "database.db"
        )
        return

    # --------------------------------------------------------
    # SERVICE
    # --------------------------------------------------------

    try:
        service = RealFirmsProductionService()
    except Exception as exc:
        st.error("Production service initialization failed.")
        st.exception(exc)
        return

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:
        db = FireGuardDatabase()
    except Exception as exc:
        st.error("Database initialization failed.")
        st.exception(exc)
        return

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "24H Threshold",
        f"{service.thresholds['24h']:.2f}",
    )

    col2.metric(
        "48H Threshold",
        f"{service.thresholds['48h']:.2f}",
    )

    col3.metric(
        "72H Threshold",
        f"{service.thresholds['72h']:.2f}",
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    st.subheader("📍 FIRMS Input")

    c1, c2 = st.columns(2)

    with c1:
        latitude = st.number_input(
            "Latitude",
            value=37.200000,
            format="%.6f",
        )

        longitude = st.number_input(
            "Longitude",
            value=50.000000,
            format="%.6f",
        )

        brightness = st.number_input(
            "Brightness",
            value=330.0,
            min_value=0.0,
        )

        bright_t31 = st.number_input(
            "Bright T31",
            value=300.0,
            min_value=0.0,
        )

        frp = st.number_input(
            "FRP",
            value=10.0,
            min_value=0.0,
        )

    with c2:
        scan = st.number_input(
            "Scan",
            value=1.0,
            min_value=0.0,
        )

        track = st.number_input(
            "Track",
            value=1.0,
            min_value=0.0,
        )

        confidence = st.number_input(
            "Confidence",
            value=80.0,
            min_value=0.0,
            max_value=100.0,
        )

        hour = st.number_input(
            "Hour",
            value=12,
            min_value=0,
            max_value=23,
            step=1,
        )

        minute = st.number_input(
            "Minute",
            value=0,
            min_value=0,
            max_value=59,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # CATEGORICAL FIRMS DATA
    # --------------------------------------------------------

    st.subheader("🛰️ FIRMS Metadata")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        daynight = st.selectbox(
            "Day / Night",
            ["D", "N"],
        )

    with c2:
        satellite = st.text_input(
            "Satellite",
            value="N",
        )

    with c3:
        instrument = st.text_input(
            "Instrument",
            value="MODIS",
        )

    with c4:
        fire_type = st.text_input(
            "Type",
            value="0",
        )

    season = st.selectbox(
        "Season",
        ["winter", "spring", "summer", "autumn"],
        index=2,
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🔥 Run Seasonal Forecast",
        type="primary",
        use_container_width=True,
    ):
        record = {
            "latitude": latitude,
            "longitude": longitude,
            "brightness": brightness,
            "scan": scan,
            "track": track,
            "confidence": confidence,
            "bright_t31": bright_t31,
            "frp": frp,
            "hour": int(hour),
            "minute": int(minute),
            "daynight": daynight,
            "satellite": satellite,
            "instrument": instrument,
            "type": fire_type,
            "season": season,
        }

        # ----------------------------------------------------
        # RUN REAL PRODUCTION INFERENCE
        # ----------------------------------------------------

        try:
            result = service.predict_record(record)

        except AttributeError:
            try:
                result = service.predict(
                    pd.DataFrame([record])
                )
            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)
                return

        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)
            return

        # ----------------------------------------------------
        # NORMALIZE RESULT
        # ----------------------------------------------------

        if isinstance(result, pd.DataFrame):

            if result.empty:
                st.error(
                    "The production service returned no prediction."
                )
                return

            row = result.iloc[0].to_dict()

        elif isinstance(result, dict):

            row = result

        else:

            st.error(
                "Unsupported prediction result returned "
                "by production service."
            )
            return

        # ----------------------------------------------------
        # EXTRACT REAL PROBABILITIES
        # ----------------------------------------------------

        try:

            p24 = float(
                row.get(
                    "probability_24h",
                    row.get(
                        "prob_24h",
                        row.get(
                            "24h_probability",
                            0.0,
                        ),
                    ),
                )
            )

            p48 = float(
                row.get(
                    "probability_48h",
                    row.get(
                        "prob_48h",
                        row.get(
                            "48h_probability",
                            0.0,
                        ),
                    ),
                )
            )

            p72 = float(
                row.get(
                    "probability_72h",
                    row.get(
                        "prob_72h",
                        row.get(
                            "72h_probability",
                            0.0,
                        ),
                    ),
                )
            )

        except (TypeError, ValueError) as exc:

            st.error(
                "Production prediction probabilities "
                "could not be parsed."
            )
            st.exception(exc)
            return

        # ----------------------------------------------------
        # VALIDATE PROBABILITIES
        # ----------------------------------------------------

        probabilities = {
            24: p24,
            48: p48,
            72: p72,
        }

        for horizon, probability in probabilities.items():

            if not 0.0 <= probability <= 1.0:

                st.error(
                    f"Invalid {horizon}h prediction probability: "
                    f"{probability}"
                )
                return

        # ----------------------------------------------------
        # REGISTER REAL PREDICTIONS IN DATABASE
        # ----------------------------------------------------

        prediction_timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        try:

            prediction_ids = []

            for horizon, probability in probabilities.items():

                prediction_id = db.add_prediction(
                    timestamp=prediction_timestamp,
                    model_version=MODEL_VERSION,
                    feature_version=FEATURE_VERSION,
                    probability=probability,
                    uncertainty=None,
                    horizon=horizon,
                )

                prediction_ids.append(prediction_id)

        except Exception as exc:

            st.error(
                "Prediction succeeded, but database registration failed."
            )
            st.exception(exc)
            return

        # ----------------------------------------------------
        # DATABASE CONFIRMATION
        # ----------------------------------------------------

        st.success(
            "Production prediction completed and "
            "registered in the database."
        )

        st.caption(
            f"Prediction IDs: {', '.join(map(str, prediction_ids))}"
        )

        # ----------------------------------------------------
        # RESULT CARDS
        # ----------------------------------------------------

        st.subheader("🔥 Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:

            if p24 >= service.thresholds["24h"]:

                st.error(
                    f"### 24 Hours\n"
                    f"## {p24 * 100:.1f}%\n"
                    f"🔥 HIGH RISK"
                )

            else:

                st.success(
                    f"### 24 Hours\n"
                    f"## {p24 * 100:.1f}%\n"
                    f"🟢 BELOW THRESHOLD"
                )

        with c2:

            if p48 >= service.thresholds["48h"]:

                st.error(
                    f"### 48 Hours\n"
                    f"## {p48 * 100:.1f}%\n"
                    f"🔥 HIGH RISK"
                )

            else:

                st.success(
                    f"### 48 Hours\n"
                    f"## {p48 * 100:.1f}%\n"
                    f"🟢 BELOW THRESHOLD"
                )

        with c3:

            if p72 >= service.thresholds["72h"]:

                st.error(
                    f"### 72 Hours\n"
                    f"## {p72 * 100:.1f}%\n"
                    f"🔥 HIGH RISK"
                )

            else:

                st.success(
                    f"### 72 Hours\n"
                    f"## {p72 * 100:.1f}%\n"
                    f"🟢 BELOW THRESHOLD"
                )

        # ----------------------------------------------------
        # PROBABILITY TREND
        # ----------------------------------------------------

        st.divider()

        st.subheader("📈 Probability Trend")

        chart_data = pd.DataFrame(
            {
                "Forecast Horizon": [
                    "24 Hours",
                    "48 Hours",
                    "72 Hours",
                ],
                "Probability": [
                    p24 * 100,
                    p48 * 100,
                    p72 * 100,
                ],
            }
        )

        st.line_chart(
            chart_data.set_index(
                "Forecast Horizon"
            ),
            y="Probability",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        st.divider()

        st.subheader("🤖 Model Information")

        model_info = {
            "Model Version": MODEL_VERSION,
            "Feature Version": FEATURE_VERSION,
            "Feature Count": getattr(
                service,
                "feature_count",
                len(service.feature_columns),
            ),
            "24H Threshold": service.thresholds["24h"],
            "48H Threshold": service.thresholds["48h"],
            "72H Threshold": service.thresholds["72h"],
        }

        st.dataframe(
            pd.DataFrame([model_info]),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Prediction is generated by the existing Real FIRMS "
            "production models. No retraining or model modification "
            "is performed. Successful predictions are persisted "
            "to the FireGuard prediction history database."
        )