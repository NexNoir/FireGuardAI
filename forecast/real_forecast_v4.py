from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FireGuardForecastV4:
    """
    FireGuard Forecast V4 — Production Inference

    Responsibilities
    ----------------
    - Load existing V4 model artifacts.
    - Enforce the exact training feature schema.
    - Reproduce the V4 feature-engineering logic.
    - Apply the stored calibrator.
    - Apply the stored threshold.
    - Produce 24h / 48h / 72h forecasts.
    - Support production inference from REAL chronological sensor history.

    Safety rules
    ------------
    - No retraining.
    - No dataset modification.
    - No V3 overwrite.
    - No synthetic sensor fallback in production history inference.
    - No future observations in feature construction.
    - Model probability is returned read-only to downstream AlertEngine.
    """

    # ==========================================================
    # SINGLE SOURCE OF TRUTH — FEATURE SCHEMA
    # Must match the training schema exactly.
    # ==========================================================

    SENSOR_FEATURES = [
        "temperature",
        "humidity",
        "smoke",
        "hour",
        "minute",

        "smoke_change_1m",
        "smoke_change_5m",
        "smoke_change_15m",
        "smoke_change_30m",
        "smoke_change_60m",

        "temperature_change_5m",
        "temperature_change_15m",
        "temperature_change_30m",
        "temperature_change_60m",

        "humidity_change_5m",
        "humidity_change_15m",
        "humidity_change_30m",

        "smoke_mean_5m",
        "smoke_mean_15m",
        "smoke_mean_30m",
        "smoke_mean_60m",

        "smoke_std_15m",
        "smoke_std_30m",
        "smoke_max_30m",

        "temperature_mean_15m",
        "temperature_mean_30m",

        "humidity_mean_15m",
        "humidity_mean_30m",
    ]

    SENSOR_PLUS_FLAME_FEATURES = SENSOR_FEATURES + [
        "flame"
    ]

    # ----------------------------------------------------------
    # Fallback thresholds.
    # Used only when an artifact lacks threshold metadata.
    # Artifact threshold remains the preferred source.
    # ----------------------------------------------------------

    FALLBACK_THRESHOLDS = {
        "sensor_only": {
            "24h": 0.33,
            "48h": 0.20,
            "72h": 0.41,
        },
        "sensor_plus_flame": {
            "24h": 0.32,
            "48h": 0.20,
            "72h": 0.41,
        },
    }

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(
        self,
        experiment: str = "sensor_only",
    ):
        self.base_dir = (
            Path(__file__).resolve().parent.parent
        )

        self.data_dir = (
            self.base_dir / "data"
        )

        self.model_dir = (
            self.base_dir / "saved_models"
        )

        self.dataset_path = (
            self.data_dir
            / "fireguard_forecast_60days.csv"
        )

        self.horizons = [
            "24h",
            "48h",
            "72h",
        ]

        self.feature_sets = {
            "sensor_only": list(
                self.SENSOR_FEATURES
            ),
            "sensor_plus_flame": list(
                self.SENSOR_PLUS_FLAME_FEATURES
            ),
        }

        self.models = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.active_experiment = (
            "sensor_only"
        )

        self.feature_columns = list(
            self.feature_sets[
                self.active_experiment
            ]
        )

        self.set_experiment(
            experiment
        )

    # ==========================================================
    # VALIDATE EXPERIMENT
    # ==========================================================

    @staticmethod
    def _validate_experiment(
        experiment: str,
    ) -> None:

        if experiment not in {
            "sensor_only",
            "sensor_plus_flame",
        }:
            raise ValueError(
                "experiment must be "
                "'sensor_only' or "
                "'sensor_plus_flame'"
            )

    # ==========================================================
    # ARTIFACT INSPECTION — READ ONLY
    # ==========================================================

    def inspect_v4_artifacts(self):
        """
        Inspect all six V4 artifacts without modification.
        """

        print()
        print("=" * 70)
        print(
            "MODEL INSPECTION "
            "(V4 artifacts – read-only)"
        )
        print("=" * 70)

        results = {}

        for experiment in (
            "sensor_only",
            "sensor_plus_flame",
        ):

            expected = list(
                self.feature_sets[
                    experiment
                ]
            )

            for horizon in self.horizons:

                filename = (
                    f"fireguard_forecast_"
                    f"{experiment}_{horizon}_v4.joblib"
                )

                path = (
                    self.model_dir
                    / filename
                )

                key = (
                    f"{experiment}_{horizon}"
                )

                print()
                print(
                    f"{experiment} / {horizon}"
                )
                print(
                    f"File: {filename}"
                )

                if not path.exists():

                    print(
                        "Status: FILE NOT FOUND"
                    )

                    results[key] = {
                        "exists": False,
                        "load_ok": False,
                        "schema_pass": False,
                    }

                    continue

                try:

                    payload = joblib.load(
                        path
                    )

                except Exception as exc:

                    print(
                        f"Status: LOAD FAILED - {exc}"
                    )

                    results[key] = {
                        "exists": True,
                        "load_ok": False,
                        "schema_pass": False,
                    }

                    continue

                info = {
                    "exists": True,
                    "load_ok": True,
                    "object_type": type(
                        payload
                    ).__name__,
                }

                if not isinstance(
                    payload,
                    dict,
                ):

                    info.update(
                        {
                            "schema_pass": False,
                            "calibrator": False,
                            "threshold": None,
                        }
                    )

                    print(
                        "Object type : "
                        f"{type(payload).__name__}"
                        " (legacy estimator)"
                    )

                    results[key] = info
                    continue

                keys = list(
                    payload.keys()
                )

                info["keys"] = keys

                model = payload.get(
                    "model"
                )

                calibrator = payload.get(
                    "calibrator"
                )

                threshold = payload.get(
                    "threshold"
                )

                features = (
                    payload.get("features")
                    or payload.get(
                        "feature_names"
                    )
                )

                version = payload.get(
                    "version"
                )

                calibration_method = (
                    payload.get(
                        "calibration_method"
                    )
                )

                if model is not None:

                    info[
                        "model_type"
                    ] = type(
                        model
                    ).__name__

                    info[
                        "n_features_in_"
                    ] = getattr(
                        model,
                        "n_features_in_",
                        None,
                    )

                if calibrator is not None:

                    info["calibrator"] = True

                    info[
                        "calibrator_type"
                    ] = type(
                        calibrator
                    ).__name__

                else:

                    info["calibrator"] = False

                if threshold is not None:

                    try:

                        info[
                            "threshold"
                        ] = float(
                            threshold
                        )

                    except Exception:

                        info[
                            "threshold"
                        ] = None

                else:

                    info[
                        "threshold"
                    ] = None

                if features is not None:

                    features = list(
                        features
                    )

                    info[
                        "features"
                    ] = features

                    info[
                        "feature_count"
                    ] = len(
                        features
                    )

                    order_match = (
                        features
                        == expected
                    )

                    count_match = (
                        len(features)
                        == len(expected)
                    )

                    info[
                        "schema_pass"
                    ] = (
                        order_match
                        and count_match
                    )

                else:

                    info[
                        "features"
                    ] = None

                    info[
                        "feature_count"
                    ] = 0

                    info[
                        "schema_pass"
                    ] = False

                if version:
                    info[
                        "version"
                    ] = version

                if calibration_method:
                    info[
                        "calibration_method"
                    ] = calibration_method

                results[key] = info

        return results

    # ==========================================================
    # LOAD V4 MODELS
    # ==========================================================

    def load_models(
        self,
        experiment: Optional[str] = None,
    ):

        experiments = (
            [experiment]
            if experiment
            else [
                "sensor_only",
                "sensor_plus_flame",
            ]
        )

        for exp in experiments:

            self._validate_experiment(
                exp
            )

            loaded = {}

            expected = list(
                self.feature_sets[exp]
            )

            for horizon in self.horizons:

                filename = (
                    f"fireguard_forecast_"
                    f"{exp}_{horizon}_v4.joblib"
                )

                path = (
                    self.model_dir
                    / filename
                )

                if not path.exists():
                    continue

                try:

                    payload = joblib.load(
                        path
                    )

                except Exception as exc:

                    print(
                        f"Failed to load "
                        f"{filename}: {exc}"
                    )

                    continue

                if not isinstance(
                    payload,
                    dict,
                ):

                    print(
                        f"Unexpected payload "
                        f"format: {filename}"
                    )

                    continue

                if "model" not in payload:

                    print(
                        f"Model missing in "
                        f"{filename}"
                    )

                    continue

                model = payload[
                    "model"
                ]

                calibrator = payload.get(
                    "calibrator"
                )

                if calibrator is None:

                    raise RuntimeError(
                        "Calibrator missing in "
                        f"V4 artifact: {filename}"
                    )

                threshold = payload.get(
                    "threshold"
                )

                if threshold is None:

                    threshold = (
                        self.FALLBACK_THRESHOLDS[
                            exp
                        ][
                            horizon
                        ]
                    )

                threshold = float(
                    threshold
                )

                features = (
                    payload.get("features")
                    or payload.get(
                        "feature_names"
                    )
                )

                if features is None:

                    features = expected

                else:

                    features = list(
                        features
                    )

                if (
                    list(features)
                    != expected
                ):

                    raise RuntimeError(
                        "SCHEMA ERROR in "
                        f"{filename}\n"
                        f"Expected: {expected}\n"
                        f"Got: {features}"
                    )

                loaded[horizon] = {
                    "model": model,
                    "calibrator": calibrator,
                    "threshold": threshold,
                    "features": features,
                    "version": payload.get(
                        "version",
                        "v4",
                    ),
                    "calibration_method": (
                        payload.get(
                            "calibration_method",
                            "sigmoid",
                        )
                    ),
                }

            self.models[exp] = loaded

        return True

    # ==========================================================
    # TRAINING-EQUIVALENT FEATURE ENGINEERING
    # ==========================================================

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        df["timestamp"] = (
            pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )
        )

        df = (
            df.dropna(
                subset=["timestamp"]
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ------------------------------------------------------
        # Required raw columns
        # ------------------------------------------------------

        required = [
            "temperature",
            "humidity",
            "smoke",
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing required raw "
                f"sensor columns: {missing}"
            )

        # Strict numeric conversion
        for col in required:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = df.dropna(
            subset=required
        ).reset_index(
            drop=True
        )

        # ------------------------------------------------------
        # Flame
        # ------------------------------------------------------

        if "flame" not in df.columns:
            df["flame"] = 0

        df["flame"] = pd.to_numeric(
            df["flame"],
            errors="coerce",
        )

        if df["flame"].isna().any():
            raise ValueError(
                "Invalid flame values "
                "in live history"
            )

        # ------------------------------------------------------
        # Time
        # ------------------------------------------------------

        df["hour"] = (
            df["timestamp"]
            .dt.hour
        )

        df["minute"] = (
            df["timestamp"]
            .dt.minute
        )

        # ------------------------------------------------------
        # Changes
        # ------------------------------------------------------

        df[
            "smoke_change_1m"
        ] = (
            df["smoke"]
            - df["smoke"].shift(1)
        )

        df[
            "smoke_change_5m"
        ] = (
            df["smoke"]
            - df["smoke"].shift(5)
        )

        df[
            "smoke_change_15m"
        ] = (
            df["smoke"]
            - df["smoke"].shift(15)
        )

        df[
            "smoke_change_30m"
        ] = (
            df["smoke"]
            - df["smoke"].shift(30)
        )

        df[
            "smoke_change_60m"
        ] = (
            df["smoke"]
            - df["smoke"].shift(60)
        )

        df[
            "temperature_change_5m"
        ] = (
            df["temperature"]
            - df["temperature"].shift(5)
        )

        df[
            "temperature_change_15m"
        ] = (
            df["temperature"]
            - df["temperature"].shift(15)
        )

        df[
            "temperature_change_30m"
        ] = (
            df["temperature"]
            - df["temperature"].shift(30)
        )

        df[
            "temperature_change_60m"
        ] = (
            df["temperature"]
            - df["temperature"].shift(60)
        )

        df[
            "humidity_change_5m"
        ] = (
            df["humidity"]
            - df["humidity"].shift(5)
        )

        df[
            "humidity_change_15m"
        ] = (
            df["humidity"]
            - df["humidity"].shift(15)
        )

        df[
            "humidity_change_30m"
        ] = (
            df["humidity"]
            - df["humidity"].shift(30)
        )

        # ------------------------------------------------------
        # Rolling
        # ------------------------------------------------------

        df[
            "smoke_mean_5m"
        ] = (
            df["smoke"]
            .rolling(5)
            .mean()
        )

        df[
            "smoke_mean_15m"
        ] = (
            df["smoke"]
            .rolling(15)
            .mean()
        )

        df[
            "smoke_mean_30m"
        ] = (
            df["smoke"]
            .rolling(30)
            .mean()
        )

        df[
            "smoke_mean_60m"
        ] = (
            df["smoke"]
            .rolling(60)
            .mean()
        )

        df[
            "smoke_std_15m"
        ] = (
            df["smoke"]
            .rolling(15)
            .std()
        )

        df[
            "smoke_std_30m"
        ] = (
            df["smoke"]
            .rolling(30)
            .std()
        )

        df[
            "smoke_max_30m"
        ] = (
            df["smoke"]
            .rolling(30)
            .max()
        )

        df[
            "temperature_mean_15m"
        ] = (
            df["temperature"]
            .rolling(15)
            .mean()
        )

        df[
            "temperature_mean_30m"
        ] = (
            df["temperature"]
            .rolling(30)
            .mean()
        )

        df[
            "humidity_mean_15m"
        ] = (
            df["humidity"]
            .rolling(15)
            .mean()
        )

        df[
            "humidity_mean_30m"
        ] = (
            df["humidity"]
            .rolling(30)
            .mean()
        )

        return df

    # ==========================================================
    # REAL LIVE HISTORY → CURRENT FEATURE VECTOR
    # ==========================================================

    def build_live_features(
        self,
        history: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build a production inference row from REAL sensor history.

        IMPORTANT:
        No synthetic defaults are allowed.

        If the history does not contain enough real rows to
        calculate the exact V4 schema, inference fails explicitly.
        """

        if not isinstance(
            history,
            list,
        ):

            raise TypeError(
                "history must be a list"
            )

        if not history:

            raise ValueError(
                "No real sensor history available"
            )

        df = pd.DataFrame(
            history
        )

        required = {
            "timestamp",
            "temperature",
            "humidity",
            "smoke",
        }

        missing = sorted(
            required
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                "Missing real sensor fields: "
                f"{missing}"
            )

        if "flame" not in df.columns:
            df["flame"] = 0

        prepared = (
            self._prepare_features(
                df
            )
        )

        if prepared.empty:

            raise ValueError(
                "No valid feature rows after "
                "real-sensor validation"
            )

        latest = (
            prepared.iloc[-1]
        )

        expected = list(
            self.feature_sets[
                self.active_experiment
            ]
        )

        missing_features = []
        invalid_features = []

        for feature in expected:

            if feature not in latest.index:

                missing_features.append(
                    feature
                )

                continue

            value = latest[
                feature
            ]

            if pd.isna(value):

                invalid_features.append(
                    feature
                )

        if missing_features:

            raise RuntimeError(
                "SCHEMA ERROR — missing features: "
                f"{missing_features}"
            )

        if invalid_features:

            raise ValueError(
                "INSUFFICIENT REAL SENSOR HISTORY "
                "for V4 features: "
                f"{invalid_features}"
            )

        row = {
            feature: float(
                latest[feature]
            )
            for feature in expected
        }

        X = pd.DataFrame(
            [row],
            columns=expected,
        )

        if list(X.columns) != expected:

            raise RuntimeError(
                "Feature order mismatch"
            )

        return X

    # ==========================================================
    # LEGACY SINGLE-READ BUILDER
    # ==========================================================

    def _build_current_features(
        self,
        current_reading: dict[str, Any],
    ):
        """
        Compatibility method.

        IMPORTANT:
        This method is not the preferred production path.

        Production should use build_live_features(history)
        because V4 requires historical deltas and rolling windows.
        """

        if not isinstance(
            current_reading,
            dict,
        ):

            raise TypeError(
                "current_reading must be dict"
            )

        # No synthetic production values.
        required = [
            "timestamp",
            "temperature",
            "humidity",
            "smoke",
        ]

        missing = [
            key
            for key in required
            if key not in current_reading
        ]

        if missing:

            raise ValueError(
                "Single-reading inference "
                "does not contain all required "
                f"real fields: {missing}. "
                "Use get_forecast_from_history()."
            )

        return self.build_live_features(
            [current_reading]
        )

    # ==========================================================
    # CALIBRATION
    # ==========================================================

    @staticmethod
    def _apply_calibrator(
        calibrator,
        raw_probability,
    ):

        raw = np.asarray(
            raw_probability,
            dtype=float,
        ).reshape(-1, 1)

        calibrated = (
            calibrator
            .predict_proba(raw)[:, 1]
        )

        return calibrated

    # ==========================================================
    # RISK LEVEL
    # ==========================================================

    @staticmethod
    def _risk_level(
        probability: float,
        threshold: float,
    ) -> str:

        if probability < (
            threshold * 0.50
        ):
            return "LOW"

        if probability < threshold:
            return "MEDIUM"

        if probability < (
            threshold + 0.20
        ):
            return "HIGH"

        return "CRITICAL"

    # ==========================================================
    # FORECAST — INTERNAL CORE
    # ==========================================================

    def _forecast_from_features(
        self,
        X: pd.DataFrame,
    ) -> dict:

        experiment = (
            self.active_experiment
        )

        if not self.models.get(
            experiment
        ):

            self.load_models(
                experiment
            )

        if not self.models.get(
            experiment
        ):

            return {
                "success": False,
                "error": (
                    "V4 models not loaded "
                    f"for {experiment}"
                ),
                "forecast": None,
            }

        forecast = {}

        for horizon in self.horizons:

            entry = (
                self.models[
                    experiment
                ].get(horizon)
            )

            if entry is None:

                forecast[horizon] = {
                    "error": (
                        "model not loaded"
                    )
                }

                continue

            expected = list(
                entry["features"]
            )

            missing = [
                feature
                for feature in expected
                if feature not in X.columns
            ]

            if missing:

                return {
                    "success": False,
                    "error": (
                        "SCHEMA ERROR — missing "
                        f"features for {horizon}: "
                        f"{missing}"
                    ),
                    "forecast": None,
                }

            X_inf = X[
                expected
            ].copy()

            if list(
                X_inf.columns
            ) != expected:

                return {
                    "success": False,
                    "error": (
                        "Feature order mismatch "
                        f"for {horizon}"
                    ),
                    "forecast": None,
                }

            # --------------------------------------------------
            # Raw model probability
            # --------------------------------------------------

            raw_probability = float(
                entry["model"]
                .predict_proba(
                    X_inf
                )[0][1]
            )

            raw_probability = float(
                np.clip(
                    raw_probability,
                    0.0,
                    1.0,
                )
            )

            # --------------------------------------------------
            # Calibrated probability
            # --------------------------------------------------

            calibrated_probability = float(
                self._apply_calibrator(
                    entry["calibrator"],
                    [
                        raw_probability
                    ],
                )[0]
            )

            calibrated_probability = float(
                np.clip(
                    calibrated_probability,
                    0.0,
                    1.0,
                )
            )

            threshold = float(
                entry["threshold"]
            )

            prediction = int(
                calibrated_probability
                >= threshold
            )

            risk = (
                self._risk_level(
                    calibrated_probability,
                    threshold,
                )
            )

            forecast[horizon] = {
                "raw_probability": round(
                    raw_probability,
                    6,
                ),
                "raw_probability_percent": round(
                    raw_probability * 100,
                    2,
                ),
                "calibrated_probability": round(
                    calibrated_probability,
                    6,
                ),
                "calibrated_probability_percent": round(
                    calibrated_probability * 100,
                    2,
                ),
                "fire_probability": round(
                    calibrated_probability,
                    6,
                ),
                "fire_probability_percent": round(
                    calibrated_probability * 100,
                    2,
                ),
                "threshold": round(
                    threshold,
                    6,
                ),
                "prediction": prediction,
                "prediction_label": (
                    "FIRE"
                    if prediction == 1
                    else "NO FIRE"
                ),
                "risk_level": risk,
                "calibration_method": (
                    entry.get(
                        "calibration_method",
                        "sigmoid",
                    )
                ),
                "model_version": entry.get(
                    "version",
                    "v4",
                ),
                "feature_version": (
                    "V4_SENSOR_FEATURE_SCHEMA"
                ),
            }

        return {
            "success": True,
            "experiment": experiment,
            "forecast": forecast,
        }

    # ==========================================================
    # FORECAST FROM REAL HISTORY
    # ==========================================================

    def get_forecast_from_history(
        self,
        history: list[dict[str, Any]],
    ) -> dict:

        try:

            X = self.build_live_features(
                history
            )

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
                "forecast": None,
            }

        return self._forecast_from_features(
            X
        )

    # ==========================================================
    # LEGACY PUBLIC FORECAST
    # ==========================================================

    def get_forecast(
        self,
        current_reading,
    ):

        """
        Compatibility wrapper.

        For production, prefer:

            get_forecast_from_history(history)

        because V4 contains 60-minute rolling/delta features.
        """

        try:

            X = self._build_current_features(
                current_reading
            )

        except Exception as exc:

            return {
                "success": False,
                "error": str(exc),
                "forecast": None,
            }

        return self._forecast_from_features(
            X
        )

    # ==========================================================
    # SET EXPERIMENT
    # ==========================================================

    def set_experiment(
        self,
        experiment: str,
    ):

        self._validate_experiment(
            experiment
        )

        self.active_experiment = (
            experiment
        )

        self.feature_columns = list(
            self.feature_sets[
                experiment
            ]
        )

        self.load_models(
            experiment
        )

    # ==========================================================
    # V4 INFERENCE AUDIT
    # ==========================================================

    def run_inference_audit(
        self,
    ) -> bool:

        print()
        print("=" * 70)
        print(
            "FireGuard Forecast V4 "
            "— Inference Audit"
        )
        print("=" * 70)

        inspection = (
            self.inspect_v4_artifacts()
        )

        all_pass = True
        results = {}

        # ------------------------------------------------------
        # Strict artifact checks
        # ------------------------------------------------------

        for experiment in (
            "sensor_only",
            "sensor_plus_flame",
        ):

            for horizon in self.horizons:

                key = (
                    f"{experiment}_{horizon}"
                )

                info = inspection.get(
                    key,
                    {},
                )

                schema_ok = bool(
                    info.get(
                        "schema_pass",
                        False,
                    )
                )

                load_ok = bool(
                    info.get(
                        "load_ok",
                        False,
                    )
                )

                calibrator_ok = bool(
                    info.get(
                        "calibrator",
                        False,
                    )
                )

                threshold_ok = (
                    info.get(
                        "threshold"
                    )
                    is not None
                )

                passed = (
                    schema_ok
                    and load_ok
                    and calibrator_ok
                    and threshold_ok
                )

                results[key] = {
                    "schema": schema_ok,
                    "load": load_ok,
                    "calibrator": calibrator_ok,
                    "threshold": threshold_ok,
                    "pass": passed,
                }

                if not passed:
                    all_pass = False

        # ------------------------------------------------------
        # Final report
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print(
            "V4 ARTIFACT AUDIT RESULT"
        )
        print("=" * 70)

        for experiment in (
            "sensor_only",
            "sensor_plus_flame",
        ):

            for horizon in self.horizons:

                key = (
                    f"{experiment}_{horizon}"
                )

                result = results[key]

                print(
                    f"{experiment.upper()} / {horizon}"
                )

                print(
                    "  Load       : "
                    f"{'PASS' if result['load'] else 'FAIL'}"
                )

                print(
                    "  Schema     : "
                    f"{'PASS' if result['schema'] else 'FAIL'}"
                )

                print(
                    "  Calibrator : "
                    f"{'PASS' if result['calibrator'] else 'FAIL'}"
                )

                print(
                    "  Threshold  : "
                    f"{'PASS' if result['threshold'] else 'FAIL'}"
                )

                print()

        print(
            "Dataset modified : NO"
        )

        print(
            "Models retrained : NO"
        )

        print(
            "V3 models modified: NO"
        )

        print(
            "Future feature data intentionally used: NO"
        )

        print()
        print(
            "FINAL STATUS:"
        )

        if all_pass:
            print(
                "PASS"
            )
        else:
            print(
                "FAIL"
            )

        print("=" * 70)

        return all_pass


# ==========================================================
# DIRECT EXECUTION
# ==========================================================

if __name__ == "__main__":

    engine = (
        FireGuardForecastV4(
            experiment="sensor_only"
        )
    )

    success = (
        engine.run_inference_audit()
    )

    if not success:
        raise SystemExit(1)