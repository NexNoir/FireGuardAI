from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


class FireGuardForecast:

    # ==========================================================
    # SINGLE SOURCE OF TRUTH – FEATURE SCHEMA (v3 preserved)
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

    SENSOR_PLUS_FLAME_FEATURES = SENSOR_FEATURES + ["flame"]

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "saved_models"

        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = (
            self.data_dir / "fireguard_forecast_60days.csv"
        )

        self.horizons = ["24h", "48h", "72h"]

        self.feature_sets = {
            "sensor_only": list(self.SENSOR_FEATURES),
            "sensor_plus_flame": list(self.SENSOR_PLUS_FLAME_FEATURES),
        }

        self.models = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.thresholds = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.calibrators = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.metrics = {}

        self.active_experiment = "sensor_only"
        self.is_trained = False

        self.feature_columns = self.feature_sets[
            self.active_experiment
        ]

        self.calibration_method = "sigmoid"

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    def _load_data(self):

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset پیدا نشد:\n{self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])

        df = (
            df
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

        required = [
            "temperature",
            "humidity",
            "smoke",
            "flame",
        ]

        for column in required:

            if column not in df.columns:
                raise ValueError(
                    f"ستون {column} وجود ندارد."
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        print()
        print("=" * 70)
        print("FireGuard Forecast v4 (Calibration Enabled)")
        print("=" * 70)

        print(
            f"Dataset : {self.dataset_path}"
        )

        print(
            f"Records : {len(df):,}"
        )

        print(
            f"Start   : {df['timestamp'].min()}"
        )

        print(
            f"End     : {df['timestamp'].max()}"
        )

        return df

    # ==========================================================
    # FEATURE ENGINEERING (identical to v3)
    # ==========================================================

    def _prepare_features(self, df):

        df = df.copy()

        # ------------------------------------------------------
        # IMPORTANT:
        # day deliberately removed.
        # ------------------------------------------------------

        df["hour"] = (
            df["timestamp"].dt.hour
        )

        df["minute"] = (
            df["timestamp"].dt.minute
        )

        # ------------------------------------------------------
        # SMOKE CHANGE
        # ------------------------------------------------------

        df["smoke_change_1m"] = (
            df["smoke"]
            - df["smoke"].shift(1)
        )

        df["smoke_change_5m"] = (
            df["smoke"]
            - df["smoke"].shift(5)
        )

        df["smoke_change_15m"] = (
            df["smoke"]
            - df["smoke"].shift(15)
        )

        df["smoke_change_30m"] = (
            df["smoke"]
            - df["smoke"].shift(30)
        )

        df["smoke_change_60m"] = (
            df["smoke"]
            - df["smoke"].shift(60)
        )

        # ------------------------------------------------------
        # TEMPERATURE CHANGE
        # ------------------------------------------------------

        df["temperature_change_5m"] = (
            df["temperature"]
            - df["temperature"].shift(5)
        )

        df["temperature_change_15m"] = (
            df["temperature"]
            - df["temperature"].shift(15)
        )

        df["temperature_change_30m"] = (
            df["temperature"]
            - df["temperature"].shift(30)
        )

        df["temperature_change_60m"] = (
            df["temperature"]
            - df["temperature"].shift(60)
        )

        # ------------------------------------------------------
        # HUMIDITY CHANGE
        # ------------------------------------------------------

        df["humidity_change_5m"] = (
            df["humidity"]
            - df["humidity"].shift(5)
        )

        df["humidity_change_15m"] = (
            df["humidity"]
            - df["humidity"].shift(15)
        )

        df["humidity_change_30m"] = (
            df["humidity"]
            - df["humidity"].shift(30)
        )

        # ------------------------------------------------------
        # SMOKE ROLLING FEATURES
        # ------------------------------------------------------

        df["smoke_mean_5m"] = (
            df["smoke"]
            .rolling(5)
            .mean()
        )

        df["smoke_mean_15m"] = (
            df["smoke"]
            .rolling(15)
            .mean()
        )

        df["smoke_mean_30m"] = (
            df["smoke"]
            .rolling(30)
            .mean()
        )

        df["smoke_mean_60m"] = (
            df["smoke"]
            .rolling(60)
            .mean()
        )

        df["smoke_std_15m"] = (
            df["smoke"]
            .rolling(15)
            .std()
        )

        df["smoke_std_30m"] = (
            df["smoke"]
            .rolling(30)
            .std()
        )

        df["smoke_max_30m"] = (
            df["smoke"]
            .rolling(30)
            .max()
        )

        # ------------------------------------------------------
        # TEMPERATURE ROLLING FEATURES
        # ------------------------------------------------------

        df["temperature_mean_15m"] = (
            df["temperature"]
            .rolling(15)
            .mean()
        )

        df["temperature_mean_30m"] = (
            df["temperature"]
            .rolling(30)
            .mean()
        )

        # ------------------------------------------------------
        # HUMIDITY ROLLING FEATURES
        # ------------------------------------------------------

        df["humidity_mean_15m"] = (
            df["humidity"]
            .rolling(15)
            .mean()
        )

        df["humidity_mean_30m"] = (
            df["humidity"]
            .rolling(30)
            .mean()
        )

        return df

    # ==========================================================
    # TEMPORAL SPLIT (unchanged 70/15/15)
    # ==========================================================

    @staticmethod
    def _temporal_split(df):

        n = len(df)

        train_end = int(n * 0.70)

        validation_end = int(n * 0.85)

        train = df.iloc[
            :train_end
        ].copy()

        validation = df.iloc[
            train_end:validation_end
        ].copy()

        test = df.iloc[
            validation_end:
        ].copy()

        return train, validation, test

    # ==========================================================
    # METRICS
    # ==========================================================

    @staticmethod
    def _metrics(y_true, y_pred):

        return {
            "accuracy": float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

            "precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

            "f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

            "confusion_matrix":
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1]
                ).tolist()
        }

    @staticmethod
    def _probability_metrics(y_true, probabilities):
        """Brier, LogLoss, PR-AUC, ROC-AUC, ECE"""

        y_true = np.asarray(y_true).astype(int)
        probabilities = np.asarray(probabilities, dtype=float)

        # Clip for numerical safety
        probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)

        result = {}

        try:
            result["brier"] = float(
                brier_score_loss(y_true, probabilities)
            )
        except Exception:
            result["brier"] = None

        try:
            result["log_loss"] = float(
                log_loss(y_true, probabilities)
            )
        except Exception:
            result["log_loss"] = None

        try:
            result["pr_auc"] = float(
                average_precision_score(y_true, probabilities)
            )
        except Exception:
            result["pr_auc"] = None

        try:
            result["roc_auc"] = float(
                roc_auc_score(y_true, probabilities)
            )
        except Exception:
            result["roc_auc"] = None

        result["ece"] = FireGuardForecast._expected_calibration_error(
            y_true, probabilities, n_bins=10
        )

        return result

    @staticmethod
    def _expected_calibration_error(y_true, probabilities, n_bins=10):
        """
        Deterministic ECE with equal-width bins.
        """
        y_true = np.asarray(y_true).astype(int)
        probabilities = np.asarray(probabilities, dtype=float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins[1:-1], right=False)

        ece = 0.0
        total = len(y_true)

        if total == 0:
            return 0.0

        for b in range(n_bins):
            mask = bin_indices == b
            count = np.sum(mask)
            if count == 0:
                continue
            conf = np.mean(probabilities[mask])
            acc = np.mean(y_true[mask])
            ece += (count / total) * abs(acc - conf)

        return float(ece)

    # ==========================================================
    # THRESHOLD TUNING
    # Validation ONLY
    # ==========================================================

    @staticmethod
    def _find_best_threshold(
        y_true,
        probabilities
    ):

        best_threshold = 0.50
        best_f1 = -1

        results = []

        thresholds = np.arange(
            0.20,
            0.81,
            0.01
        )

        for threshold in thresholds:

            prediction = (
                probabilities >= threshold
            ).astype(int)

            metrics = FireGuardForecast._metrics(
                y_true,
                prediction
            )

            results.append({
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            })

            if metrics["f1"] > best_f1:

                best_f1 = metrics["f1"]

                best_threshold = float(
                    threshold
                )

        return (
            best_threshold,
            results
        )

    # ==========================================================
    # CALIBRATION (Validation only – no TEST leakage)
    # ==========================================================

    @staticmethod
    def _fit_calibrator(raw_val_proba, y_val):
        """
        Fit a LogisticRegression calibrator (sigmoid)
        on validation raw probabilities only.
        """
        X_cal = np.asarray(raw_val_proba).reshape(-1, 1)
        y_cal = np.asarray(y_val).astype(int)

        calibrator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        calibrator.fit(X_cal, y_cal)
        return calibrator

    @staticmethod
    def _apply_calibrator(calibrator, raw_proba):
        raw = np.asarray(raw_proba, dtype=float).reshape(-1, 1)
        calibrated = calibrator.predict_proba(raw)[:, 1]
        return calibrated

    # ==========================================================
    # TRAIN ONE MODEL
    # ==========================================================

    def _train_one(
        self,
        df,
        experiment,
        horizon
    ):

        target_column = (
            f"fire_next_{horizon}"
        )

        features = list(
            self.feature_sets[
                experiment
            ]
        )

        required_columns = (
            features
            + [target_column]
        )

        data = df[
            required_columns
        ].copy()

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data[target_column] = pd.to_numeric(
            data[target_column],
            errors="coerce"
        )

        data = data.dropna()

        data[target_column] = (
            data[target_column]
            .astype(int)
        )

        # Enforce exact column order
        data = data[features + [target_column]]

        train, validation, test = (
            self._temporal_split(data)
        )

        X_train = train[features]
        y_train = train[target_column]

        X_validation = (
            validation[features]
        )

        y_validation = (
            validation[target_column]
        )

        X_test = test[features]
        y_test = test[target_column]

        # ------------------------------------------------------
        # SCHEMA ASSERTION BEFORE FIT
        # ------------------------------------------------------

        expected_features = list(
            self.feature_sets[experiment]
        )

        if list(X_train.columns) != expected_features:
            raise RuntimeError(
                f"Feature schema mismatch before fit!\n"
                f"Experiment: {experiment}\n"
                f"Horizon: {horizon}\n"
                f"Expected ({len(expected_features)}): {expected_features}\n"
                f"Got ({len(X_train.columns)}): {list(X_train.columns)}"
            )

        print()
        print("=" * 70)
        print(
            f"{experiment.upper()} / {horizon}"
        )
        print("=" * 70)

        print(
            f"Feature count : {len(features)}"
        )

        print(
            f"Feature schema:"
        )
        print(features)

        print(
            f"Train      : {len(X_train):,}"
        )

        print(
            f"Validation : {len(X_validation):,}"
        )

        print(
            f"Test       : {len(X_test):,}"
        )

        print(
            "Train:",
            y_train.value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            "Validation:",
            y_validation.value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            "Test:",
            y_test.value_counts()
            .sort_index()
            .to_dict()
        )

        # ------------------------------------------------------
        # MODEL (Random Forest on TRAIN only)
        # ------------------------------------------------------

        model = RandomForestClassifier(

            n_estimators=400,

            max_depth=16,

            min_samples_leaf=3,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        )

        print()
        print(
            "Training Random Forest..."
        )

        model.fit(
            X_train,
            y_train
        )

        # Post-fit feature count check
        if model.n_features_in_ != len(expected_features):
            raise RuntimeError(
                f"Model n_features_in_ mismatch!\n"
                f"Experiment: {experiment} / {horizon}\n"
                f"Schema: {len(expected_features)}\n"
                f"Model: {model.n_features_in_}"
            )

        # ------------------------------------------------------
        # RAW VALIDATION PROBABILITY
        # ------------------------------------------------------

        raw_val_proba = (
            model
            .predict_proba(
                X_validation
            )[:, 1]
        )

        # ------------------------------------------------------
        # CALIBRATION (Validation only – no TEST)
        # ------------------------------------------------------

        print()
        print(
            "Fitting calibrator (sigmoid) on VALIDATION only..."
        )

        calibrator = self._fit_calibrator(
            raw_val_proba,
            y_validation
        )

        calibrated_val_proba = self._apply_calibrator(
            calibrator,
            raw_val_proba
        )

        # ------------------------------------------------------
        # THRESHOLD on CALIBRATED validation probabilities
        # ------------------------------------------------------

        (
            best_threshold,
            threshold_results
        ) = self._find_best_threshold(
            y_validation,
            calibrated_val_proba
        )

        print()
        print(
            f"Best threshold (calibrated validation): "
            f"{best_threshold:.2f}"
        )

        # ------------------------------------------------------
        # VALIDATION METRICS (raw & calibrated)
        # ------------------------------------------------------

        raw_val_pred = (
            raw_val_proba >= best_threshold
        ).astype(int)

        cal_val_pred = (
            calibrated_val_proba >= best_threshold
        ).astype(int)

        raw_val_metrics = self._metrics(
            y_validation,
            raw_val_pred
        )

        cal_val_metrics = self._metrics(
            y_validation,
            cal_val_pred
        )

        raw_val_prob_metrics = self._probability_metrics(
            y_validation,
            raw_val_proba
        )

        cal_val_prob_metrics = self._probability_metrics(
            y_validation,
            calibrated_val_proba
        )

        print()
        print("Raw Validation Metrics")
        print(
            f"Accuracy  : {raw_val_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {raw_val_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {raw_val_metrics['recall']:.4f}"
        )
        print(
            f"F1-Score  : {raw_val_metrics['f1']:.4f}"
        )
        print(
            "Confusion Matrix:"
        )
        print(
            np.array(raw_val_metrics["confusion_matrix"])
        )

        print()
        print("Calibrated Validation Metrics")
        print(
            f"Accuracy  : {cal_val_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {cal_val_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {cal_val_metrics['recall']:.4f}"
        )
        print(
            f"F1-Score  : {cal_val_metrics['f1']:.4f}"
        )
        print(
            "Confusion Matrix:"
        )
        print(
            np.array(cal_val_metrics["confusion_matrix"])
        )

        print()
        print("Calibration (Validation)")
        print(f"Method           : {self.calibration_method}")
        print(
            f"Brier Raw        : {raw_val_prob_metrics['brier']:.6f}"
            if raw_val_prob_metrics['brier'] is not None
            else "Brier Raw        : N/A"
        )
        print(
            f"Brier Calibrated : {cal_val_prob_metrics['brier']:.6f}"
            if cal_val_prob_metrics['brier'] is not None
            else "Brier Calibrated : N/A"
        )
        print(
            f"LogLoss Raw      : {raw_val_prob_metrics['log_loss']:.6f}"
            if raw_val_prob_metrics['log_loss'] is not None
            else "LogLoss Raw      : N/A"
        )
        print(
            f"LogLoss Calibrated: {cal_val_prob_metrics['log_loss']:.6f}"
            if cal_val_prob_metrics['log_loss'] is not None
            else "LogLoss Calibrated: N/A"
        )
        print(
            f"ECE Raw          : {raw_val_prob_metrics['ece']:.6f}"
        )
        print(
            f"ECE Calibrated   : {cal_val_prob_metrics['ece']:.6f}"
        )

        # ------------------------------------------------------
        # INDEPENDENT TEST (frozen model + calibrator + threshold)
        # ------------------------------------------------------

        raw_test_proba = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        calibrated_test_proba = self._apply_calibrator(
            calibrator,
            raw_test_proba
        )

        # Sanity: probabilities in [0,1] and no NaN
        for name, proba in [
            ("raw_test", raw_test_proba),
            ("cal_test", calibrated_test_proba),
        ]:
            if np.any(np.isnan(proba)):
                raise RuntimeError(
                    f"NaN detected in {name} probabilities "
                    f"for {experiment}/{horizon}"
                )
            if np.any(proba < 0) or np.any(proba > 1):
                raise RuntimeError(
                    f"Probability out of [0,1] in {name} "
                    f"for {experiment}/{horizon}"
                )

        raw_test_pred = (
            raw_test_proba >= best_threshold
        ).astype(int)

        cal_test_pred = (
            calibrated_test_proba >= best_threshold
        ).astype(int)

        raw_test_metrics = self._metrics(
            y_test,
            raw_test_pred
        )

        cal_test_metrics = self._metrics(
            y_test,
            cal_test_pred
        )

        raw_test_prob_metrics = self._probability_metrics(
            y_test,
            raw_test_proba
        )

        cal_test_prob_metrics = self._probability_metrics(
            y_test,
            calibrated_test_proba
        )

        print()
        print("Independent TEST Metrics (Raw)")
        print(
            f"Accuracy  : {raw_test_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {raw_test_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {raw_test_metrics['recall']:.4f}"
        )
        print(
            f"F1-Score  : {raw_test_metrics['f1']:.4f}"
        )
        if raw_test_prob_metrics["pr_auc"] is not None:
            print(
                f"PR-AUC    : {raw_test_prob_metrics['pr_auc']:.4f}"
            )
        if raw_test_prob_metrics["roc_auc"] is not None:
            print(
                f"ROC-AUC   : {raw_test_prob_metrics['roc_auc']:.4f}"
            )
        print(
            f"Brier     : {raw_test_prob_metrics['brier']:.6f}"
            if raw_test_prob_metrics['brier'] is not None
            else "Brier     : N/A"
        )
        print(
            f"LogLoss   : {raw_test_prob_metrics['log_loss']:.6f}"
            if raw_test_prob_metrics['log_loss'] is not None
            else "LogLoss   : N/A"
        )
        print(
            f"ECE       : {raw_test_prob_metrics['ece']:.6f}"
        )
        print(
            "Confusion Matrix:"
        )
        print(
            np.array(raw_test_metrics["confusion_matrix"])
        )

        print()
        print("Independent TEST Metrics (Calibrated)")
        print(
            f"Accuracy  : {cal_test_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision : {cal_test_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {cal_test_metrics['recall']:.4f}"
        )
        print(
            f"F1-Score  : {cal_test_metrics['f1']:.4f}"
        )
        if cal_test_prob_metrics["pr_auc"] is not None:
            print(
                f"PR-AUC    : {cal_test_prob_metrics['pr_auc']:.4f}"
            )
        if cal_test_prob_metrics["roc_auc"] is not None:
            print(
                f"ROC-AUC   : {cal_test_prob_metrics['roc_auc']:.4f}"
            )
        print(
            f"Brier     : {cal_test_prob_metrics['brier']:.6f}"
            if cal_test_prob_metrics['brier'] is not None
            else "Brier     : N/A"
        )
        print(
            f"LogLoss   : {cal_test_prob_metrics['log_loss']:.6f}"
            if cal_test_prob_metrics['log_loss'] is not None
            else "LogLoss   : N/A"
        )
        print(
            f"ECE       : {cal_test_prob_metrics['ece']:.6f}"
        )
        print(
            "Confusion Matrix:"
        )
        print(
            np.array(cal_test_metrics["confusion_matrix"])
        )

        cm = np.array(cal_test_metrics["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        print()
        print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

        # ------------------------------------------------------
        # FEATURE IMPORTANCE
        # ------------------------------------------------------

        importance = (
            pd.Series(
                model.feature_importances_,
                index=features
            )
            .sort_values(
                ascending=False
            )
        )

        print()
        print(
            "Top Features:"
        )

        print(
            importance.head(15)
        )

        # ------------------------------------------------------
        # SAVE MODEL (v4 payload)
        # ------------------------------------------------------

        model_path = (
            self.model_dir
            / (
                "fireguard_forecast_"
                f"{experiment}_"
                f"{horizon}_v4.joblib"
            )
        )

        save_payload = {
            "model": model,
            "calibrator": calibrator,
            "threshold": best_threshold,
            "features": features,
            "feature_names": features,
            "experiment": experiment,
            "horizon": horizon,
            "calibration_method": self.calibration_method,
            "version": "v4",
        }

        joblib.dump(
            save_payload,
            model_path
        )

        # ------------------------------------------------------
        # SAVE THRESHOLD JSON
        # ------------------------------------------------------

        threshold_path = (
            self.model_dir
            / (
                "fireguard_threshold_"
                f"{experiment}_"
                f"{horizon}_v4.json"
            )
        )

        threshold_data = {
            "experiment": experiment,
            "horizon": horizon,
            "threshold": best_threshold,
            "features": features,
            "feature_count": len(features),
            "calibration_method": self.calibration_method,
            "validation_f1_calibrated":
                cal_val_metrics["f1"],
            "test_f1_calibrated":
                cal_test_metrics["f1"],
            "version": "v4",
        }

        with open(
            threshold_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                threshold_data,
                f,
                ensure_ascii=False,
                indent=4
            )

        # ------------------------------------------------------
        # AUDIT PRINT
        # ------------------------------------------------------

        print()
        print("-" * 70)
        print("AUDIT SUMMARY")
        print(f"Experiment      : {experiment}")
        print(f"Horizon         : {horizon}")
        print(f"Feature count   : {len(features)}")
        print(f"Schema          : PASS")
        print(f"Train size      : {len(X_train):,}")
        print(f"Validation size : {len(X_validation):,}")
        print(f"Test size       : {len(X_test):,}")
        print(f"Best threshold  : {best_threshold:.2f}")
        print(f"Calibration     : {self.calibration_method}")
        print(f"Val F1 (cal)    : {cal_val_metrics['f1']:.4f}")
        print(f"Test F1 (cal)   : {cal_test_metrics['f1']:.4f}")
        print(f"Saved model path: {model_path}")
        print("-" * 70)

        return {
            "model": model,
            "calibrator": calibrator,
            "threshold": best_threshold,
            "validation_raw": raw_val_metrics,
            "validation_calibrated": cal_val_metrics,
            "test_raw": raw_test_metrics,
            "test_calibrated": cal_test_metrics,
            "raw_val_prob_metrics": raw_val_prob_metrics,
            "cal_val_prob_metrics": cal_val_prob_metrics,
            "raw_test_prob_metrics": raw_test_prob_metrics,
            "cal_test_prob_metrics": cal_test_prob_metrics,
            "threshold_search": threshold_results,
            "feature_importance": importance.to_dict(),
            "model_path": str(model_path),
            "threshold_path": str(threshold_path),
            "features": features,
            "train_size": len(X_train),
            "validation_size": len(X_validation),
            "test_size": len(X_test),
        }

    # ==========================================================
    # TRAIN ALL
    # ==========================================================

    def train(self):

        df = self._load_data()

        print()
        print(
            "Feature Engineering v4 (schema identical to v3)..."
        )

        df = self._prepare_features(
            df
        )

        self.models = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.thresholds = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.calibrators = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.metrics = {}

        # ------------------------------------------------------
        # BOTH EXPERIMENTS
        # ------------------------------------------------------

        for experiment in (
            "sensor_only",
            "sensor_plus_flame"
        ):

            self.metrics[
                experiment
            ] = {}

            for horizon in (
                "24h",
                "48h",
                "72h"
            ):

                result = self._train_one(
                    df,
                    experiment,
                    horizon
                )

                self.models[
                    experiment
                ][horizon] = {
                    "model": result["model"],
                    "calibrator": result["calibrator"],
                    "features": result["features"],
                }

                self.thresholds[
                    experiment
                ][horizon] = (
                    result["threshold"]
                )

                self.calibrators[
                    experiment
                ][horizon] = (
                    result["calibrator"]
                )

                self.metrics[
                    experiment
                ][horizon] = {
                    "experiment": experiment,
                    "horizon": horizon,
                    "feature_count": len(result["features"]),
                    "feature_names": result["features"],
                    "train_size": result["train_size"],
                    "validation_size": result["validation_size"],
                    "test_size": result["test_size"],
                    "threshold": result["threshold"],
                    "calibration_method": self.calibration_method,
                    "raw_validation_metrics": result["validation_raw"],
                    "calibrated_validation_metrics": result["validation_calibrated"],
                    "raw_test_metrics": result["test_raw"],
                    "calibrated_test_metrics": result["test_calibrated"],
                    "brier_score_raw": result["raw_test_prob_metrics"]["brier"],
                    "brier_score_calibrated": result["cal_test_prob_metrics"]["brier"],
                    "log_loss_raw": result["raw_test_prob_metrics"]["log_loss"],
                    "log_loss_calibrated": result["cal_test_prob_metrics"]["log_loss"],
                    "ece_raw": result["raw_test_prob_metrics"]["ece"],
                    "ece_calibrated": result["cal_test_prob_metrics"]["ece"],
                    "pr_auc_raw": result["raw_test_prob_metrics"]["pr_auc"],
                    "pr_auc_calibrated": result["cal_test_prob_metrics"]["pr_auc"],
                    "roc_auc_raw": result["raw_test_prob_metrics"]["roc_auc"],
                    "roc_auc_calibrated": result["cal_test_prob_metrics"]["roc_auc"],
                    "confusion_matrix": result["test_calibrated"]["confusion_matrix"],
                    "feature_importance": result["feature_importance"],
                    "model_path": result["model_path"],
                    "threshold_path": result["threshold_path"],
                }

        # ------------------------------------------------------
        # SAVE COMPLETE METRICS (v4)
        # ------------------------------------------------------

        metrics_path = (
            self.model_dir
            / "forecast_metrics_v4.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.metrics,
                f,
                ensure_ascii=False,
                indent=4
            )

        # ------------------------------------------------------
        # SAVE FEATURES
        # ------------------------------------------------------

        feature_path = (
            self.model_dir
            / "forecast_features_v4.json"
        )

        with open(
            feature_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.feature_sets,
                f,
                ensure_ascii=False,
                indent=4
            )

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print(
            "FINAL TEST SUMMARY (Calibrated)"
        )
        print("=" * 70)

        for experiment in self.metrics:

            print()
            print(
                experiment.upper()
            )

            for horizon in (
                "24h",
                "48h",
                "72h"
            ):

                result = (
                    self.metrics[
                        experiment
                    ][horizon]
                )

                test = result[
                    "calibrated_test_metrics"
                ]

                print(
                    f"{horizon} | "
                    f"Threshold="
                    f"{result['threshold']:.2f} | "
                    f"Precision="
                    f"{test['precision']:.3f} | "
                    f"Recall="
                    f"{test['recall']:.3f} | "
                    f"F1="
                    f"{test['f1']:.3f} | "
                    f"Features="
                    f"{result['feature_count']}"
                )

        # ------------------------------------------------------
        # FINAL SELF-CHECK
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL SELF-CHECK (Schema / Calibration / Leakage)")
        print("=" * 70)

        all_pass = True
        warnings_list = []

        for experiment in (
            "sensor_only",
            "sensor_plus_flame"
        ):

            expected = list(
                self.feature_sets[experiment]
            )
            expected_count = len(expected)

            for horizon in (
                "24h",
                "48h",
                "72h"
            ):

                entry = self.models[experiment][horizon]
                model = entry["model"]
                calibrator = entry["calibrator"]
                features = entry["features"]

                model_n = model.n_features_in_

                schema_ok = (
                    model_n == expected_count
                    and list(features) == expected
                )

                cal_ok = calibrator is not None
                thr_ok = horizon in self.thresholds[experiment]
                method_ok = self.calibration_method == "sigmoid"

                status = "PASS" if (
                    schema_ok and cal_ok and thr_ok and method_ok
                ) else "FAIL"

                if status == "FAIL":
                    all_pass = False

                print(
                    f"{experiment.upper()} / {horizon}  "
                    f"Schema: {expected_count}  "
                    f"Model: {model_n}  "
                    f"Calibrator: {'YES' if cal_ok else 'NO'}  "
                    f"Threshold: {'YES' if thr_ok else 'NO'}  "
                    f"{status}"
                )

        # Explicit leakage checks (already enforced by design)
        print()
        print("Leakage checks:")
        print("  - Calibration used TEST data : NO")
        print("  - Threshold used TEST data   : NO")
        print("  - Chronological split        : YES (70/15/15)")
        print("  - Feature schema order       : FIXED")

        if not all_pass:
            print()
            print("❌ SELF-CHECK FAILED – Schema or calibration mismatch detected.")
            raise RuntimeError(
                "Training aborted due to self-check failure."
            )

        print()
        print("✅ All 6 models passed self-check.")

        self.is_trained = True

        print()
        print(
            "Metrics saved:"
        )
        print(
            metrics_path
        )

        # ------------------------------------------------------
        # FINAL QUALITY GATE MESSAGE
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("🔥 FireGuard Forecast v4 COMPLETE")
        print("=" * 70)
        print("Calibration: ENABLED")
        print(f"Calibration method: {self.calibration_method}")
        print("Schema:")
        print(f"SENSOR_ONLY = {len(self.SENSOR_FEATURES)}")
        print(f"SENSOR_PLUS_FLAME = {len(self.SENSOR_PLUS_FLAME_FEATURES)}")
        print("Models trained: 6")
        print("Independent TEST evaluations: 6")
        print("V3 models preserved: YES")
        print()
        print("Final Quality Gate:")
        if all_pass:
            print("🟢 PASS")
        else:
            print("🔴 FAIL")
        print("=" * 70)

        return True

    # ==========================================================
    # LOAD SAVED MODELS (v4 preferred, v3 fallback)
    # ==========================================================

    def _load_saved_models(
        self,
        experiment=None
    ):

        if experiment is None:
            experiment = (
                self.active_experiment
            )

        loaded = {}

        for horizon in (
            "24h",
            "48h",
            "72h"
        ):

            # Prefer v4
            model_path_v4 = (
                self.model_dir
                / (
                    "fireguard_forecast_"
                    f"{experiment}_"
                    f"{horizon}_v4.joblib"
                )
            )

            model_path_v3 = (
                self.model_dir
                / (
                    "fireguard_forecast_"
                    f"{experiment}_"
                    f"{horizon}_v3.joblib"
                )
            )

            threshold_path_v4 = (
                self.model_dir
                / (
                    "fireguard_threshold_"
                    f"{experiment}_"
                    f"{horizon}_v4.json"
                )
            )

            threshold_path_v3 = (
                self.model_dir
                / (
                    "fireguard_threshold_"
                    f"{experiment}_"
                    f"{horizon}_v3.json"
                )
            )

            model_path = None
            threshold_path = None
            version = None

            if model_path_v4.exists():
                model_path = model_path_v4
                threshold_path = threshold_path_v4
                version = "v4"
            elif model_path_v3.exists():
                model_path = model_path_v3
                threshold_path = threshold_path_v3
                version = "v3"
            else:
                continue

            try:

                payload = joblib.load(
                    model_path
                )

                if isinstance(payload, dict) and "model" in payload:
                    model = payload["model"]
                    features = list(
                        payload.get(
                            "features",
                            payload.get(
                                "feature_names",
                                self.feature_sets[experiment]
                            )
                        )
                    )
                    threshold = float(
                        payload.get("threshold", 0.50)
                    )
                    calibrator = payload.get("calibrator", None)
                else:
                    # Legacy raw model
                    model = payload
                    features = list(
                        self.feature_sets[experiment]
                    )
                    threshold = 0.50
                    calibrator = None

                loaded[horizon] = {
                    "model": model,
                    "calibrator": calibrator,
                    "features": list(features),
                    "version": version,
                }

                # Prefer threshold from threshold json if available
                if threshold_path and threshold_path.exists():

                    with open(
                        threshold_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        threshold_data = (
                            json.load(f)
                        )

                    self.thresholds[
                        experiment
                    ][horizon] = float(
                        threshold_data[
                            "threshold"
                        ]
                    )

                else:

                    self.thresholds[
                        experiment
                    ][horizon] = threshold

                if calibrator is not None:
                    self.calibrators[
                        experiment
                    ][horizon] = calibrator

            except Exception as e:

                print(
                    f"خطا در بارگذاری "
                    f"{experiment}/{horizon}: "
                    f"{e}"
                )

        if len(loaded) == 3:

            self.models[
                experiment
            ] = loaded

            return True

        return False

    # ==========================================================
    # BUILD CURRENT FEATURES
    # ==========================================================

    def _build_current_features(
        self,
        current_reading
    ):

        if not isinstance(
            current_reading,
            dict
        ):

            raise TypeError(
                "current_reading باید dict باشد."
            )

        timestamp = (
            current_reading.get(
                "timestamp"
            )
        )

        if timestamp is None:

            timestamp = pd.Timestamp.now()

        else:

            timestamp = pd.to_datetime(
                timestamp,
                errors="coerce"
            )

            if pd.isna(timestamp):

                timestamp = (
                    pd.Timestamp.now()
                )

        temperature = float(
            current_reading.get(
                "temperature",
                25
            )
        )

        humidity = float(
            current_reading.get(
                "humidity",
                60
            )
        )

        smoke = float(
            current_reading.get(
                "smoke",
                30
            )
        )

        flame = int(
            current_reading.get(
                "flame",
                0
            )
        )

        values = {

            "temperature":
                temperature,

            "humidity":
                humidity,

            "smoke":
                smoke,

            "flame":
                flame,

            "hour":
                timestamp.hour,

            "minute":
                timestamp.minute,

            "smoke_change_1m":
                float(
                    current_reading.get(
                        "smoke_change_1m",
                        0
                    )
                ),

            "smoke_change_5m":
                float(
                    current_reading.get(
                        "smoke_change_5m",
                        0
                    )
                ),

            "smoke_change_15m":
                float(
                    current_reading.get(
                        "smoke_change_15m",
                        0
                    )
                ),

            "smoke_change_30m":
                float(
                    current_reading.get(
                        "smoke_change_30m",
                        0
                    )
                ),

            "smoke_change_60m":
                float(
                    current_reading.get(
                        "smoke_change_60m",
                        0
                    )
                ),

            "temperature_change_5m":
                float(
                    current_reading.get(
                        "temperature_change_5m",
                        0
                    )
                ),

            "temperature_change_15m":
                float(
                    current_reading.get(
                        "temperature_change_15m",
                        0
                    )
                ),

            "temperature_change_30m":
                float(
                    current_reading.get(
                        "temperature_change_30m",
                        0
                    )
                ),

            "temperature_change_60m":
                float(
                    current_reading.get(
                        "temperature_change_60m",
                        0
                    )
                ),

            "humidity_change_5m":
                float(
                    current_reading.get(
                        "humidity_change_5m",
                        0
                    )
                ),

            "humidity_change_15m":
                float(
                    current_reading.get(
                        "humidity_change_15m",
                        0
                    )
                ),

            "humidity_change_30m":
                float(
                    current_reading.get(
                        "humidity_change_30m",
                        0
                    )
                ),

            "smoke_mean_5m":
                float(
                    current_reading.get(
                        "smoke_mean_5m",
                        smoke
                    )
                ),

            "smoke_mean_15m":
                float(
                    current_reading.get(
                        "smoke_mean_15m",
                        smoke
                    )
                ),

            "smoke_mean_30m":
                float(
                    current_reading.get(
                        "smoke_mean_30m",
                        smoke
                    )
                ),

            "smoke_mean_60m":
                float(
                    current_reading.get(
                        "smoke_mean_60m",
                        smoke
                    )
                ),

            "smoke_std_15m":
                float(
                    current_reading.get(
                        "smoke_std_15m",
                        0
                    )
                ),

            "smoke_std_30m":
                float(
                    current_reading.get(
                        "smoke_std_30m",
                        0
                    )
                ),

            "smoke_max_30m":
                float(
                    current_reading.get(
                        "smoke_max_30m",
                        smoke
                    )
                ),

            "temperature_mean_15m":
                float(
                    current_reading.get(
                        "temperature_mean_15m",
                        temperature
                    )
                ),

            "temperature_mean_30m":
                float(
                    current_reading.get(
                        "temperature_mean_30m",
                        temperature
                    )
                ),

            "humidity_mean_15m":
                float(
                    current_reading.get(
                        "humidity_mean_15m",
                        humidity
                    )
                ),

            "humidity_mean_30m":
                float(
                    current_reading.get(
                        "humidity_mean_30m",
                        humidity
                    )
                ),
        }

        # Build DataFrame with exact schema order of active experiment
        expected = list(
            self.feature_sets[
                self.active_experiment
            ]
        )

        row = {col: values.get(col, 0.0) for col in expected}

        return pd.DataFrame(
            [row],
            columns=expected
        )

    # ==========================================================
    # GET FORECAST
    # ==========================================================

    def get_forecast(
        self,
        current_reading
    ):

        experiment = (
            self.active_experiment
        )

        if not self.models.get(
            experiment
        ):

            if not self._load_saved_models(
                experiment
            ):

                return {
                    "success": False,

                    "error":
                        "مدل‌های v4/v3 پیدا نشدند.",

                    "forecast":
                        None
                }

        try:

            X = (
                self._build_current_features(
                    current_reading
                )
            )

        except Exception as e:

            return {
                "success": False,
                "error": str(e),
                "forecast": None
            }

        forecast = {}

        for horizon in (
            "24h",
            "48h",
            "72h"
        ):

            entry = (
                self.models[
                    experiment
                ].get(horizon)
            )

            if entry is None:

                forecast[
                    horizon
                ] = {
                    "error":
                        "model not loaded"
                }

                continue

            # Support both new dict format and legacy raw model
            if isinstance(entry, dict) and "model" in entry:
                model = entry["model"]
                calibrator = entry.get("calibrator", None)
                expected_features = list(
                    entry.get(
                        "features",
                        self.feature_sets[experiment]
                    )
                )
            else:
                model = entry
                calibrator = None
                expected_features = list(
                    self.feature_sets[experiment]
                )

            # Enforce exact feature order and count
            missing = [
                c for c in expected_features
                if c not in X.columns
            ]
            if missing:
                return {
                    "success": False,
                    "error": (
                        f"Missing features for {experiment}/{horizon}: "
                        f"{missing}"
                    ),
                    "forecast": None
                }

            X_inf = X[expected_features].copy()

            if list(X_inf.columns) != expected_features:
                return {
                    "success": False,
                    "error": (
                        f"Feature order mismatch for {experiment}/{horizon}"
                    ),
                    "forecast": None
                }

            if X_inf.shape[1] != len(expected_features):
                return {
                    "success": False,
                    "error": (
                        f"Feature count mismatch: "
                        f"X has {X_inf.shape[1]}, "
                        f"model expects {len(expected_features)}"
                    ),
                    "forecast": None
                }

            # Final assert
            assert list(X_inf.columns) == expected_features
            assert X_inf.shape[1] == len(expected_features)

            raw_probability = float(
                model
                .predict_proba(X_inf)[0][1]
            )

            # Apply calibrator if available
            if calibrator is not None:
                calibrated_probability = float(
                    self._apply_calibrator(
                        calibrator,
                        [raw_probability]
                    )[0]
                )
            else:
                calibrated_probability = raw_probability

            # Clip for safety
            raw_probability = float(np.clip(raw_probability, 0.0, 1.0))
            calibrated_probability = float(
                np.clip(calibrated_probability, 0.0, 1.0)
            )

            threshold = float(
                self.thresholds[
                    experiment
                ].get(
                    horizon,
                    0.50
                )
            )

            # Primary decision uses calibrated probability
            prediction = int(
                calibrated_probability
                >= threshold
            )

            forecast[
                horizon
            ] = {

                "raw_probability":
                    round(
                        raw_probability,
                        4
                    ),

                "raw_probability_percent":
                    round(
                        raw_probability * 100,
                        2
                    ),

                "calibrated_probability":
                    round(
                        calibrated_probability,
                        4
                    ),

                "calibrated_probability_percent":
                    round(
                        calibrated_probability * 100,
                        2
                    ),

                # Primary fire probability shown to user = calibrated
                "fire_probability":
                    round(
                        calibrated_probability,
                        4
                    ),

                "fire_probability_percent":
                    round(
                        calibrated_probability * 100,
                        2
                    ),

                "threshold":
                    round(
                        threshold,
                        2
                    ),

                "prediction":
                    prediction,

                "risk_level":
                    self._risk_level(
                        calibrated_probability,
                        threshold
                    ),

                "calibration_method":
                    self.calibration_method if calibrator is not None else "none",
            }

        return {
            "success": True,

            "experiment":
                experiment,

            "forecast":
                forecast
        }

    # ==========================================================
    # SET EXPERIMENT
    # ==========================================================

    def set_experiment(
        self,
        experiment
    ):

        if experiment not in (
            "sensor_only",
            "sensor_plus_flame"
        ):

            raise ValueError(
                "experiment باید "
                "'sensor_only' یا "
                "'sensor_plus_flame' باشد."
            )

        self.active_experiment = (
            experiment
        )

        self.feature_columns = (
            self.feature_sets[
                experiment
            ]
        )

        self._load_saved_models(
            experiment
        )

    # ==========================================================
    # RISK LEVEL
    # ==========================================================

    @staticmethod
    def _risk_level(
        probability,
        threshold
    ):

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


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    engine = FireGuardForecast()

    success = engine.train()

    if not success:

        raise SystemExit(1)