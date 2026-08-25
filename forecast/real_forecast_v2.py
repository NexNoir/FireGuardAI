# forecast/real_forecast.py
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


class FireGuardForecast:
    """
    FireGuard Forecast Engine v2

    Two independent feature experiments:
        sensor_only
        sensor_plus_flame

    Three independent horizons:
        24h / 48h / 72h

    Public API:
        train()
        get_forecast(current_reading)
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "saved_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # v3 dataset; fallback keeps compatibility with the older filename.
        v3_path = self.data_dir / "fireguard_forecast_60days.csv"
        old_path = self.data_dir / "fireguard_forecast_30days.csv"
        self.dataset_path = v3_path if v3_path.exists() else old_path

        self.horizons = {
            "24h": 24,
            "48h": 48,
            "72h": 72,
        }

        self.feature_sets = {
            "sensor_only": [
                "temperature",
                "humidity",
                "smoke",
                "hour",
                "minute",
                "temperature_change_5m",
                "temperature_change_15m",
                "humidity_change_5m",
                "humidity_change_15m",
                "smoke_change_5m",
                "smoke_change_15m",
                "smoke_change_30m",
                "temperature_rolling_mean",
                "humidity_rolling_mean",
                "smoke_rolling_mean",
            ],
            "sensor_plus_flame": [
                "temperature",
                "humidity",
                "smoke",
                "flame",
                "hour",
                "minute",
                "temperature_change_5m",
                "temperature_change_15m",
                "humidity_change_5m",
                "humidity_change_15m",
                "smoke_change_5m",
                "smoke_change_15m",
                "smoke_change_30m",
                "temperature_rolling_mean",
                "humidity_rolling_mean",
                "smoke_rolling_mean",
            ],
        }

        self.models = {
            "sensor_only": {},
            "sensor_plus_flame": {},
        }

        self.metrics = {}
        self.is_trained = False

        # Default inference model: Sensor-only.
        self.active_experiment = "sensor_only"

        # Kept for compatibility with existing code.
        self.feature_columns = self.feature_sets[
            self.active_experiment
        ]

        print(f"Dataset: {self.dataset_path}")
        print(f"Model directory: {self.model_dir}")

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    def _load_data(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset پیدا نشد:\n{self.dataset_path}\n"
                "ابتدا generate_forecast_dataset_v3.py را اجرا کنید."
            )

        df = pd.read_csv(self.dataset_path)

        if "timestamp" not in df.columns:
            raise ValueError("ستون timestamp وجود ندارد.")

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp")
        df = df.drop_duplicates("timestamp").reset_index(drop=True)

        required = [
            "temperature",
            "humidity",
            "smoke",
            "flame",
        ]

        for column in required:
            if column not in df.columns:
                raise ValueError(f"ستون مورد نیاز وجود ندارد: {column}")

            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        print(f"\nRecords: {len(df):,}")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End:   {df['timestamp'].max()}")

        return df

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def _prepare_features(self, df):
        df = df.copy()

        # IMPORTANT:
        # day is intentionally NOT used.
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute

        # Minute-level dataset: shift(N) = N minutes.
        df["temperature_change_5m"] = (
            df["temperature"] - df["temperature"].shift(5)
        )
        df["temperature_change_15m"] = (
            df["temperature"] - df["temperature"].shift(15)
        )

        df["humidity_change_5m"] = (
            df["humidity"] - df["humidity"].shift(5)
        )
        df["humidity_change_15m"] = (
            df["humidity"] - df["humidity"].shift(15)
        )

        df["smoke_change_5m"] = (
            df["smoke"] - df["smoke"].shift(5)
        )
        df["smoke_change_15m"] = (
            df["smoke"] - df["smoke"].shift(15)
        )
        df["smoke_change_30m"] = (
            df["smoke"] - df["smoke"].shift(30)
        )

        # Rolling windows use only current and past observations.
        df["temperature_rolling_mean"] = (
            df["temperature"].rolling(15).mean()
        )
        df["humidity_rolling_mean"] = (
            df["humidity"].rolling(15).mean()
        )
        df["smoke_rolling_mean"] = (
            df["smoke"].rolling(15).mean()
        )

        return df

    # ==========================================================
    # TEMPORAL SPLIT
    # ==========================================================

    @staticmethod
    def _temporal_split(df):
        n = len(df)

        train_end = int(n * 0.70)
        validation_end = int(n * 0.85)

        return (
            df.iloc[:train_end].copy(),
            df.iloc[train_end:validation_end].copy(),
            df.iloc[validation_end:].copy(),
        )

    # ==========================================================
    # METRICS
    # ==========================================================

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        return {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_true, y_pred, zero_division=0
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred
            ).tolist(),
        }

    @staticmethod
    def _print_metrics(title, metrics):
        print(f"\n{title}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1-Score  : {metrics['f1']:.4f}")
        print("Confusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))

    # ==========================================================
    # TRAIN ONE EXPERIMENT / HORIZON
    # ==========================================================

    def _train_one(self, df, experiment, horizon):
        target_column = f"fire_next_{horizon}"

        if target_column not in df.columns:
            raise ValueError(
                f"Target وجود ندارد: {target_column}"
            )

        features = self.feature_sets[experiment]

        data = df[features + [target_column]].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data[target_column] = pd.to_numeric(
            data[target_column], errors="coerce"
        )
        data = data.dropna()
        data[target_column] = data[target_column].astype(int)

        if data[target_column].nunique() < 2:
            raise ValueError(
                f"{target_column} فقط یک کلاس دارد."
            )

        train_df, validation_df, test_df = (
            self._temporal_split(data)
        )

        X_train = train_df[features]
        y_train = train_df[target_column]

        X_validation = validation_df[features]
        y_validation = validation_df[target_column]

        X_test = test_df[features]
        y_test = test_df[target_column]

        print()
        print("=" * 70)
        print(f"{experiment.upper()} / {horizon}")
        print("=" * 70)

        print(f"Train      : {len(X_train):,}")
        print(f"Validation : {len(X_validation):,}")
        print(f"Test       : {len(X_test):,}")

        print(
            "Train distribution:",
            y_train.value_counts().sort_index().to_dict()
        )
        print(
            "Validation distribution:",
            y_validation.value_counts().sort_index().to_dict()
        )
        print(
            "Test distribution:",
            y_test.value_counts().sort_index().to_dict()
        )

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        print("\nTraining Random Forest...")
        model.fit(X_train, y_train)

        validation_prediction = model.predict(X_validation)
        test_prediction = model.predict(X_test)

        validation_metrics = self._calculate_metrics(
            y_validation,
            validation_prediction,
        )
        test_metrics = self._calculate_metrics(
            y_test,
            test_prediction,
        )

        self._print_metrics(
            "Validation Metrics",
            validation_metrics,
        )
        self._print_metrics(
            "Test Metrics",
            test_metrics,
        )

        importance = (
            pd.Series(
                model.feature_importances_,
                index=features,
            )
            .sort_values(ascending=False)
        )

        print("\nTop Feature Importance:")
        print(importance.head(10))

        model_path = (
            self.model_dir
            / f"fireguard_forecast_{experiment}_{horizon}.joblib"
        )

        joblib.dump(model, model_path)

        print(f"\nModel saved: {model_path}")

        return {
            "model": model,
            "experiment": experiment,
            "horizon": horizon,
            "features": features,
            "validation": validation_metrics,
            "test": test_metrics,
            "importance": importance.to_dict(),
            "train_size": len(X_train),
            "validation_size": len(X_validation),
            "test_size": len(X_test),
        }

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(self):
        print("\n=== FireGuard Forecast v2 Training ===")

        try:
            df = self._load_data()

            time_span = (
                df["timestamp"].max()
                - df["timestamp"].min()
            )

            if time_span < pd.Timedelta(hours=72):
                raise ValueError(
                    "Dataset برای Forecast 72h کافی نیست."
                )

            print("\nFeature Engineering...")
            df = self._prepare_features(df)

            self.models = {
                "sensor_only": {},
                "sensor_plus_flame": {},
            }
            self.metrics = {}

            for experiment in self.feature_sets:
                self.metrics[experiment] = {}

                for horizon in self.horizons:
                    result = self._train_one(
                        df,
                        experiment,
                        horizon,
                    )

                    self.models[experiment][horizon] = (
                        result["model"]
                    )

                    self.metrics[experiment][horizon] = result

            metrics_path = (
                self.model_dir
                / "forecast_metrics_v2.json"
            )

            with open(
                metrics_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.metrics,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    default=str,
                )

            feature_path = (
                self.model_dir
                / "forecast_features_v2.json"
            )

            with open(
                feature_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.feature_sets,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

            self.is_trained = True

            print("\n" + "=" * 70)
            print("✅ Sensor-only: 24h / 48h / 72h")
            print("✅ Sensor + flame: 24h / 48h / 72h")
            print("✅ Temporal 70/15/15")
            print("✅ Metrics + Confusion Matrix")
            print("✅ Models and metrics saved")
            print("=" * 70)

            return True

        except Exception as e:
            self.is_trained = False
            print("\n❌ Forecast training error:")
            print(str(e))
            return False

    # ==========================================================
    # CURRENT FEATURE BUILD
    # ==========================================================

    def _build_current_features(self, current_reading):
        if not isinstance(current_reading, dict):
            raise TypeError(
                "current_reading باید dict باشد."
            )

        timestamp = current_reading.get("timestamp")
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        else:
            timestamp = pd.to_datetime(
                timestamp,
                errors="coerce",
            )
            if pd.isna(timestamp):
                timestamp = pd.Timestamp.now()

        temperature = float(
            current_reading.get("temperature", 25)
        )
        humidity = float(
            current_reading.get("humidity", 60)
        )
        smoke = float(
            current_reading.get("smoke", 30)
        )
        flame = int(
            current_reading.get("flame", 0)
        )

        values = {
            "temperature": temperature,
            "humidity": humidity,
            "smoke": smoke,
            "flame": flame,
            "hour": timestamp.hour,
            "minute": timestamp.minute,

            "temperature_change_5m": float(
                current_reading.get(
                    "temperature_change_5m", 0
                )
            ),
            "temperature_change_15m": float(
                current_reading.get(
                    "temperature_change_15m", 0
                )
            ),
            "humidity_change_5m": float(
                current_reading.get(
                    "humidity_change_5m", 0
                )
            ),
            "humidity_change_15m": float(
                current_reading.get(
                    "humidity_change_15m", 0
                )
            ),
            "smoke_change_5m": float(
                current_reading.get(
                    "smoke_change_5m", 0
                )
            ),
            "smoke_change_15m": float(
                current_reading.get(
                    "smoke_change_15m", 0
                )
            ),
            "smoke_change_30m": float(
                current_reading.get(
                    "smoke_change_30m", 0
                )
            ),
            "temperature_rolling_mean": float(
                current_reading.get(
                    "temperature_rolling_mean",
                    temperature,
                )
            ),
            "humidity_rolling_mean": float(
                current_reading.get(
                    "humidity_rolling_mean",
                    humidity,
                )
            ),
            "smoke_rolling_mean": float(
                current_reading.get(
                    "smoke_rolling_mean",
                    smoke,
                )
            ),
        }

        return pd.DataFrame(
            [values],
            columns=self.feature_sets[
                self.active_experiment
            ],
        )

    # ==========================================================
    # LOAD SAVED MODELS
    # ==========================================================

    def _load_saved_models(self, experiment=None):
        experiment = experiment or self.active_experiment

        loaded = {}

        for horizon in self.horizons:
            path = (
                self.model_dir
                / f"fireguard_forecast_{experiment}_{horizon}.joblib"
            )

            if path.exists():
                try:
                    loaded[horizon] = joblib.load(path)
                except Exception as e:
                    print(
                        f"⚠️ خطا در بارگذاری "
                        f"{experiment}/{horizon}: {e}"
                    )

        if len(loaded) == len(self.horizons):
            self.models[experiment] = loaded
            return True

        return False

    # ==========================================================
    # GET FORECAST
    # ==========================================================

    def get_forecast(self, current_reading):
        experiment = self.active_experiment

        if not self.models.get(experiment):
            if not self._load_saved_models(experiment):
                return {
                    "success": False,
                    "error": (
                        f"مدل‌های {experiment} آموزش داده نشده‌اند."
                    ),
                    "forecast": None,
                }

        try:
            X = self._build_current_features(
                current_reading
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "forecast": None,
            }

        results = {}

        for horizon, model in self.models[
            experiment
        ].items():
            try:
                probability = float(
                    model.predict_proba(X)[0][1]
                )

                prediction = int(
                    probability >= 0.5
                )

                results[horizon] = {
                    "fire_probability": round(
                        probability, 4
                    ),
                    "fire_probability_percent": round(
                        probability * 100, 2
                    ),
                    "prediction": prediction,
                    "risk_level": self._risk_level(
                        probability
                    ),
                }

            except Exception as e:
                results[horizon] = {
                    "error": str(e)
                }

        return {
            "success": True,
            "experiment": experiment,
            "forecast": results,
        }

    # ==========================================================
    # EXPERIMENT SELECTION
    # ==========================================================

    def set_experiment(self, experiment):
        if experiment not in self.feature_sets:
            raise ValueError(
                "experiment باید یکی از "
                "'sensor_only' یا 'sensor_plus_flame' باشد."
            )

        self.active_experiment = experiment
        self.feature_columns = self.feature_sets[experiment]

        if not self.models.get(experiment):
            self._load_saved_models(experiment)

    # ==========================================================
    # RISK LEVEL
    # ==========================================================

    @staticmethod
    def _risk_level(probability):
        if probability < 0.20:
            return "LOW"
        if probability < 0.50:
            return "MEDIUM"
        if probability < 0.75:
            return "HIGH"
        return "CRITICAL"


if __name__ == "__main__":
    engine = FireGuardForecast()

    success = engine.train()

    if not success:
        raise SystemExit(1)
