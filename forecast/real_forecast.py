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
    FireGuard Forecast Engine

    آموزش سه مدل مستقل:
        24h
        48h
        72h

    منبع اصلی داده:
        data/fireguard_forecast_30days.csv

    API عمومی:
        train()
        get_forecast(current_reading)
    """

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(self):

        self.base_dir = (
            Path(__file__).resolve().parent.parent
        )

        self.data_dir = (
            self.base_dir / "data"
        )

        self.model_dir = (
            Path(__file__).parent.parent / "saved_models"
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.dataset_path = (
            self.data_dir
            / "fireguard_forecast_30days.csv"
        )

        self.models = {}

        self.metrics = {}

        self.is_trained = False

        self.feature_columns = [
            "temperature",
            "humidity",
            "smoke",
            "flame",
            "hour",
            "minute",
            "day",

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
        ]

        self.horizons = {
            "24h": 24,
            "48h": 48,
            "72h": 72,
        }

        print(
            f"📁 Model directory: {self.model_dir}"
        )

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    def _load_data(self):

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                "\n❌ Synthetic Forecast Dataset پیدا نشد.\n"
                f"مسیر مورد انتظار:\n{self.dataset_path}\n\n"
                "ابتدا generate_forecast_dataset.py را اجرا کنید."
            )

        print(
            "\n📂 بارگذاری Dataset Forecast..."
        )

        df = pd.read_csv(
            self.dataset_path
        )

        if "timestamp" not in df.columns:

            raise ValueError(
                "❌ ستون timestamp در Dataset وجود ندارد."
            )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )

        df = df.drop_duplicates(
            subset=["timestamp"]
        )

        print(
            f"تعداد رکوردهای Dataset: {len(df):,}"
        )

        print(
            f"شروع: {df['timestamp'].min()}"
        )

        print(
            f"پایان: {df['timestamp'].max()}"
        )

        return df

    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def _prepare_features(self, df):

        df = df.copy()

        numeric_columns = [
            "temperature",
            "humidity",
            "smoke",
            "flame",
        ]

        for column in numeric_columns:

            if column not in df.columns:

                raise ValueError(
                    f"❌ ستون مورد نیاز وجود ندارد: {column}"
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ------------------------------------------------------
        # Time Features
        # ------------------------------------------------------

        df["hour"] = (
            df["timestamp"].dt.hour
        )

        df["minute"] = (
            df["timestamp"].dt.minute
        )

        df["day"] = (
            df["timestamp"].dt.day
        )

        # ------------------------------------------------------
        # Changes / Trends
        # ------------------------------------------------------

        df["temperature_change_5m"] = (
            df["temperature"]
            - df["temperature"].shift(5)
        )

        df["temperature_change_15m"] = (
            df["temperature"]
            - df["temperature"].shift(15)
        )

        df["humidity_change_5m"] = (
            df["humidity"]
            - df["humidity"].shift(5)
        )

        df["humidity_change_15m"] = (
            df["humidity"]
            - df["humidity"].shift(15)
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

        # ------------------------------------------------------
        # Rolling Statistics
        # ------------------------------------------------------

        df["temperature_rolling_mean"] = (
            df["temperature"]
            .rolling(15)
            .mean()
        )

        df["humidity_rolling_mean"] = (
            df["humidity"]
            .rolling(15)
            .mean()
        )

        df["smoke_rolling_mean"] = (
            df["smoke"]
            .rolling(15)
            .mean()
        )

        # ------------------------------------------------------
        # فقط Featureهای مدل
        # ------------------------------------------------------

        return df

    # ==========================================================
    # TARGET CHECK
    # ==========================================================

    def _target_column(self, horizon):

        column = (
            f"fire_next_{horizon}"
        )

        return column

    # ==========================================================
    # TIME SPLIT
    # ==========================================================

    def _temporal_split(
        self,
        df
    ):

        """
        Split زمانی:

        70% Train
        15% Validation
        15% Test

        بدون Shuffle
        """

        n = len(df)

        train_end = int(
            n * 0.70
        )

        validation_end = int(
            n * 0.85
        )

        train_df = df.iloc[
            :train_end
        ].copy()

        validation_df = df.iloc[
            train_end:validation_end
        ].copy()

        test_df = df.iloc[
            validation_end:
        ].copy()

        return (
            train_df,
            validation_df,
            test_df
        )

    # ==========================================================
    # TRAIN ONE MODEL
    # ==========================================================

    def _train_model(
        self,
        df,
        horizon
    ):

        target_column = (
            self._target_column(
                horizon
            )
        )

        if target_column not in df.columns:

            raise ValueError(
                f"❌ Target وجود ندارد: {target_column}"
            )

        data = df[
            self.feature_columns
            + [target_column]
        ].copy()

        # ------------------------------------------------------
        # حذف NaNهای ناشی از Rolling / Shift
        # ------------------------------------------------------

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()

        # ------------------------------------------------------
        # تبدیل Target
        # ------------------------------------------------------

        data[target_column] = (
            pd.to_numeric(
                data[target_column],
                errors="coerce"
            )
        )

        data = data.dropna(
            subset=[target_column]
        )

        data[target_column] = (
            data[target_column]
            .astype(int)
        )

        if data[target_column].nunique() < 2:

            raise ValueError(
                f"❌ Target {target_column} فقط یک کلاس دارد."
            )

        # ------------------------------------------------------
        # Temporal Split
        # ------------------------------------------------------

        train_df, validation_df, test_df = (
            self._temporal_split(
                data
            )
        )

        X_train = train_df[
            self.feature_columns
        ]

        y_train = train_df[
            target_column
        ]

        X_validation = validation_df[
            self.feature_columns
        ]

        y_validation = validation_df[
            target_column
        ]

        X_test = test_df[
            self.feature_columns
        ]

        y_test = test_df[
            target_column
        ]

        print()
        print(
            f"========== {horizon} Forecast =========="
        )

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
            f"Train class distribution:"
        )

        print(
            y_train.value_counts()
            .sort_index()
            .to_dict()
        )

        # ------------------------------------------------------
        # Random Forest
        # ------------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        print(
            f"\n🌲 آموزش Random Forest برای {horizon}..."
        )

        model.fit(
            X_train,
            y_train
        )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        validation_prediction = (
            model.predict(
                X_validation
            )
        )

        validation_metrics = (
            self._calculate_metrics(
                y_validation,
                validation_prediction
            )
        )

        # ------------------------------------------------------
        # Test
        # ------------------------------------------------------

        test_prediction = (
            model.predict(
                X_test
            )
        )

        test_metrics = (
            self._calculate_metrics(
                y_test,
                test_prediction
            )
        )

        print(
            "\n📊 Validation:"
        )

        self._print_metrics(
            validation_metrics
        )

        print(
            "\n📊 Test:"
        )

        self._print_metrics(
            test_metrics
        )

        # ------------------------------------------------------
        # Feature Importance
        # ------------------------------------------------------

        importance = pd.Series(
            model.feature_importances_,
            index=self.feature_columns
        ).sort_values(
            ascending=False
        )

        print(
            "\n🔎 مهم‌ترین Featureها:"
        )

        print(
            importance.head(10)
        )

        # ------------------------------------------------------
        # Save Model
        # ------------------------------------------------------

        model_path = (
            self.model_dir
            / f"fireguard_forecast_{horizon}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\n💾 Model saved:"
        )

        print(
            model_path
        )

        return {
            "model": model,
            "validation": validation_metrics,
            "test": test_metrics,
            "importance": importance.to_dict(),
            "train_size": len(X_train),
            "validation_size": len(X_validation),
            "test_size": len(X_test),
        }

    # ==========================================================
    # METRICS
    # ==========================================================

    @staticmethod
    def _calculate_metrics(
        y_true,
        y_pred
    ):

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

            "confusion_matrix": (
                confusion_matrix(
                    y_true,
                    y_pred
                ).tolist()
            ),
        }

    # ==========================================================
    # PRINT METRICS
    # ==========================================================

    @staticmethod
    def _print_metrics(
        metrics
    ):

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1-Score  : {metrics['f1']:.4f}"
        )

        print(
            f"Confusion Matrix:"
        )

        print(
            np.array(
                metrics["confusion_matrix"]
            )
        )

    # ==========================================================
    # TRAIN
    # ==========================================================

    def train(self):

        print(
            "=== شروع آموزش Forecast واقعی ==="
        )

        try:

            df = self._load_data()

            # --------------------------------------------------
            # بررسی پوشش زمانی
            # --------------------------------------------------

            time_span = (
                df["timestamp"].max()
                - df["timestamp"].min()
            )

            print()
            print(
                "=== پوشش زمانی Dataset ==="
            )

            print(
                f"شروع : {df['timestamp'].min()}"
            )

            print(
                f"پایان : {df['timestamp'].max()}"
            )

            print(
                f"مدت   : {time_span}"
            )

            if time_span < pd.Timedelta(
                hours=72
            ):

                print(
                    "\n❌ Dataset برای Forecast "
                    "72 ساعته کافی نیست."
                )

                self.is_trained = False

                return False

            # --------------------------------------------------
            # Feature Engineering
            # --------------------------------------------------

            print(
                "\n⚙️ Feature Engineering..."
            )

            df = self._prepare_features(
                df
            )

            # --------------------------------------------------
            # Train 24 / 48 / 72
            # --------------------------------------------------

            for horizon in self.horizons:

                result = (
                    self._train_model(
                        df,
                        horizon
                    )
                )

                self.models[
                    horizon
                ] = result["model"]

                self.metrics[
                    horizon
                ] = result

            # --------------------------------------------------
            # Save Metrics
            # --------------------------------------------------

            metrics_path = (
                self.model_dir
                / "forecast_metrics.json"
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
                    indent=4,
                    default=str
                )

            # --------------------------------------------------
            # Save Feature Columns
            # --------------------------------------------------

            feature_path = (
                self.model_dir
                / "forecast_features.json"
            )

            with open(
                feature_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.feature_columns,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            self.is_trained = True

            print()
            print(
                "=" * 60
            )

            print(
                "✅ هر سه مدل با موفقیت آموزش داده شدند."
            )

            print(
                "✅ 24h"
            )

            print(
                "✅ 48h"
            )

            print(
                "✅ 72h"
            )

            print(
                "=" * 60
            )

            return True

        except Exception as e:

            self.is_trained = False

            print()
            print(
                "❌ خطا در آموزش Forecast:"
            )

            print(
                str(e)
            )

            return False

    # ==========================================================
    # BUILD CURRENT FEATURES
    # ==========================================================

    def _build_current_features(
        self,
        current_reading
    ):

        """
        تبدیل current_reading به Featureهای مورد نیاز مدل.

        current_reading می‌تواند dict باشد.
        """

        if not isinstance(
            current_reading,
            dict
        ):

            raise TypeError(
                "current_reading باید dict باشد."
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

        timestamp = current_reading.get(
            "timestamp"
        )

        if timestamp is None:

            timestamp = pd.Timestamp.now()

        else:

            timestamp = pd.to_datetime(
                timestamp,
                errors="coerce"
            )

            if pd.isna(timestamp):

                timestamp = pd.Timestamp.now()

        # ------------------------------------------------------
        # اگر trend از سنسور ارسال نشده باشد،
        # مقدار صفر استفاده می‌شود.
        # ------------------------------------------------------

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

            "day":
                timestamp.day,

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

            "temperature_rolling_mean":
                float(
                    current_reading.get(
                        "temperature_rolling_mean",
                        temperature
                    )
                ),

            "humidity_rolling_mean":
                float(
                    current_reading.get(
                        "humidity_rolling_mean",
                        humidity
                    )
                ),

            "smoke_rolling_mean":
                float(
                    current_reading.get(
                        "smoke_rolling_mean",
                        smoke
                    )
                ),
        }

        return pd.DataFrame(
            [values],
            columns=self.feature_columns
        )

    # ==========================================================
    # LOAD SAVED MODELS
    # ==========================================================

    def _load_saved_models(
        self
    ):

        for horizon in self.horizons:

            path = (
                self.model_dir
                / f"fireguard_forecast_{horizon}.joblib"
            )

            if path.exists():

                try:

                    self.models[
                        horizon
                    ] = joblib.load(
                        path
                    )

                except Exception as e:

                    print(
                        f"⚠️ خطا در بارگذاری {horizon}: {e}"
                    )

        return len(
            self.models
        ) == len(
            self.horizons
        )

    # ==========================================================
    # GET FORECAST
    # ==========================================================

    def get_forecast(
        self,
        current_reading
    ):

        # ------------------------------------------------------
        # اگر مدل‌ها در حافظه نیستند، از Disk بخوان
        # ------------------------------------------------------

        if not self.models:

            loaded = (
                self._load_saved_models()
            )

            if not loaded:

                return {
                    "success": False,
                    "error":
                        "مدل Forecast آموزش داده نشده است.",
                    "forecast": None
                }

        # ------------------------------------------------------
        # ساخت Feature
        # ------------------------------------------------------

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

        results = {}

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------

        for horizon, model in self.models.items():

            try:

                probability = float(
                    model.predict_proba(
                        X
                    )[0][1]
                )

                prediction = int(
                    probability >= 0.5
                )

                results[horizon] = {

                    "fire_probability":
                        round(
                            probability,
                            4
                        ),

                    "fire_probability_percent":
                        round(
                            probability * 100,
                            2
                        ),

                    "prediction":
                        prediction,

                    "risk_level":
                        self._risk_level(
                            probability
                        )
                }

            except Exception as e:

                results[horizon] = {
                    "error": str(e)
                }

        return {
            "success": True,
            "forecast": results
        }

    # ==========================================================
    # RISK LEVEL
    # ==========================================================

    @staticmethod
    def _risk_level(
        probability
    ):

        if probability < 0.20:

            return "LOW"

        elif probability < 0.50:

            return "MEDIUM"

        elif probability < 0.75:

            return "HIGH"

        else:

            return "CRITICAL"