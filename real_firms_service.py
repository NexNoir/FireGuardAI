# real_firms_service.py

"""
🔥 FIREGUARD — سرویس تولید مدل فصلی جنگل‌های هیرکانی
Real FIRMS V1 - Seasonal Model for Hyrcanian Forests
Period: 2001-2025
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# تنظیمات مسیرها
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_models" / "real_firms_v1"

THRESHOLD_CONFIG = BASE_DIR / "data" / "retraining" / "real_firms_threshold_config_v1.json"


# ============================================================================
# ویژگی‌های مدل فصلی (۱۵ ویژگی)
# ============================================================================

FEATURE_COLUMNS = [
    "latitude",          # عرض جغرافیایی
    "longitude",         # طول جغرافیایی
    "brightness",        # روشنایی حرارتی (از ماهواره)
    "scan",              # اندازه اسکن
    "track",             # اندازه تراک
    "confidence",        # اطمینان تشخیص (از NASA)
    "bright_t31",        # دمای باند حرارتی
    "frp",               # نرخ انتشار آتش
    "hour",              # ساعت (از acq_time)
    "minute",            # دقیقه (از acq_time)
    "daynight_encoded",  # شب/روز (D=0, N=1)
    "satellite_encoded", # ماهواره (MODIS/VIIRS)
    "instrument_encoded",# ابزار
    "type_encoded",      # نوع آتش
    "season_encoded",    # فصل (0=Spring, 1=Summer, 2=Autumn, 3=Winter)
]

# آستانه‌های تأییدشده
DEFAULT_THRESHOLDS = {
    "24h": 0.35,
    "48h": 0.35,
    "72h": 0.30,
}

# نام فایل‌های مدل
MODEL_FILES = {
    "24h": "fireguard_real_firms_sensor_only_24h_v1.joblib",
    "48h": "fireguard_real_firms_sensor_only_48h_v1.joblib",
    "72h": "fireguard_real_firms_sensor_only_72h_v1.joblib",
}


# ============================================================================
# کلاس سرویس تولید مدل فصلی
# ============================================================================

class SeasonalFireModelService:
    """
    سرویس پیش‌بینی فصلی آتش‌سوزی جنگل‌های هیرکانی
    
    ویژگی‌های فصلی:
    - spring (بهار): 0
    - summer (تابستان): 1
    - autumn (پاییز): 2
    - winter (زمستان): 3
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}
        self._load_thresholds()
        self._load_models()
    
    # ------------------------------------------------------------------------
    # بارگذاری آستانه‌ها
    # ------------------------------------------------------------------------
    
    def _load_thresholds(self):
        """بارگذاری آستانه‌های فصلی از فایل JSON"""
        
        if not THRESHOLD_CONFIG.exists():
            print(f"⚠️ فایل آستانه پیدا نشد، استفاده از مقادیر پیش‌فرض")
            self.thresholds = DEFAULT_THRESHOLDS.copy()
            return
        
        with open(THRESHOLD_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        thresholds = config.get("thresholds", {})
        
        for horizon in ["24h", "48h", "72h"]:
            if horizon in thresholds:
                self.thresholds[horizon] = float(thresholds[horizon])
            else:
                self.thresholds[horizon] = DEFAULT_THRESHOLDS[horizon]
    
    # ------------------------------------------------------------------------
    # بارگذاری مدل‌ها
    # ------------------------------------------------------------------------
    
    def _load_models(self):
        """بارگذاری سه مدل فصلی (24h, 48h, 72h)"""
        
        print("🔄 در حال بارگذاری مدل‌های فصلی...")
        
        for horizon, filename in MODEL_FILES.items():
            model_path = MODEL_DIR / filename
            
            if not model_path.exists():
                print(f"⚠️ مدل {horizon} یافت نشد: {model_path}")
                continue
            
            try:
                artifact = joblib.load(model_path)
                
                # استخراج مدل از دیکشنری
                if isinstance(artifact, dict):
                    model = artifact.get("model")
                    if model is None:
                        model = artifact.get("estimator")
                else:
                    model = artifact
                
                if model is None or not hasattr(model, "predict_proba"):
                    raise ValueError("مدل معتبر نیست یا predict_proba ندارد")
                
                self.models[horizon] = model
                print(f"✅ مدل {horizon} بارگذاری شد")
                
            except Exception as e:
                print(f"❌ خطا در بارگذاری مدل {horizon}: {e}")
        
        if len(self.models) == 0:
            raise RuntimeError("❌ هیچ مدلی بارگذاری نشد!")
    
    # ------------------------------------------------------------------------
    # تبدیل داده‌ها به ویژگی‌های مدل
    # ------------------------------------------------------------------------
    
    def _prepare_features(self, record: Dict[str, Any]) -> pd.DataFrame:
        """تبدیل رکورد ورودی به ۱۵ ویژگی مدل"""
        
        # بررسی وجود ویژگی‌های مورد نیاز
        required = ["latitude", "longitude", "brightness", "scan", "track",
                    "confidence", "bright_t31", "frp", "hour", "minute"]
        
        missing = [col for col in required if col not in record]
        if missing:
            raise ValueError(f"ویژگی‌های زیر وجود ندارند: {missing}")
        
        # ساخت DataFrame
        df = pd.DataFrame([record])
        
        # --------------------------------------------------------------------
        # ۱. کدگذاری daynight
        # --------------------------------------------------------------------
        if "daynight" in record:
            daynight_map = {"D": 0, "DAY": 0, "N": 1, "NIGHT": 1}
            df["daynight_encoded"] = df["daynight"].map(daynight_map).fillna(0)
        elif "daynight_encoded" in record:
            df["daynight_encoded"] = record["daynight_encoded"]
        else:
            # پیش‌فرض: روز (D)
            df["daynight_encoded"] = 0
        
        # --------------------------------------------------------------------
        # ۲. کدگذاری satellite
        # --------------------------------------------------------------------
        if "satellite" in record:
            satellite_map = {"AQUA": 0, "TERRA": 1, "SNPP": 2, "NOAA-20": 3}
            df["satellite_encoded"] = df["satellite"].map(satellite_map).fillna(-1)
        elif "satellite_encoded" in record:
            df["satellite_encoded"] = record["satellite_encoded"]
        else:
            df["satellite_encoded"] = -1
        
        # --------------------------------------------------------------------
        # ۳. کدگذاری instrument
        # --------------------------------------------------------------------
        if "instrument" in record:
            instrument_map = {"MODIS": 0, "VIIRS": 1}
            df["instrument_encoded"] = df["instrument"].map(instrument_map).fillna(-1)
        elif "instrument_encoded" in record:
            df["instrument_encoded"] = record["instrument_encoded"]
        else:
            df["instrument_encoded"] = -1
        
        # --------------------------------------------------------------------
        # ۴. کدگذاری type
        # --------------------------------------------------------------------
        if "type" in record:
            # اگر type عددی باشد، همان را استفاده کن
            if isinstance(record["type"], (int, float)):
                df["type_encoded"] = record["type"]
            else:
                df["type_encoded"] = 0
        elif "type_encoded" in record:
            df["type_encoded"] = record["type_encoded"]
        else:
            df["type_encoded"] = 0
        
        # --------------------------------------------------------------------
        # ۵. کدگذاری season (فصل)
        # --------------------------------------------------------------------
        if "season" in record:
            season_map = {
                "spring": 0, "Spring": 0, "SPRING": 0,
                "summer": 1, "Summer": 1, "SUMMER": 1,
                "autumn": 2, "Autumn": 2, "AUTUMN": 2,
                "fall": 2, "Fall": 2,
                "winter": 3, "Winter": 3, "WINTER": 3
            }
            df["season_encoded"] = df["season"].map(season_map).fillna(0)
        elif "season_encoded" in record:
            df["season_encoded"] = record["season_encoded"]
        else:
            # تشخیص خودکار فصل از ماه
            if "month" in record:
                month = record["month"]
            elif "date" in record:
                month = pd.to_datetime(record["date"]).month
            else:
                # پیش‌فرض: بهار
                month = 3
            
            if month in [3, 4, 5]:
                df["season_encoded"] = 0  # spring
            elif month in [6, 7, 8]:
                df["season_encoded"] = 1  # summer
            elif month in [9, 10, 11]:
                df["season_encoded"] = 2  # autumn
            else:
                df["season_encoded"] = 3  # winter
        
        # --------------------------------------------------------------------
        # ۶. انتخاب ۱۵ ویژگی به ترتیب
        # --------------------------------------------------------------------
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"ویژگی {col} در داده‌ها وجود ندارد")
        
        X = df[FEATURE_COLUMNS].copy()
        
        # تبدیل به عددی
        for col in FEATURE_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        
        # بررسی missing values
        if X.isna().any().any():
            raise ValueError("داده‌های ورودی دارای مقادیر缺失 هستند")
        
        return X
    
    # ------------------------------------------------------------------------
    # پیش‌بینی تک رکورد
    # ------------------------------------------------------------------------
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        پیش‌بینی برای یک رکورد ورودی
        
        Args:
            record: دیکشنری شامل ویژگی‌های مورد نیاز
            
        Returns:
            {
                "prob_24h": 0.73,
                "pred_24h": 1,
                "prob_48h": 0.65,
                "pred_48h": 1,
                "prob_72h": 0.45,
                "pred_72h": 0
            }
        """
        
        # آماده‌سازی ویژگی‌ها
        X = self._prepare_features(record)
        
        results = {}
        
        for horizon in ["24h", "48h", "72h"]:
            if horizon not in self.models:
                results[f"prob_{horizon}"] = None
                results[f"pred_{horizon}"] = None
                continue
            
            model = self.models[horizon]
            threshold = self.thresholds.get(horizon, 0.35)
            
            # پیش‌بینی
            probabilities = model.predict_proba(X)[:, 1]
            probability = float(probabilities[0])
            prediction = 1 if probability >= threshold else 0
            
            results[f"prob_{horizon}"] = probability
            results[f"pred_{horizon}"] = prediction
        
        return results
    
    # ------------------------------------------------------------------------
    # پیش‌بینی دسته‌ای
    # ------------------------------------------------------------------------
    
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """پیش‌بینی برای چند رکورد"""
        
        results = []
        for record in records:
            results.append(self.predict(record))
        return results
    
    # ------------------------------------------------------------------------
    # وضعیت سرویس
    # ------------------------------------------------------------------------
    
    def status(self) -> Dict[str, Any]:
        """گزارش وضعیت سرویس"""
        
        return {
            "service": "FIREGUARD SEASONAL FIRE MODEL",
            "region": "Hyrcanian Forests",
            "period": "2001-2025",
            "features": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "models_loaded": list(self.models.keys()),
            "thresholds": self.thresholds,
            "model_dir": str(MODEL_DIR),
            "status": "READY"
        }


# ============================================================================
# توابع کمکی برای استفاده سریع
# ============================================================================

def get_service() -> SeasonalFireModelService:
    """دریافت نمونه سرویس (Singleton)"""
    global _service
    if "_service" not in globals():
        _service = SeasonalFireModelService()
    return _service


def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    """پیش‌بینی سریع تک رکورد"""
    service = get_service()
    return service.predict(record)


def predict_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """پیش‌بینی سریع دسته‌ای"""
    service = get_service()
    return service.predict_batch(records)


# ============================================================================
# تست مستقل
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔥 FIREGUARD — سرویس مدل فصلی جنگل‌های هیرکانی")
    print("=" * 70)
    
    # بارگذاری سرویس
    service = get_service()
    
    # نمایش وضعیت
    print("\n📊 وضعیت سرویس:")
    status = service.status()
    for key, value in status.items():
        if key != "features":
            print(f"  {key}: {value}")
    
    # تست با یک نمونه
    print("\n🧪 تست پیش‌بینی با داده‌های نمونه:")
    
    sample_record = {
        "latitude": 37.0,
        "longitude": 50.0,
        "brightness": 310.5,
        "scan": 1.0,
        "track": 1.0,
        "confidence": 85.0,
        "bright_t31": 305.0,
        "frp": 45.0,
        "hour": 14,
        "minute": 30,
        "daynight": "D",
        "satellite": "TERRA",
        "instrument": "MODIS",
        "type": 0,
        "season": "summer"
    }
    
    try:
        result = service.predict(sample_record)
        print("\n📈 نتایج پیش‌بینی:")
        for horizon in ["24h", "48h", "72h"]:
            prob = result[f"prob_{horizon}"]
            pred = result[f"pred_{horizon}"]
            status_text = "🔥 خطر آتش" if pred else "✅ ایمن"
            print(f"  {horizon}: {prob*100:.1f}% → {status_text}")
    except Exception as e:
        print(f"❌ خطا در پیش‌بینی: {e}")
    
    print("\n" + "=" * 70)
    print("✅ سرویس مدل فصلی آماده است!")
    print("=" * 70)