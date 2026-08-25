# features/feature_builder.py
from typing import List, Dict, Any, Optional
from datetime import datetime
import statistics
import config

def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def build_features(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    از تاریخچه خوانش‌های معتبر، ویژگی‌های واقعی و بدون leakage می‌سازد.
    history باید به ترتیب زمانی صعودی باشد (قدیمی → جدید).
    فقط از داده‌های گذشته و حال استفاده می‌کند.
    """
    if not history or len(history) < 1:
        return None

    current = history[-1]
    prev = history[-2] if len(history) >= 2 else None

    features = {
        "temperature": _safe_float(current.get("temperature")),
        "humidity": _safe_float(current.get("humidity")),
        "smoke": _safe_float(current.get("smoke")),
        "flame": int(current.get("flame", 0) or 0),
    }

    # timestamp و ویژگی‌های زمانی ساده (بدون leakage)
    ts = current.get("timestamp")
    if isinstance(ts, datetime):
        features["hour"] = ts.hour
        features["month"] = ts.month
    else:
        features["hour"] = None
        features["month"] = None

    # --- delta ---
    if prev is not None:
        features["temperature_delta"] = _safe_float(current.get("temperature")) - _safe_float(prev.get("temperature"), 0)
        features["humidity_delta"] = _safe_float(current.get("humidity")) - _safe_float(prev.get("humidity"), 0)
        features["smoke_delta"] = _safe_float(current.get("smoke")) - _safe_float(prev.get("smoke"), 0)
    else:
        features["temperature_delta"] = 0.0
        features["humidity_delta"] = 0.0
        features["smoke_delta"] = 0.0

    # --- rate (بر اساس اختلاف زمان واقعی) ---
    if prev is not None and isinstance(current.get("timestamp"), datetime) and isinstance(prev.get("timestamp"), datetime):
        dt_seconds = (current["timestamp"] - prev["timestamp"]).total_seconds()
        if dt_seconds <= 0:
            dt_seconds = config.DEFAULT_SAMPLE_SECONDS
        features["temperature_rate"] = features["temperature_delta"] / dt_seconds
        features["humidity_rate"] = features["humidity_delta"] / dt_seconds
        features["smoke_rate"] = features["smoke_delta"] / dt_seconds
    else:
        features["temperature_rate"] = 0.0
        features["humidity_rate"] = 0.0
        features["smoke_rate"] = 0.0

    # --- rolling stats (فقط گذشته + حال) ---
    window = min(config.ROLLING_WINDOW, len(history))
    recent = history[-window:]

    temps = [_safe_float(r.get("temperature")) for r in recent if _safe_float(r.get("temperature")) is not None]
    hums = [_safe_float(r.get("humidity")) for r in recent if _safe_float(r.get("humidity")) is not None]
    smokes = [_safe_float(r.get("smoke")) for r in recent if _safe_float(r.get("smoke")) is not None]

    def _rolling(values):
        if len(values) == 0:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }

    t_roll = _rolling(temps)
    h_roll = _rolling(hums)
    s_roll = _rolling(smokes)

    features["temp_rolling_mean"] = t_roll["mean"]
    features["temp_rolling_std"] = t_roll["std"]
    features["temp_rolling_min"] = t_roll["min"]
    features["temp_rolling_max"] = t_roll["max"]

    features["humidity_rolling_mean"] = h_roll["mean"]
    features["humidity_rolling_std"] = h_roll["std"]
    features["humidity_rolling_min"] = h_roll["min"]
    features["humidity_rolling_max"] = h_roll["max"]

    features["smoke_rolling_mean"] = s_roll["mean"]
    features["smoke_rolling_std"] = s_roll["std"]
    features["smoke_rolling_min"] = s_roll["min"]
    features["smoke_rolling_max"] = s_roll["max"]

    # --- acceleration ساده (تغییر rate) ---
    # برای محاسبه نیاز به حداقل ۳ نقطه داریم
    if len(history) >= 3:
        prev2 = history[-3]
        # rate قبلی
        if (isinstance(prev.get("timestamp"), datetime) and 
            isinstance(prev2.get("timestamp"), datetime)):
            dt_prev = (prev["timestamp"] - prev2["timestamp"]).total_seconds()
            if dt_prev <= 0:
                dt_prev = config.DEFAULT_SAMPLE_SECONDS
            temp_delta_prev = _safe_float(prev.get("temperature"), 0) - _safe_float(prev2.get("temperature"), 0)
            smoke_delta_prev = _safe_float(prev.get("smoke"), 0) - _safe_float(prev2.get("smoke"), 0)
            temp_rate_prev = temp_delta_prev / dt_prev
            smoke_rate_prev = smoke_delta_prev / dt_prev
            features["temperature_acceleration"] = features["temperature_rate"] - temp_rate_prev
            features["smoke_acceleration"] = features["smoke_rate"] - smoke_rate_prev
        else:
            features["temperature_acceleration"] = 0.0
            features["smoke_acceleration"] = 0.0
    else:
        features["temperature_acceleration"] = 0.0
        features["smoke_acceleration"] = 0.0

    return features