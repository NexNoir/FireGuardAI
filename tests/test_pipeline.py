# tests/test_pipeline.py
from datetime import datetime, timedelta
from sensors.data_validator import validate_reading, detect_stuck
from features.feature_builder import build_features

def test_validate_good_reading():
    raw = {
        "temp": 28.5,
        "humidity": 55.0,
        "smoke": 420,
        "flame": 0,
        "time": datetime.now().isoformat()
    }
    result = validate_reading(raw)
    assert result["is_valid"] is True
    assert result["temperature"] == 28.5
    print("✓ test_validate_good_reading passed")

def test_validate_impossible_temp():
    raw = {
        "temp": -55.0,
        "humidity": 50,
        "smoke": 100,
        "flame": 0,
        "time": datetime.now().isoformat()
    }
    result = validate_reading(raw)
    assert result["is_valid"] is False
    assert "temperature_impossible" in result["errors"]
    print("✓ test_validate_impossible_temp passed")

def test_feature_no_leakage():
    now = datetime.now()
    history = []
    for i in range(5):
        history.append({
            "temperature": 25 + i * 0.5,
            "humidity": 50 - i,
            "smoke": 300 + i * 20,
            "flame": 0,
            "timestamp": now - timedelta(seconds=(4 - i) * 5)
        })
    feats = build_features(history)
    assert feats is not None
    assert "temperature" in feats
    assert "smoke_delta" in feats
    assert "smoke_rolling_mean" in feats
    assert "pressure" not in feats          # نباید وجود داشته باشد
    assert "wind" not in feats
    assert "fwi" not in feats
    assert "frp" not in feats
    print("✓ test_feature_no_leakage passed")
    print("  Sample features:", {k: round(v, 3) if isinstance(v, float) else v for k, v in list(feats.items())[:8]})

if __name__ == "__main__":
    test_validate_good_reading()
    test_validate_impossible_temp()
    test_feature_no_leakage()
    print("\nهمه تست‌های پایه پاس شدند.")