import math


def clean_sensor_data(data):
    required = ["temperature", "humidity", "smoke", "flame"]

    if not all(k in data for k in required):
        return None

    try:
        values = {
            "temperature": float(data["temperature"]),
            "humidity": float(data["humidity"]),
            "smoke": float(data["smoke"]),
            "flame": int(data["flame"]),
        }
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for k, v in values.items() if k != "flame"):
        return None

    if not -50 <= values["temperature"] <= 80:
        return None

    if not 0 <= values["humidity"] <= 100:
        return None

    if values["smoke"] < 0:
        return None

    if values["flame"] not in (0, 1):
        return None

    return values


def test_pipeline_valid_data():
    result = clean_sensor_data({
        "temperature": 25,
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert result is not None


def test_pipeline_missing_data():
    result = clean_sensor_data({
        "temperature": 25,
        "humidity": 45,
    })

    assert result is None


def test_pipeline_invalid_temperature():
    result = clean_sensor_data({
        "temperature": 999,
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert result is None


def test_pipeline_nan():
    result = clean_sensor_data({
        "temperature": float("nan"),
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert result is None


def test_pipeline_smoke_spike_is_kept_as_sensor_data():
    result = clean_sensor_data({
        "temperature": 25,
        "humidity": 45,
        "smoke": 5000,
        "flame": 0,
    })

    assert result is not None
    assert result["smoke"] == 5000