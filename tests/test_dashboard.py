def sensor_status(timestamp_age_seconds, stale_limit=60):
    if timestamp_age_seconds > stale_limit:
        return "STALE DATA"

    return "LIVE"


def forecast_status(forecast):
    if forecast is None:
        return "FORECAST UNAVAILABLE"

    return "AVAILABLE"


def format_probability(probability):
    if probability is None:
        return "N/A"

    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability")

    return f"{probability:.1%}"


def test_live_data():
    assert sensor_status(10) == "LIVE"


def test_stale_data():
    assert sensor_status(120) == "STALE DATA"


def test_forecast_unavailable():
    assert forecast_status(None) == "FORECAST UNAVAILABLE"


def test_forecast_available():
    assert forecast_status({
        "risk_72h": 0.5
    }) == "AVAILABLE"


def test_probability_display():
    assert format_probability(0.73) == "73.0%"


def test_invalid_probability_display():
    try:
        format_probability(1.5)
        assert False
    except ValueError:
        assert True