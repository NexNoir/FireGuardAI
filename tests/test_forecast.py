def validate_forecast(forecast):
    if forecast is None:
        return False

    required = ["current_risk", "risk_24h", "risk_48h", "risk_72h"]

    if not all(k in forecast for k in required):
        return False

    for key in required:
        value = forecast[key]

        if value is None:
            return False

        if not 0 <= float(value) <= 1:
            return False

    return True


def test_valid_forecast():
    forecast = {
        "current_risk": 0.2,
        "risk_24h": 0.3,
        "risk_48h": 0.4,
        "risk_72h": 0.5,
    }

    assert validate_forecast(forecast)


def test_forecast_unavailable():
    assert not validate_forecast(None)


def test_forecast_probability_bounds():
    forecast = {
        "current_risk": 0.2,
        "risk_24h": 0.3,
        "risk_48h": 0.4,
        "risk_72h": 1.2,
    }

    assert not validate_forecast(forecast)