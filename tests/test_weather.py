def validate_weather(weather):
    if weather is None:
        return None

    required = ["temperature", "humidity"]

    if not all(k in weather for k in required):
        return None

    try:
        temperature = float(weather["temperature"])
        humidity = float(weather["humidity"])
    except (TypeError, ValueError):
        return None

    if not -90 <= temperature <= 70:
        return None

    if not 0 <= humidity <= 100:
        return None

    return {
        "temperature": temperature,
        "humidity": humidity,
    }


def test_weather_valid():
    result = validate_weather({
        "temperature": 25,
        "humidity": 50,
    })

    assert result is not None


def test_weather_unavailable():
    assert validate_weather(None) is None


def test_no_fake_weather():
    result = validate_weather({})

    assert result is None


def test_invalid_weather_rejected():
    result = validate_weather({
        "temperature": "unknown",
        "humidity": 50,
    })

    assert result is None