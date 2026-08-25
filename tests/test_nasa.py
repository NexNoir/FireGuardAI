def validate_nasa_observation(observation):
    if observation is None:
        return None

    result = {}

    if "frp" in observation:
        try:
            frp = float(observation["frp"])
            if frp >= 0:
                result["frp"] = frp
        except (TypeError, ValueError):
            pass

    if "confidence" in observation:
        try:
            confidence = float(observation["confidence"])
            if 0 <= confidence <= 100:
                result["confidence"] = confidence
        except (TypeError, ValueError):
            pass

    return result


def test_nasa_valid_observation():
    result = validate_nasa_observation({
        "frp": 120,
        "confidence": 85,
    })

    assert result["frp"] == 120
    assert result["confidence"] == 85


def test_nasa_unavailable():
    assert validate_nasa_observation(None) is None


def test_no_fake_frp():
    result = validate_nasa_observation({
        "confidence": 80,
    })

    assert "frp" not in result


def test_invalid_frp_rejected():
    result = validate_nasa_observation({
        "frp": -10,
    })

    assert "frp" not in result


def test_invalid_confidence_rejected():
    result = validate_nasa_observation({
        "confidence": 150,
    })

    assert "confidence" not in result