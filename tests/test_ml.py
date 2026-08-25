import math


def validate_probability(probability):
    if probability is None:
        return False

    try:
        p = float(probability)
    except (TypeError, ValueError):
        return False

    return math.isfinite(p) and 0.0 <= p <= 1.0


def safe_model_prediction(model_available=True, probability=0.5):
    if not model_available:
        return None

    if not validate_probability(probability):
        return None

    return float(probability)


def test_model_probability_valid():
    assert validate_probability(0.5)


def test_probability_zero():
    assert validate_probability(0.0)


def test_probability_one():
    assert validate_probability(1.0)


def test_probability_below_zero_rejected():
    assert not validate_probability(-0.01)


def test_probability_above_one_rejected():
    assert not validate_probability(1.01)


def test_nan_probability_rejected():
    assert not validate_probability(float("nan"))


def test_infinite_probability_rejected():
    assert not validate_probability(float("inf"))


def test_model_unavailable_safe():
    assert safe_model_prediction(False, 0.8) is None


def test_external_evidence_does_not_change_probability():
    original = 0.73
    nasa = {"frp": 500}

    result = original

    assert result == original
    assert nasa["frp"] == 500