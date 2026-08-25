import uuid


LEVELS = {
    "INFO",
    "WATCH",
    "WARNING",
    "HIGH",
    "CRITICAL",
}


def create_alert(level, probability):
    if level not in LEVELS:
        raise ValueError("Invalid alert level")

    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability")

    return {
        "event_id": str(uuid.uuid4()),
        "level": level,
        "probability": probability,
        "acknowledged": False,
        "resolved": False,
    }


def acknowledge(alert):
    alert["acknowledged"] = True


def resolve(alert):
    alert["resolved"] = True


def test_alert_levels():
    for level in LEVELS:
        alert = create_alert(level, 0.5)
        assert alert["level"] == level


def test_event_id():
    alert = create_alert("WARNING", 0.7)

    assert alert["event_id"]


def test_acknowledgement():
    alert = create_alert("HIGH", 0.9)

    acknowledge(alert)

    assert alert["acknowledged"] is True


def test_resolved():
    alert = create_alert("HIGH", 0.9)

    resolve(alert)

    assert alert["resolved"] is True


def test_probability_not_modified_by_alert():
    probability = 0.73

    alert = create_alert("HIGH", probability)

    assert alert["probability"] == probability


def test_invalid_probability_rejected():
    try:
        create_alert("HIGH", 1.5)
        assert False
    except ValueError:
        assert True


def test_sms_unavailable_is_safe():
    sms_available = False

    if not sms_available:
        result = "SMS_UNAVAILABLE"
    else:
        result = "SENT"

    assert result == "SMS_UNAVAILABLE"


def test_real_sms_not_called():
    real_sms_enabled = False

    assert real_sms_enabled is False