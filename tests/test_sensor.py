import math


def validate_sensor_reading(temp, humidity, smoke, flame):
    if temp is None or not math.isfinite(float(temp)):
        return False
    if humidity is None or not math.isfinite(float(humidity)):
        return False
    if smoke is None or not math.isfinite(float(smoke)):
        return False
    if flame not in (0, 1):
        return False

    if float(temp) < -50 or float(temp) > 80:
        return False

    if float(humidity) < 0 or float(humidity) > 100:
        return False

    if float(smoke) < 0:
        return False

    return True


def test_valid_sensor_reading():
    assert validate_sensor_reading(25, 45, 120, 0)


def test_sensor_disconnected():
    assert not validate_sensor_reading(None, 45, 120, 0)


def test_bad_temperature():
    assert not validate_sensor_reading(150, 45, 120, 0)


def test_humidity_below_range():
    assert not validate_sensor_reading(25, -1, 120, 0)


def test_humidity_above_range():
    assert not validate_sensor_reading(25, 101, 120, 0)


def test_negative_smoke():
    assert not validate_sensor_reading(25, 45, -10, 0)


def test_invalid_flame():
    assert not validate_sensor_reading(25, 45, 120, 2)