import math


def build_features(data):
    features = [
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["smoke"]),
        float(data["flame"]),
    ]

    if not all(math.isfinite(x) for x in features):
        raise ValueError("Invalid feature value")

    return features


def test_features_created():
    features = build_features({
        "temperature": 25,
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert len(features) == 4


def test_features_no_nan():
    features = build_features({
        "temperature": 25,
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert not any(math.isnan(x) for x in features)


def test_features_no_infinity():
    features = build_features({
        "temperature": 25,
        "humidity": 45,
        "smoke": 120,
        "flame": 0,
    })

    assert not any(math.isinf(x) for x in features)


def test_invalid_feature_rejected():
    try:
        build_features({
            "temperature": float("nan"),
            "humidity": 45,
            "smoke": 120,
            "flame": 0,
        })
        assert False
    except ValueError:
        assert True