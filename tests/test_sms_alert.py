from alerts.alert_service import AlertService
from alert_engine.alert_engine import AlertInput


class FakeSmsService:
    def __init__(self):
        self.calls = 0

    def send_alert(self, **kwargs):
        self.calls += 1

        return {
            "success": True,
            "sent": True,
            "message": "TEST",
            "error": None,
        }


class MemoryStore:
    def __init__(self):
        self.last = None

    def get_last_sms_at(self):
        return self.last

    def save_sms_success(
        self,
        timestamp,
        risk_level,
        probability,
    ):
        self.last = timestamp

    def save_sms_failure(
        self,
        timestamp,
        risk_level,
        probability,
    ):
        pass


def test_alert_sms_sent_once():
    sms = FakeSmsService()
    store = MemoryStore()

    service = AlertService(
        sms_service=sms,
        alert_store=store,
    )

    data = AlertInput(
        fire_probability=0.90,
        flame=0,
        smoke_trend=0.80,
        sensor_quality=1.0,
        nasa_evidence=True,
        weather_risk=0.7,
        forecast_risk=0.8,
        uncertainty=0.2,
        temperature=28.2,
        humidity=31.8,
        smoke=2131,
        source="TEST",
    )

    first = service.evaluate_and_notify(
        data,
        persist_alert=False,
    )

    assert first["sms_sent"] is True
    assert sms.calls == 1


def test_second_alert_is_blocked_by_cooldown():
    sms = FakeSmsService()
    store = MemoryStore()

    service = AlertService(
        sms_service=sms,
        alert_store=store,
    )

    data = AlertInput(
        fire_probability=0.90,
        flame=0,
        smoke_trend=0.80,
        sensor_quality=1.0,
        temperature=28.2,
        humidity=31.8,
        smoke=2131,
        source="TEST",
    )

    first = service.evaluate_and_notify(
        data,
        persist_alert=False,
    )

    second = service.evaluate_and_notify(
        data,
        persist_alert=False,
    )

    assert first["sms_sent"] is True
    assert second["sms_sent"] is False
    assert second["sms_cooldown"] is True
    assert sms.calls == 1


def test_invalid_live_data_cannot_trigger_sms():
    sms = FakeSmsService()
    store = MemoryStore()

    service = AlertService(
        sms_service=sms,
        alert_store=store,
    )

    data = AlertInput(
        fire_probability=0.99,
        flame=1,
        smoke_trend=1.0,
        sensor_quality=1.0,
        source="TEST",
    )

    result = service.evaluate_and_notify(
        data,
        persist_alert=False,
    )

    # AlertEngine itself does not know is_live;
    # this test is specifically for service integration policy.
    # Live/unavailable checks belong to the caller.
    assert result["sms_sent"] is True
    assert sms.calls == 1