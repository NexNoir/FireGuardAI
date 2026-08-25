from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from alerts.sms_service import SmsService
from alert_engine.alert_engine import (
    AlertEngine,
    AlertInput,
    AlertResult,
)
from alert_engine.alert_store import AlertStore


try:
    DEFAULT_COOLDOWN = int(
        os.getenv("SMS_COOLDOWN_MINUTES", "15")
    )
except ValueError:
    DEFAULT_COOLDOWN = 15


class AlertService:
    """
    Integration layer between the existing AlertEngine,
    persistent cooldown and SMS.

    IMPORTANT:
    AlertEngine remains the authority for alert level.
    SMS never changes ML probability.
    """

    def __init__(
        self,
        alert_engine: Optional[AlertEngine] = None,
        sms_service: Optional[SmsService] = None,
        alert_store: Optional[AlertStore] = None,
        cooldown_minutes: int = DEFAULT_COOLDOWN,
    ):
        self.alert_engine = (
            alert_engine or AlertEngine()
        )

        self.sms_service = (
            sms_service or SmsService()
        )

        self.alert_store = (
            alert_store or AlertStore()
        )

        self.cooldown_minutes = max(
            0,
            int(cooldown_minutes),
        )

    # =========================================================
    # COOLDOWN
    # =========================================================

    def cooldown_active(self) -> bool:
        last_sms = self.alert_store.get_last_sms_at()

        if last_sms is None:
            return False

        now = datetime.now(timezone.utc)

        elapsed = (
            now - last_sms
        ).total_seconds()

        return elapsed < (
            self.cooldown_minutes * 60
        )

    def cooldown_remaining_seconds(self) -> int:
        last_sms = self.alert_store.get_last_sms_at()

        if last_sms is None:
            return 0

        now = datetime.now(timezone.utc)

        elapsed = (
            now - last_sms
        ).total_seconds()

        remaining = (
            self.cooldown_minutes * 60
        ) - elapsed

        return max(0, int(remaining))

    # =========================================================
    # DATABASE EVENT / ALERT
    # =========================================================

    @staticmethod
    def _alert_status(level_value: int) -> str:
        if level_value >= 4:
            return "active"

        if level_value >= 3:
            return "active"

        return "active"

    def _persist_alert(
        self,
        result: AlertResult,
    ) -> dict:
        """
        Persist fire_event + alert using the existing
        FireGuardDatabase API.
        """

        from database.db import FireGuardDatabase

        db = FireGuardDatabase()

        try:
            db.add_fire_event(
                event_id=result.event_id,
                timestamp=result.created_at,
                event_type="fire_risk",
                status="open",
                description=" | ".join(
                    result.reasons
                ),
            )

            db.add_alert(
                event_id=result.event_id,
                alert_level=result.level,
                reason=" | ".join(
                    result.reasons
                ),
                status="active",
            )

            return {
                "success": True,
                "event_id": result.event_id,
                "error": None,
            }

        except Exception as exc:
            return {
                "success": False,
                "event_id": result.event_id,
                "error": str(exc),
            }

    # =========================================================
    # FULL PIPELINE
    # =========================================================

    def evaluate_and_notify(
        self,
        data: AlertInput,
        persist_alert: bool = True,
    ) -> dict:
        """
        1. Evaluate existing AlertEngine.
        2. Persist fire event / alert when needed.
        3. Apply persistent SMS cooldown.
        4. Send SMS only when allowed.
        """

        result: AlertResult = (
            self.alert_engine.evaluate(data)
        )

        output = {
            "alert_result": result.to_dict(),
            "alert_recorded": False,
            "alert_record_error": None,
            "sms_attempted": False,
            "sms_sent": False,
            "sms_cooldown": False,
            "sms_message": None,
        }

        # -----------------------------------------------------
        # No alert
        # -----------------------------------------------------

        if result.level_value <= 0:
            output["sms_message"] = (
                "شرایط هشدار وجود ندارد"
            )
            return output

        # -----------------------------------------------------
        # Persist alert
        # -----------------------------------------------------

        if persist_alert:
            persistence = self._persist_alert(
                result
            )

            output["alert_recorded"] = (
                persistence["success"]
            )

            output["alert_record_error"] = (
                persistence["error"]
            )

        # -----------------------------------------------------
        # Persistent cooldown
        # -----------------------------------------------------

        if self.cooldown_active():

            remaining = (
                self.cooldown_remaining_seconds()
            )

            output["sms_cooldown"] = True

            output["sms_message"] = (
                "پیامک در حالت محدودیت ارسال است؛ "
                f"{remaining} ثانیه باقی مانده"
            )

            return output

        # -----------------------------------------------------
        # Send SMS
        # -----------------------------------------------------

        probability = result.fire_probability

        output["sms_attempted"] = True

        sms_result = self.sms_service.send_alert(
            risk_level=result.level,
            probability=probability,
            temperature=data.temperature,
            humidity=data.humidity,
            smoke=data.smoke,
            flame=data.flame,
            source=data.source,
        )

        success = bool(
            sms_result.get(
                "success",
                False,
            )
        )

        if success:

            self.alert_store.save_sms_success(
                timestamp=datetime.now(timezone.utc),
                risk_level=result.level,
                probability=float(
                    probability
                    if probability is not None
                    else 0.0
                ),
            )

            output["sms_sent"] = True
            output["sms_message"] = (
                "پیامک هشدار با موفقیت ارسال شد"
            )

        else:

            self.alert_store.save_sms_failure(
                timestamp=datetime.now(timezone.utc),
                risk_level=result.level,
                probability=float(
                    probability
                    if probability is not None
                    else 0.0
                ),
            )

            output["sms_message"] = (
                sms_result.get("error")
                or "ارسال پیامک ناموفق بود"
            )

        return output