from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

SMS_ENABLED = (
    os.getenv("SMS_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

KAVENEGAR_API_KEY = os.getenv(
    "KAVENEGAR_API_KEY",
    "",
).strip()

SMS_SENDER = os.getenv(
    "SMS_SENDER",
    "",
).strip()

SMS_RECEIVER = os.getenv(
    "SMS_RECEIVER",
    "",
).strip()

try:
    SMS_COOLDOWN_MINUTES = max(
        1,
        int(os.getenv("SMS_COOLDOWN_MINUTES", "15")),
    )
except (TypeError, ValueError):
    SMS_COOLDOWN_MINUTES = 15


try:
    from kavenegar import KavenegarAPI
except ImportError:
    KavenegarAPI = None


# ============================================================
# HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    try:

        if value is None:
            return default

        result = float(value)

        # NaN check
        if result != result:
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return default


# ============================================================
# FORMAT PROBABILITY
# ============================================================

def _probability_text(
    value: Any,
) -> str:

    probability = _safe_float(value)

    if probability is None:
        return "نامشخص"

    # اگر مقدار به صورت درصد وارد شده باشد
    # مثال: 97 -> 97%
    if probability > 1:
        probability = probability / 100.0

    probability = max(
        0.0,
        min(1.0, probability),
    )

    return f"{probability:.0%}"


# ============================================================
# FORMAT RISK LEVEL
# ============================================================

def _risk_text(
    value: Any,
) -> str:

    labels = {
        "INFO": "عادی",
        "NORMAL": "عادی",
        "WATCH": "مراقبت",
        "WARNING": "هشدار",
        "HIGH": "بالا",
        "CRITICAL": "بحرانی",
    }

    level = str(
        value or "INFO"
    ).strip().upper()

    return labels.get(
        level,
        level,
    )


# ============================================================
# FORMAT SMOKE
# ============================================================

def _smoke_text(
    smoke: Any,
) -> str:

    value = _safe_float(
        smoke,
        0.0,
    )

    if value is None:
        return "نامشخص"

    try:
        elevated = float(
            os.getenv(
                "SMOKE_ELEVATED",
                "1000",
            )
        )
    except (TypeError, ValueError):
        elevated = 1000.0

    try:
        high = float(
            os.getenv(
                "SMOKE_HIGH",
                "1800",
            )
        )
    except (TypeError, ValueError):
        high = 1800.0

    try:
        very_high = float(
            os.getenv(
                "SMOKE_VERY_HIGH",
                "2500",
            )
        )
    except (TypeError, ValueError):
        very_high = 2500.0

    if value >= very_high:
        level = "بسیار بالا"

    elif value >= high:
        level = "بالا"

    elif value >= elevated:
        level = "متوسط"

    else:
        level = "عادی"

    return f"{value:.0f} ({level})"


# ============================================================
# FORMAT FLAME
# ============================================================

def _flame_text(
    flame: Any,
) -> str:

    flame_value = _safe_int(
        flame,
        0,
    )

    return (
        "بله"
        if flame_value == 1
        else "خیر"
    )


# ============================================================
# BUILD SMS MESSAGE
# ============================================================

def _build_message(
    risk_level: str,
    probability: Any,
    temperature: Any,
    humidity: Any,
    smoke: Any,
    flame: Any,
    source: str,
) -> str:

    # --------------------------------------------------------
    # FORMAT ALL VALUES
    # --------------------------------------------------------

    risk_text = _risk_text(
        risk_level
    )

    probability_text = _probability_text(
        probability
    )

    temp = _safe_float(
        temperature
    )

    humidity_value = _safe_float(
        humidity
    )

    temperature_text = (
        f"{temp:.1f}°C"
        if temp is not None
        else "نامشخص"
    )

    humidity_text = (
        f"{humidity_value:.1f}%"
        if humidity_value is not None
        else "نامشخص"
    )

    smoke_text = _smoke_text(
        smoke
    )

    flame_text = _flame_text(
        flame
    )

    source_text = str(
        source or "FireGuard"
    ).strip()

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    return (
        "🚨 هشدار آتش‌سوزی جنگل\n"
        f"سطح خطر: {risk_text}\n"
        f"احتمال: {probability_text}\n"
        f"دما: {temperature_text}\n"
        f"رطوبت: {humidity_text}\n"
        f"دود: {smoke_text}\n"
        f"شعله: {flame_text}\n"
        f"منبع: {source_text}"
    )


# ============================================================
# SMS SERVICE
# ============================================================

class SmsService:
    """
    سرویس ارسال پیامک FireGuard.

    این کلاس مستقل از Streamlit است و می‌تواند
    از Alert Engine یا تست مستقیم استفاده شود.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        api_key: Optional[str] = None,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
    ):

        self.enabled = (
            SMS_ENABLED
            if enabled is None
            else bool(enabled)
        )

        self.api_key = (
            KAVENEGAR_API_KEY
            if api_key is None
            else str(api_key).strip()
        )

        self.sender = (
            SMS_SENDER
            if sender is None
            else str(sender).strip()
        )

        self.receiver = (
            SMS_RECEIVER
            if receiver is None
            else str(receiver).strip()
        )


    # ========================================================
    # CONFIGURATION STATUS
    # ========================================================

    def configuration_status(
        self,
    ) -> Dict[str, Any]:

        errors = []

        if not self.enabled:
            errors.append(
                "SMS غیرفعال است"
            )

        if not self.api_key:
            errors.append(
                "KAVENEGAR_API_KEY تنظیم نشده است"
            )

        if not self.receiver:
            errors.append(
                "SMS_RECEIVER تنظیم نشده است"
            )

        if KavenegarAPI is None:
            errors.append(
                "کتابخانه kavenegar نصب نشده است"
            )

        return {
            "configured": len(errors) == 0,
            "enabled": self.enabled,
            "api_key_configured": bool(
                self.api_key
            ),
            "sender_configured": bool(
                self.sender
            ),
            "receiver_configured": bool(
                self.receiver
            ),
            "errors": errors,
        }


    # ========================================================
    # SEND ALERT
    # ========================================================

    def send_alert(
        self,
        risk_level: str,
        probability: Any,
        temperature: Any = None,
        humidity: Any = None,
        smoke: Any = None,
        flame: Any = 0,
        source: str = "FireGuard",
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # BUILD MESSAGE FIRST
        # ----------------------------------------------------

        message = _build_message(
            risk_level=risk_level,
            probability=probability,
            temperature=temperature,
            humidity=humidity,
            smoke=smoke,
            flame=flame,
            source=source,
        )

        # ----------------------------------------------------
        # CHECK CONFIGURATION
        # ----------------------------------------------------

        configuration = (
            self.configuration_status()
        )

        if not configuration["configured"]:

            return {
                "success": False,
                "sent": False,
                "provider": "kavenegar",
                "error": "; ".join(
                    configuration["errors"]
                ),
                "message": message,
                "provider_response": None,
            }

        # ----------------------------------------------------
        # SEND TO KAVENEGAR
        # ----------------------------------------------------

        try:

            api = KavenegarAPI(
                self.api_key
            )

            params = {
                "receptor": self.receiver,
                "message": message,
            }

            if self.sender:
                params["sender"] = self.sender

            response = api.sms_send(
                params
            )

            return {
                "success": True,
                "sent": True,
                "provider": "kavenegar",
                "error": None,
                "message": message,
                "provider_response": response,
            }

        except Exception as exc:

            return {
                "success": False,
                "sent": False,
                "provider": "kavenegar",
                "error": str(exc),
                "message": message,
                "provider_response": None,
            }


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def send_alert_sms(
    risk_level: str,
    probability: Any,
    temp: Any,
    humidity: Any,
    smoke: Any,
    flame: Any,
    source: str = "FireGuard",
) -> Tuple[bool, str]:

    result = SmsService().send_alert(
        risk_level=risk_level,
        probability=probability,
        temperature=temp,
        humidity=humidity,
        smoke=smoke,
        flame=flame,
        source=source,
    )

    if result["success"]:

        return (
            True,
            "پیامک با موفقیت به کاوه‌نگار ارسال شد",
        )

    return (
        False,
        f"ارسال پیامک ناموفق بود: "
        f"{result.get('error', 'خطای نامشخص')}",
    )