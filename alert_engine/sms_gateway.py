
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


class SMSGateway:
    """
    SMS gateway safety wrapper.

    IMPORTANT:
    Real SMS is deliberately disabled in Stage 10.
    """

    SMS_ENABLED = False
    DRY_RUN = True
    COOLDOWN_MINUTES = 15

    def __init__(self):
        self.last_sent_at: datetime | None = None

    def send(
        self,
        event_id: str,
        level: str,
        message: str,
        *,
        force: bool = False,
    ) -> tuple[bool, str]:

        # ----------------------------------------------------------
        # HARD SAFETY BLOCK
        # ----------------------------------------------------------
        if not self.SMS_ENABLED:
            return (
                False,
                "SMS BLOCKED: SMS_ENABLED=False",
            )

        if self.DRY_RUN:
            return (
                False,
                "SMS BLOCKED: DRY_RUN=True",
            )

        now = datetime.now()

        if not force and self.last_sent_at is not None:
            elapsed = now - self.last_sent_at
            cooldown = timedelta(minutes=self.COOLDOWN_MINUTES)

            if elapsed < cooldown:
                remaining = cooldown - elapsed

                return (
                    False,
                    f"SMS BLOCKED: cooldown active "
                    f"({int(remaining.total_seconds())} seconds remaining)",
                )

        # ----------------------------------------------------------
        # REAL SMS IMPLEMENTATION IS INTENTIONALLY NOT PRESENT.
        #
        # Do not add Kavenegar credentials here during Stage 10.
        # ----------------------------------------------------------

        return (
            False,
            "SMS BLOCKED: real gateway is not enabled in Stage 10",
        )
