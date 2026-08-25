# -*- coding: utf-8 -*-
"""
========================================================================
🔥 FIREGUARD — START FIREGUARD V2
========================================================================

Local production startup.

Architecture:
    FireGuard
       ↓
    RealFirmsProductionService
       ↓
    24H / 48H / 72H models
       ↓
    Threshold predictions

NO API
NO HTTP
NO FASTAPI
NO UVICORN
NO RETRAINING
NO MODEL MODIFICATION
NO DATASET MODIFICATION
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\vista\Desktop\fireguard_v2.0"
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------
# SUPPRESS NON-FATAL SKLEARN WARNING
# ---------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.utils.parallel",
)


# ---------------------------------------------------------------------
# IMPORT PRODUCTION SERVICE
# ---------------------------------------------------------------------

try:
    from real_firms_service import RealFirmsProductionService

except Exception as exc:

    print("=" * 72)
    print("🔥 FIREGUARD STARTUP ERROR")
    print("=" * 72)
    print()
    print("Could not load RealFirmsProductionService.")
    print()
    print(f"Error: {exc}")
    print()
    sys.exit(1)


# ---------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------

def main():

    print("=" * 72)
    print("🔥 FIREGUARD — LOCAL PRODUCTION STARTUP V2")
    print("=" * 72)
    print()

    print("ARCHITECTURE")
    print("-" * 72)
    print("API       : NO")
    print("HTTP      : NO")
    print("FastAPI   : NO")
    print("Uvicorn   : NO")
    print("Local     : YES")
    print()

    print("LOADING REAL FIRMS PRODUCTION SERVICE")
    print("-" * 72)

    try:

        service = RealFirmsProductionService()

    except Exception as exc:

        print()
        print("❌ Production service failed to load.")
        print()
        print(f"Error: {exc}")
        print()

        sys.exit(1)

    print("Production service: PASS")
    print()

    # -----------------------------------------------------------------
    # SERVICE STATUS
    # -----------------------------------------------------------------

    print("=" * 72)
    print("PRODUCTION SERVICE STATUS")
    print("=" * 72)

    status = None

    # Support different service implementations.
    if hasattr(service, "status"):

        try:

            status = service.status()

        except TypeError:

            status = None

    elif hasattr(service, "get_status"):

        try:

            status = service.get_status()

        except TypeError:

            status = None

    if isinstance(status, dict):

        for key, value in status.items():
            print(f"{key}: {value}")

    else:

        # Fallback to known service attributes.
        feature_count = getattr(
            service,
            "feature_count",
            15,
        )

        thresholds = getattr(
            service,
            "thresholds",
            {
                "24h": 0.35,
                "48h": 0.35,
                "72h": 0.30,
            },
        )

        print(f"feature_count: {feature_count}")
        print(f"thresholds: {thresholds}")

    print()

    # -----------------------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------------------

    print("=" * 72)
    print("FINAL STARTUP CHECK")
    print("=" * 72)

    required_methods = [
        "predict",
        "predict_batch",
    ]

    all_methods_ok = True

    for method_name in required_methods:

        exists = callable(
            getattr(
                service,
                method_name,
                None,
            )
        )

        if exists:
            print(
                f"{method_name:<20}: PASS"
            )
        else:
            print(
                f"{method_name:<20}: FAIL"
            )
            all_methods_ok = False

    print()

    if not all_methods_ok:

        print("=" * 72)
        print("❌ FIREGUARD STARTUP BLOCKED")
        print("=" * 72)
        print()
        print(
            "Production service does not expose "
            "the required local inference methods."
        )

        sys.exit(1)

    print("=" * 72)
    print("STATUS: 🟢 FIREGUARD LOCAL PRODUCTION READY")
    print("=" * 72)
    print()
    print("READY FOR:")
    print("  Local application inference")
    print("  Local dashboard integration")
    print("  Local batch prediction")
    print()
    print("NOT USED:")
    print("  API")
    print("  HTTP")
    print("  FastAPI")
    print("  Uvicorn")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()