
"""
FIREGUARD — Clean Production Runtime

Purpose:
    Keep production inference output clean by suppressing
    known non-fatal sklearn/joblib warnings.

This file does NOT:
    - retrain models
    - modify models
    - modify datasets
"""

import os
import warnings


# ----------------------------------------------------------------------
# Reduce unnecessary joblib/scikit-learn worker noise
# ----------------------------------------------------------------------

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


# ----------------------------------------------------------------------
# Suppress known non-fatal sklearn parallel warning
# ----------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


# ----------------------------------------------------------------------
# General known sklearn warning cleanup
# ----------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)


def configure_runtime():
    """
    Apply production runtime configuration.

    Call this BEFORE loading the FireGuard models.
    """
    return True
