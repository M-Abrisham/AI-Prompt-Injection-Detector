"""Safe environment-variable parsing helpers.

Centralised here so every module uses the same crash-proof pattern
instead of bare ``int(os.getenv(...))``.
"""

import math
import os


def safe_int_env(name, default, lo=0, hi=None):
    """Read an int from *name*, clamping to [lo, hi]. Falls back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


def safe_float_env(name, default, lo=0.0, hi=float("inf")):
    """Read a float from *name*, clamping to [lo, hi]. Falls back to *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(val):
        return default
    if val < lo or val > hi:
        return default
    return val
