"""Shared environment-variable parsing helpers for Layer 2 modules.

Consolidates the previously duplicated `_safe_float_env` / `_safe_int_env`
helpers from `ascii_art_detector.py` and `whitespace_stego.py`, and replaces
the less-safe `_env_float` / `_env_int` variants previously defined in
`obfuscation.py`.

Both helpers:
  - Return *default* when the variable is absent, empty, or unparseable.
  - Reject non-finite floats (NaN, +/-inf) — guards against operators
    setting e.g. `NA0S_PUNCTUATION_FLOOD_RATIO=inf`.
  - Optionally clamp to a [lo, hi] range; values outside the range fall
    back to *default* (they are NOT silently clamped to the boundary, to
    make configuration errors loud rather than silent).

Range parameters are optional — callers that historically used the
unbounded `_env_float` / `_env_int` can pass `lo=None, hi=None` (the
default) to preserve the existing unbounded behaviour while still
benefiting from finite-ness checks.
"""

import math
import os


def safe_float_env(name, default, lo=None, hi=None):
    """Read a float from the environment, with safety checks.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Value returned when the variable is absent, unparseable, non-finite,
        or outside the [lo, hi] range.
    lo : float or None
        Lower bound (inclusive).  ``None`` disables the lower-bound check.
    hi : float or None
        Upper bound (inclusive).  ``None`` disables the upper-bound check.

    Returns
    -------
    float
        The parsed value, or *default* if any check fails.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    if not math.isfinite(val):
        return default
    if lo is not None and val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val


def safe_int_env(name, default, lo=None, hi=None):
    """Read an int from the environment, with safety checks.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Value returned when the variable is absent, unparseable, or outside
        the [lo, hi] range.
    lo : int or None
        Lower bound (inclusive).  ``None`` disables the lower-bound check.
    hi : int or None
        Upper bound (inclusive).  ``None`` disables the upper-bound check.

    Returns
    -------
    int
        The parsed value, or *default* if any check fails.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        return default
    if lo is not None and val < lo:
        return default
    if hi is not None and val > hi:
        return default
    return val
