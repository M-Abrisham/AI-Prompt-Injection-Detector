# src/na0s/multi_turn_validator.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.multi_turn"""
import warnings as _warnings
_warnings.warn("na0s.multi_turn_validator is deprecated; use na0s.detectors.multi_turn instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import multi_turn as _mod  # noqa: E402
from na0s.detectors.multi_turn import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
