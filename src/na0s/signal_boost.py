# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.fusion.signal_boost"""
import sys as _sys
import warnings as _warnings
from na0s.fusion import signal_boost as _canonical

_warnings.warn(
    "na0s.signal_boost is deprecated; use na0s.fusion.signal_boost instead",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _canonical
