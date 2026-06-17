# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.fusion.groundedness"""
import sys as _sys
import warnings as _warnings
from na0s.fusion import groundedness as _canonical

_warnings.warn(
    "na0s.groundedness is deprecated; use na0s.fusion.groundedness instead",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _canonical
