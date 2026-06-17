# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.fusion.voting"""
import sys as _sys
import warnings as _warnings
from na0s.fusion import voting as _canonical

_warnings.warn(
    "na0s._voting is deprecated; use na0s.fusion.voting instead",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _canonical
