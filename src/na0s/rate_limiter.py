# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.judge.rate_limiter"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.rate_limiter is deprecated; use na0s.judge.rate_limiter instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.judge.rate_limiter")
_sys.modules[__name__] = _canonical
