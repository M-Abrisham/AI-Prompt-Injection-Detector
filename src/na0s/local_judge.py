# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.judge.local_judge"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.local_judge is deprecated; use na0s.judge.local_judge instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.judge.local_judge")
_sys.modules[__name__] = _canonical
