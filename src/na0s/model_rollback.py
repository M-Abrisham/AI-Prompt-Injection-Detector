# src/na0s/model_rollback.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.model_rollback"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.model_rollback is deprecated; use na0s.integrity.model_rollback instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.model_rollback")
_sys.modules[__name__] = _canonical
