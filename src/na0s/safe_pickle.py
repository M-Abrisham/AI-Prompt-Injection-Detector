# src/na0s/safe_pickle.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.safe_pickle"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.safe_pickle is deprecated; use na0s.integrity.safe_pickle instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.safe_pickle")
_sys.modules[__name__] = _canonical
