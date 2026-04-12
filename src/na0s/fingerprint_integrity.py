# src/na0s/fingerprint_integrity.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.fingerprint"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.fingerprint_integrity is deprecated; use na0s.integrity.fingerprint instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.fingerprint")
_sys.modules[__name__] = _canonical
