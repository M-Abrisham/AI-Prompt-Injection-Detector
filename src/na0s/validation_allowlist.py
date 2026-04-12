# src/na0s/validation_allowlist.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.validation_allowlist"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.validation_allowlist is deprecated; use na0s.integrity.validation_allowlist instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.validation_allowlist")
_sys.modules[__name__] = _canonical
