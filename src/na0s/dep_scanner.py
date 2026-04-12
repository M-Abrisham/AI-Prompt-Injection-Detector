# src/na0s/dep_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.dep_scanner"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.dep_scanner is deprecated; use na0s.integrity.dep_scanner instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.dep_scanner")
_sys.modules[__name__] = _canonical
