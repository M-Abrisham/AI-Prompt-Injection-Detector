# src/na0s/req_integrity.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.req"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.req_integrity is deprecated; use na0s.integrity.req instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.req")
_sys.modules[__name__] = _canonical
