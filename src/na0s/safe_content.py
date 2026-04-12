# src/na0s/safe_content.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.safe_content"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.safe_content is deprecated; use na0s.integrity.safe_content instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.safe_content")
_sys.modules[__name__] = _canonical
