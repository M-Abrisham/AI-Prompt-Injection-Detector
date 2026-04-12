# src/na0s/model_encryption.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.model_encryption"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.model_encryption is deprecated; use na0s.integrity.model_encryption instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.model_encryption")
_sys.modules[__name__] = _canonical
