# src/na0s/sbom.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.sbom"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.sbom is deprecated; use na0s.integrity.sbom instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.sbom")
_sys.modules[__name__] = _canonical
