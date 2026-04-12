# src/na0s/ensemble.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.fusion.ensemble

Replaces itself in sys.modules so that imports and mock-patching both
resolve to the canonical module object.
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.ensemble is deprecated; use na0s.fusion.ensemble instead",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _importlib.import_module("na0s.fusion.ensemble")
