# src/na0s/complexity_router.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.fusion.complexity_router

Replaces itself in sys.modules so that imports and mock-patching both
resolve to the canonical module object.
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.complexity_router is deprecated; use na0s.fusion.complexity_router instead",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _importlib.import_module("na0s.fusion.complexity_router")
