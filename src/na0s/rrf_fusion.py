# src/na0s/rrf_fusion.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.fusion.rrf

Replaces itself in sys.modules so that imports and mock-patching both
resolve to the canonical module object.
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.rrf_fusion is deprecated; use na0s.fusion.rrf instead",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _importlib.import_module("na0s.fusion.rrf")
