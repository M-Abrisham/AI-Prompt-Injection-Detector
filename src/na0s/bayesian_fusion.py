# src/na0s/bayesian_fusion.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.fusion.bayesian

Replaces itself in sys.modules so that imports and mock-patching both
resolve to the canonical module object.
"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.bayesian_fusion is deprecated; use na0s.fusion.bayesian instead",
    DeprecationWarning,
    stacklevel=2,
)

_sys.modules[__name__] = _importlib.import_module("na0s.fusion.bayesian")
