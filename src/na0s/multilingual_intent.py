# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.detectors.multilingual_intent"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.multilingual_intent is deprecated; use na0s.detectors.multilingual_intent instead",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _importlib.import_module("na0s.detectors.multilingual_intent")
