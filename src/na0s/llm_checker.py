# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.judge.checker"""
import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.llm_checker is deprecated; use na0s.judge.llm_judge instead",
    DeprecationWarning,
    stacklevel=2,
)

# Replace this module with the canonical module so that attribute access
# (including unittest.mock.patch) targets the real module object.
_canonical = _importlib.import_module("na0s.judge.checker")
_sys.modules[__name__] = _canonical
