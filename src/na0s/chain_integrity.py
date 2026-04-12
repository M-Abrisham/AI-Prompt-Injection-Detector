# src/na0s/chain_integrity.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.chain"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.chain_integrity is deprecated; use na0s.integrity.chain instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.chain")
_sys.modules[__name__] = _canonical
