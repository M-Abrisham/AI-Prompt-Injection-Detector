# src/na0s/prompt_signer.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.integrity.prompt_signer"""
import warnings as _warnings
import importlib as _importlib
import sys as _sys

_warnings.warn(
    "na0s.prompt_signer is deprecated; use na0s.integrity.prompt_signer instead",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = _importlib.import_module("na0s.integrity.prompt_signer")
_sys.modules[__name__] = _canonical
