# SHIM -- do not add new code here
"""Backward-compat package shim. Canonical location: na0s.obfuscation

Aliases the package AND each submodule into sys.modules under the old
``na0s.layer2`` name, so ``from na0s.layer2.obfuscation import X`` returns the
SAME module object as ``na0s.obfuscation.obfuscation`` (mock-patching safe).
"""
import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.layer2 is deprecated; use na0s.obfuscation instead",
    DeprecationWarning,
    stacklevel=2,
)
_canonical = _importlib.import_module("na0s.obfuscation")
_sys.modules[__name__] = _canonical
for _info in _pkgutil.iter_modules(_canonical.__path__):
    try:
        _sub = _importlib.import_module(f"na0s.obfuscation.{_info.name}")
    except Exception:
        continue
    _sys.modules[f"{__name__}.{_info.name}"] = _sub
