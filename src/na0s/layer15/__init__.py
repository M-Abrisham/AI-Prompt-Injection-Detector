# SHIM -- do not add new code here
"""Backward-compat package shim. Canonical location: na0s.threat_intel

Aliases the package AND each submodule into sys.modules under the old
``na0s.layer15`` name, so ``from na0s.layer15.atlas_sync import X`` returns the
SAME module object as ``na0s.threat_intel.atlas_sync`` (preserves identity for
mock-patching and module-level state).
"""
import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.layer15 is deprecated; use na0s.threat_intel instead",
    DeprecationWarning,
    stacklevel=2,
)
_canonical = _importlib.import_module("na0s.threat_intel")
_sys.modules[__name__] = _canonical
for _info in _pkgutil.iter_modules(_canonical.__path__):
    try:
        _sub = _importlib.import_module(f"na0s.threat_intel.{_info.name}")
    except Exception:
        continue
    _sys.modules[f"{__name__}.{_info.name}"] = _sub
