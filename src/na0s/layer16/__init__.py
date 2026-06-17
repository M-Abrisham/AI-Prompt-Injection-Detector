# SHIM -- do not add new code here
"""Backward-compat package shim. Canonical location: na0s.conversation

Aliases the package AND every (possibly nested) submodule into sys.modules
under the old ``na0s.layer16`` name, so e.g.
``from na0s.layer16.detectors.scheming import X`` returns the SAME module
object as ``na0s.conversation.detectors.scheming`` -- preserving object
identity for ``unittest.mock.patch`` on nested old paths.

Recursion (pkgutil.walk_packages) is required because this package has nested
subpackages (detectors/, storage/, testing/, baselines/); a flat
iter_modules would alias only the top level and leave nested old paths as
distinct objects (mock.patch would then patch a stale copy -> false green).

Optional backends (storage.redis_backend, etc.) may raise ImportError at
import time when their extra is not installed. walk_packages imports each
submodule eagerly, so an onerror handler swallows those failures: the shim
still loads and aliases everything importable.
"""
import importlib as _importlib
import pkgutil as _pkgutil
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "na0s.layer16 is deprecated; use na0s.conversation instead",
    DeprecationWarning,
    stacklevel=2,
)

_CANONICAL = "na0s.conversation"
_OLD = __name__  # "na0s.layer16"

_canonical = _importlib.import_module(_CANONICAL)

# Alias the package object itself first so any submodule import that resolves
# its parent via sys.modules[_OLD] gets the canonical package.
_sys.modules[_OLD] = _canonical


def _onerror(_name):
    # Swallow eager-import failures (e.g. optional-backend ImportError such as
    # storage.redis_backend when 'redis' is not installed). walk_packages would
    # otherwise propagate and abort aliasing of the remaining modules.
    pass


# Recursively walk the canonical package and alias every importable submodule
# under the old name, preserving object identity.
for _info in _pkgutil.walk_packages(_canonical.__path__, _CANONICAL + ".", onerror=_onerror):
    _suffix = _info.name[len(_CANONICAL) + 1:]  # e.g. "detectors.scheming"
    try:
        _sub = _importlib.import_module(_info.name)
    except Exception:
        # Optional dependency missing or submodule import failed; skip aliasing
        # it but keep going so the rest of the package stays usable.
        continue
    _sys.modules[f"{_OLD}.{_suffix}"] = _sub
