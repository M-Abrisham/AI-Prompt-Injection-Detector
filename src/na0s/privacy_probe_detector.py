# src/na0s/privacy_probe_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rules.registry.privacy_probe"""
import warnings as _warnings
_warnings.warn("na0s.privacy_probe_detector is deprecated; use na0s.rules.registry.privacy_probe instead", DeprecationWarning, stacklevel=2)

from na0s.rules.registry import privacy_probe as _mod  # noqa: E402
from na0s.rules.registry.privacy_probe import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
