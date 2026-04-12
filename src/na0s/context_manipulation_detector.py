# src/na0s/context_manipulation_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.context_manipulation"""
import warnings as _warnings
_warnings.warn("na0s.context_manipulation_detector is deprecated; use na0s.detectors.context_manipulation instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import context_manipulation as _mod  # noqa: E402
from na0s.detectors.context_manipulation import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
