# src/na0s/visual_injection_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.visual_injection"""
import warnings as _warnings
_warnings.warn("na0s.visual_injection_detector is deprecated; use na0s.detectors.visual_injection instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import visual_injection as _mod  # noqa: E402
from na0s.detectors.visual_injection import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
