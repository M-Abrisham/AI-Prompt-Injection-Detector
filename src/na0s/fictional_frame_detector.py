# src/na0s/fictional_frame_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.fictional_frame"""
import warnings as _warnings
_warnings.warn("na0s.fictional_frame_detector is deprecated; use na0s.detectors.fictional_frame instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import fictional_frame as _mod  # noqa: E402
from na0s.detectors.fictional_frame import *  # noqa: F401,F403,E402

globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
