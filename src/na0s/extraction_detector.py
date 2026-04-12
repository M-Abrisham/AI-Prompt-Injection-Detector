# src/na0s/extraction_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.detectors.extraction"""
import warnings as _warnings
_warnings.warn("na0s.extraction_detector is deprecated; use na0s.detectors.extraction instead", DeprecationWarning, stacklevel=2)

from na0s.detectors import extraction as _mod  # noqa: E402
from na0s.detectors.extraction import *  # noqa: F401,F403,E402

# Re-export private names so that ``from na0s.extraction_detector import _foo``
# keeps working during the deprecation window.
globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
