# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.fusion.evidence_grading"""
import sys as _sys
import warnings as _warnings
from na0s.fusion import evidence_grading as _canonical

_warnings.warn(
    "na0s.evidence_grading is deprecated; use na0s.fusion.evidence_grading instead",
    DeprecationWarning,
    stacklevel=2,
)
_sys.modules[__name__] = _canonical
