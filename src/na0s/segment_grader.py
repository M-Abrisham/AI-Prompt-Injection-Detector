# src/na0s/segment_grader.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.output.segment_grader"""
from na0s.output.segment_grader import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.segment_grader is deprecated; use na0s.output instead", DeprecationWarning, stacklevel=2)
