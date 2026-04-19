# src/na0s/output_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.output.scanner"""
from na0s.output.scanner import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.output_scanner is deprecated; use na0s.output instead", DeprecationWarning, stacklevel=2)
