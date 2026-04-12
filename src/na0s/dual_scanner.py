# src/na0s/dual_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.dual_scanner"""
from na0s.rag.dual_scanner import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.dual_scanner is deprecated; use na0s.rag.dual_scanner instead", DeprecationWarning, stacklevel=2)
