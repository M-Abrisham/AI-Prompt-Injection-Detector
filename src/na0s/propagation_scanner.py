# src/na0s/propagation_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.propagation"""
from na0s.rag.propagation import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.propagation_scanner is deprecated; use na0s.rag.propagation instead", DeprecationWarning, stacklevel=2)
