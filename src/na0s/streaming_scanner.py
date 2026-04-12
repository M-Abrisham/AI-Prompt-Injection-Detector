# src/na0s/streaming_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.streaming"""
from na0s.rag.streaming import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.streaming_scanner is deprecated; use na0s.rag.streaming instead", DeprecationWarning, stacklevel=2)
