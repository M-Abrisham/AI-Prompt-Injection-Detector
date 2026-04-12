# src/na0s/rag_poison_detector.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.poison_detector"""
from na0s.rag.poison_detector import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.rag_poison_detector is deprecated; use na0s.rag.poison_detector instead", DeprecationWarning, stacklevel=2)
