# src/na0s/rag_attribution.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.attribution"""
from na0s.rag.attribution import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.rag_attribution is deprecated; use na0s.rag.attribution instead", DeprecationWarning, stacklevel=2)
