# src/na0s/rag_attribution.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.output.attribution"""
from na0s.output.attribution import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn("na0s.rag_attribution is deprecated; use na0s.output instead", DeprecationWarning, stacklevel=2)
