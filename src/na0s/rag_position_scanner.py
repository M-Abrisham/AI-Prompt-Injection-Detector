# src/na0s/rag_position_scanner.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.rag.position_scanner"""
from na0s.rag.position_scanner import *  # noqa: F401,F403
from na0s.rag.position_scanner import _default_position_weights, _score_chunk  # noqa: F401
import warnings as _warnings
_warnings.warn("na0s.rag_position_scanner is deprecated; use na0s.rag.position_scanner instead", DeprecationWarning, stacklevel=2)
