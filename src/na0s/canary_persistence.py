# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.persistence"""
from na0s.canary.persistence import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_persistence is deprecated; use na0s.canary.persistence instead",
    DeprecationWarning,
    stacklevel=2,
)
