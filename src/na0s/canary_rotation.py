# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.rotation"""
from na0s.canary.rotation import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_rotation is deprecated; use na0s.canary.rotation instead",
    DeprecationWarning,
    stacklevel=2,
)
