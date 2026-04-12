# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.session"""
from na0s.canary.session import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_session is deprecated; use na0s.canary.session instead",
    DeprecationWarning,
    stacklevel=2,
)
