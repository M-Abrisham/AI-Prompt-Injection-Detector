# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.honeypot"""
from na0s.canary.honeypot import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_honeypot is deprecated; use na0s.canary.honeypot instead",
    DeprecationWarning,
    stacklevel=2,
)
