# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.alert"""
from na0s.canary.alert import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_alert is deprecated; use na0s.canary.alert instead",
    DeprecationWarning,
    stacklevel=2,
)
