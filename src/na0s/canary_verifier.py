# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.canary.verifier"""
from na0s.canary.verifier import *  # noqa: F401,F403
import warnings as _warnings
_warnings.warn(
    "na0s.canary_verifier is deprecated; use na0s.canary.verifier instead",
    DeprecationWarning,
    stacklevel=2,
)
