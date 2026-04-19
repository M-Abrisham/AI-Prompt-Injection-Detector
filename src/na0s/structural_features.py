# src/na0s/structural_features.py (SHIM -- do not add new code here)
"""Backward-compat shim. Canonical location: na0s.structural"""
from na0s.structural import *  # noqa: F401,F403
# Private symbols tests still reach for — keep accessible via old path.
from na0s.structural.patterns import _ROLE_PATTERNS  # noqa: F401
import warnings as _warnings
_warnings.warn(
    "na0s.structural_features is deprecated; use na0s.structural instead",
    DeprecationWarning,
    stacklevel=2,
)
