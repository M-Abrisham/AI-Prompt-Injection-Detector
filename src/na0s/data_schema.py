# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.dataset.schema"""
from na0s.dataset.schema import *  # noqa: F401,F403
import warnings
warnings.warn(
    "na0s.data_schema is deprecated; import from na0s.dataset.schema",
    DeprecationWarning, stacklevel=2,
)
