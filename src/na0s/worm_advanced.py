# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.worm.advanced"""
from na0s.worm.advanced import *  # noqa: F401,F403
from na0s.worm.advanced import _hash_desc, _levenshtein  # noqa: F401 — private names used downstream
import warnings as _warnings
_warnings.warn("na0s.worm_advanced is deprecated; use na0s.worm.advanced instead", DeprecationWarning, stacklevel=2)
