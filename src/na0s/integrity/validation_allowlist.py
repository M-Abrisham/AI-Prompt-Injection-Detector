# src/na0s/integrity/validation_allowlist.py (SHIM -- do not add new code here)
"""Backward-compatibility shim.

Canonical location: ``na0s.validation.allowlist``.  Allowlisting is a
validation concern, not a supply-chain integrity concern, so the module
moved out of ``integrity/`` and into the new ``validation/`` sub-package.

Remove in v1.0.0.
"""
from __future__ import annotations

import warnings

from na0s.validation.allowlist import AllowlistDB  # noqa: F401

warnings.warn(
    "`na0s.integrity.validation_allowlist` is deprecated; import "
    "`AllowlistDB` from `na0s.validation` instead (removes in v1.0.0).",
    DeprecationWarning,
    stacklevel=2,
)
