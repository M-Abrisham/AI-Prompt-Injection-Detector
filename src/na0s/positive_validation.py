# src/na0s/positive_validation.py (SHIM -- do not add new code here)
"""Backward-compatibility shim.

The canonical location is the ``na0s.validation`` sub-package:
    PositiveValidator, ValidationResult, VALIDATION_TAXONOMY_MAP,
    DEFAULT_VALIDATION_WEIGHTS  -> na0s.validation.positive
    TrustBoundary, validate_output                  -> na0s.validation.trust_boundary

Remove in v1.0.0.
"""
from __future__ import annotations

import warnings

from na0s.validation.positive import (  # noqa: F401
    DEFAULT_VALIDATION_WEIGHTS,
    VALIDATION_TAXONOMY_MAP,
    PositiveValidator,
    ValidationResult,
    _PERSONA_OVERRIDE_PATTERNS,
)
from na0s.validation.trust_boundary import (  # noqa: F401
    TrustBoundary,
    validate_output,
)

warnings.warn(
    "`na0s.positive_validation` is deprecated; import from "
    "`na0s.validation` instead (removes in v1.0.0).",
    DeprecationWarning,
    stacklevel=2,
)
