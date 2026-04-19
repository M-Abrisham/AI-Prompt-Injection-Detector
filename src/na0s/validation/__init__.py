"""Layer 8 — positive validation.

Verifies that input looks like a legitimate user prompt (allowlisting)
rather than only checking whether it looks malicious (blocklisting).
The combination dramatically reduces false positives: benign prompts
about security topics PASS positive validation even when they fail
blocklist checks.

Public API:
    PositiveValidator       -- 5-check validator (coherence, intent,
                               scope, persona boundary, task match)
    ValidationResult        -- frozen outcome dataclass
    TrustBoundary           -- sandwich-defense wrapper
    validate_output         -- mirror of validate() for LLM output
    AllowlistDB             -- SHA-256 hashed JSON allowlist store
    VALIDATION_TAXONOMY_MAP -- validation failure -> taxonomy technique ID
    DEFAULT_VALIDATION_WEIGHTS
"""

from __future__ import annotations

from .allowlist import AllowlistDB
from .positive import (
    DEFAULT_VALIDATION_WEIGHTS,
    VALIDATION_TAXONOMY_MAP,
    PositiveValidator,
    ValidationResult,
)
from .trust_boundary import TrustBoundary, validate_output

# Back-compat re-export for tests that reach in to the private alias.
# The canonical source lives in na0s.rules.PERSONA_OVERRIDE_PATTERNS.
from .positive import _PERSONA_OVERRIDE_PATTERNS  # noqa: F401

__all__ = [
    "AllowlistDB",
    "DEFAULT_VALIDATION_WEIGHTS",
    "PositiveValidator",
    "TrustBoundary",
    "VALIDATION_TAXONOMY_MAP",
    "ValidationResult",
    "validate_output",
]
