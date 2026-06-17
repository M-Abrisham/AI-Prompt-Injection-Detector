# SHIM -- do not add new code here
"""Backward-compat shim. Canonical location: na0s.layer1

This module preserves all existing import paths:
  from na0s.rules import rule_score, RULES, SEVERITY_WEIGHTS, ...

New code should import directly from na0s.layer1 instead.
"""

# Public API
from .layer1 import (
    Rule,
    RuleHit,
    RULES,
    SEVERITY_WEIGHTS,
    ROLE_ASSIGNMENT_PATTERN,
    PERSONA_OVERRIDE_PATTERNS,
    get_paranoia_level,
    set_paranoia_level,
    rule_score,
    rule_score_detailed,
)

# Private symbols used by tests (backward compat)
from .layer1.context import (
    _CONTEXT_SUPPRESSIBLE,
    _has_contextual_framing,
    _is_legitimate_roleplay,
)
from .layer1.unicode_defense import _fold_angle_homoglyphs
