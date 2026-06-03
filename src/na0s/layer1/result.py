"""Layer 1 data types — Rule, RuleHit dataclasses and severity weights.

Zero internal dependencies (only stdlib + layer0.safe_regex utility).
"""

import re
from dataclasses import dataclass, field

from ..layer0.safe_regex import safe_compile


@dataclass
class Rule:
    name: str
    pattern: str
    technique_ids: list = field(default_factory=list)
    severity: str = "medium"
    description: str = ""
    paranoia_level: int = 1
    _compiled: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._compiled = safe_compile(
            self.pattern, re.IGNORECASE, check_safety=True,
        )


@dataclass
class RuleHit:
    name: str
    technique_ids: list = field(default_factory=list)
    severity: str = "medium"
    # Span-aware evidence (Phase: foundation/plumbing only).
    # span = (start, end) codepoint offsets of the regex match, indexing
    # into the SAME string the match ran against (which may be a folded /
    # decoded view, NOT the original text). matched_text is .group(0) from
    # that same match, so span and matched_text are intrinsically aligned
    # and travel together — never re-slice the original text with this span.
    # Both default None for backward compatibility (21 files construct
    # RuleHit positionally / by name without these fields) and for the
    # fail-closed timeout case where no Match object is available.
    span: tuple | None = None
    matched_text: str | None = None


# Severity-to-weight mapping for rule hits in weighted voting.
# Canonical definition: import from here in predict.py and cascade.py.
SEVERITY_WEIGHTS = {
    "critical_content": 0.45,
    "critical": 0.3,
    "high": 0.2,
    "medium": 0.1,
    "low": 0.05,
}
