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


# Severity-to-weight mapping for rule hits in weighted voting.
# Canonical definition: import from here in predict.py and cascade.py.
#
# Option B calibration (2026-04-20 audit):
#   - critical alone (0.45) + any ML nudge (>=0.17) crosses threshold 0.55
#   - high needs one corroborating signal (high=0.30 + any other signal)
#   - Floors (O2.1, E1, P1.3) become unnecessary when critical=0.45
#   - Rationale: floors were workarounds for under-weighted critical tier.
SEVERITY_WEIGHTS = {
    "critical_content": 0.55,
    "critical": 0.45,
    "high": 0.30,
    "medium": 0.15,
    "low": 0.07,
}
