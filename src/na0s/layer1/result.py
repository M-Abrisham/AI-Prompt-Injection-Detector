"""Layer 1 data types — Rule, RuleHit dataclasses and severity weights.

Zero internal dependencies (only stdlib + layer0.safe_regex utility).
"""

import re
from dataclasses import dataclass, field

from ..input.safe_regex import safe_compile


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
SEVERITY_WEIGHTS = {
    "critical_content": 0.45,
    "critical": 0.3,
    "high": 0.2,
    "medium": 0.1,
    "low": 0.05,
}
