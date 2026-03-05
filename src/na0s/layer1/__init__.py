"""Layer 1: Regex-based signature rule engine for prompt injection detection.

110 pre-compiled rules covering 30+ technique IDs with:
- 4-level paranoia system (PL1-PL4, env-configurable via RULES_PARANOIA_LEVEL)
- Context-aware suppression (25 suppressible rules)
- Unicode evasion defenses (homoglyph folding, Zalgo stripping, IOC refanging)
- Syllable-splitting de-hyphenation (25 Unicode dash chars, compound whitelist)
- ReDoS-safe patterns (all validated with safe_compile)

Public API:
  rule_score(text)          -> list[str]
  rule_score_detailed(text) -> list[RuleHit]
  extract_iocs(text)        -> IocResult
  refang(text)              -> str
  dehyphenate_suspicious(t) -> SplittingResult
  detect_ascii_art(text)    -> AsciiArtResult
  detect_numeric(text)      -> NumericDecodeResult
"""

from .result import Rule, RuleHit, SEVERITY_WEIGHTS
from .paranoia import get_paranoia_level, set_paranoia_level
from .rules_registry import RULES, ROLE_ASSIGNMENT_PATTERN, PERSONA_OVERRIDE_PATTERNS
from .analyzer import rule_score, rule_score_detailed
from .ioc_extractor import extract_iocs, refang, IocResult

from ..layer2.whitespace_stego import detect_whitespace_stego, StegoResult  # noqa: F401
from ..layer2.morse_code import detect_morse, MorseResult  # noqa: F401
from ..layer2.numeric_decode import (  # noqa: F401
    detect_numeric, detect_binary, detect_octal, detect_decimal,
    NumericDecodeResult,
)

from ..layer2.syllable_splitting import dehyphenate_suspicious, SplittingResult  # noqa: F401
from ..layer2.ascii_art_detector import detect_ascii_art, AsciiArtResult  # noqa: F401
