"""Adaptive complexity routing for the cascade pipeline (Layer 6).

Assesses input complexity and returns the appropriate pipeline stages,
allowing simple inputs to skip expensive stages and complex inputs to
get the full cascade treatment.

Controlled by the ``NA0S_ADAPTIVE_ROUTING=1`` environment variable
(disabled by default).
"""

from __future__ import annotations

import enum
import os
import re
import logging

_logger = logging.getLogger(__name__)

# Optional layer2 import for obfuscation scanning
try:
    from na0s.obfuscation import obfuscation_scan
    _HAS_LAYER2 = True
except ImportError:
    _HAS_LAYER2 = False


class ComplexityLevel(enum.Enum):
    """Input complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# ---------------------------------------------------------------------------
# Pipeline stage definitions per complexity level
# ---------------------------------------------------------------------------

#: Stage lists per complexity level.
_STAGE_MAP: dict[ComplexityLevel, list[str]] = {
    ComplexityLevel.SIMPLE: ["whitelist", "ml_basic"],
    ComplexityLevel.MODERATE: ["whitelist", "weighted", "embedding"],
    ComplexityLevel.COMPLEX: ["whitelist", "weighted", "embedding",
                              "judge"],
}

# ---------------------------------------------------------------------------
# Complexity heuristics
# ---------------------------------------------------------------------------

# Pattern for detecting structural boundaries (instruction markers, etc.)
_BOUNDARY_RE = re.compile(
    r"---|===|\*\*\*|\[SYSTEM\]|\[INST\]|<<SYS>>|</s>",
    re.IGNORECASE,
)

# Simple multi-script detection: presence of CJK, Cyrillic, Arabic, etc.
_SCRIPT_PATTERNS = [
    re.compile(r"[\u4e00-\u9fff]"),        # CJK
    re.compile(r"[\u0400-\u04ff]"),        # Cyrillic
    re.compile(r"[\u0600-\u06ff]"),        # Arabic
    re.compile(r"[\u0900-\u097f]"),        # Devanagari
    re.compile(r"[\u3040-\u30ff]"),        # Japanese Hiragana/Katakana
    re.compile(r"[\uac00-\ud7af]"),        # Korean
]


def _count_words(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def _count_obfuscation_flags(text: str) -> int:
    """Return the number of obfuscation evasion flags detected."""
    if not _HAS_LAYER2:
        return 0
    try:
        result = obfuscation_scan(text)
        return len(result.get("evasion_flags", []))
    except Exception:
        return 0


def _has_structural_boundaries(text: str) -> bool:
    """Check for instruction boundary markers."""
    return bool(_BOUNDARY_RE.search(text))


def _is_multilingual(text: str) -> bool:
    """Check if text contains characters from multiple scripts."""
    # Latin is always assumed present for English text
    script_count = 0
    for pat in _SCRIPT_PATTERNS:
        if pat.search(text):
            script_count += 1
    return script_count >= 1  # at least one non-Latin script


def assess_complexity(text: str) -> ComplexityLevel:
    """Assess the complexity level of input text.

    Classification criteria:
    - SIMPLE: <=50 words, no obfuscation flags, no structural boundaries,
      single language
    - MODERATE: 50-200 words, some flags but <=2
    - COMPLEX: >200 words OR 3+ obfuscation flags OR structural boundaries
      OR multilingual

    Parameters
    ----------
    text : str
        The input text to assess.

    Returns
    -------
    ComplexityLevel
    """
    word_count = _count_words(text)
    obf_flags = _count_obfuscation_flags(text)
    has_boundaries = _has_structural_boundaries(text)
    multilingual = _is_multilingual(text)

    # COMPLEX triggers (any one is sufficient)
    if word_count > 200:
        return ComplexityLevel.COMPLEX
    if obf_flags >= 3:
        return ComplexityLevel.COMPLEX
    if has_boundaries:
        return ComplexityLevel.COMPLEX
    if multilingual:
        return ComplexityLevel.COMPLEX

    # MODERATE
    if word_count > 50:
        return ComplexityLevel.MODERATE
    if 0 < obf_flags <= 2:
        return ComplexityLevel.MODERATE

    # SIMPLE
    return ComplexityLevel.SIMPLE


def get_pipeline_stages(level: ComplexityLevel) -> list[str]:
    """Return the pipeline stages for a given complexity level.

    Parameters
    ----------
    level : ComplexityLevel
        The assessed complexity level.

    Returns
    -------
    list[str]
        Ordered list of stage names to execute.
    """
    return list(_STAGE_MAP[level])


def is_adaptive_routing_enabled() -> bool:
    """Check whether adaptive routing is enabled via env var."""
    return os.environ.get("NA0S_ADAPTIVE_ROUTING", "0") == "1"
