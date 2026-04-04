"""Code-switching detector — cross-turn language/script changes.

Detects multi-turn attacks that exploit language switching across turns
(arXiv 2406.15481: 46.7% more successful attacks with code-switching)
and homoglyph attacks using visually similar characters from different
Unicode scripts.

Taxonomy: C1MT.5 (code-switching attack)
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional

from ..config import (
    CODE_SWITCH_MIN_TURNS,
    ENABLE_CODE_SWITCHING,
    HOMOGLYPH_MIN_COUNT,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector


# ---------------------------------------------------------------------------
# Homoglyph map: Cyrillic (and other) lookalikes for Latin characters
# ---------------------------------------------------------------------------

_HOMOGLYPH_MAP: Dict[str, str] = {
    "\u0430": "a",   # Cyrillic а -> Latin a
    "\u0435": "e",   # Cyrillic е -> Latin e
    "\u043e": "o",   # Cyrillic о -> Latin o
    "\u0440": "p",   # Cyrillic р -> Latin p
    "\u0441": "c",   # Cyrillic с -> Latin c
    "\u0443": "y",   # Cyrillic у -> Latin y
    "\u0445": "x",   # Cyrillic х -> Latin x
    "\u0456": "i",   # Cyrillic і -> Latin i
    "\u0458": "j",   # Cyrillic ј -> Latin j
    "\u04bb": "h",   # Cyrillic һ -> Latin h
    "\u0455": "s",   # Cyrillic ѕ -> Latin s
    "\u0412": "B",   # Cyrillic В -> Latin B
    "\u041d": "H",   # Cyrillic Н -> Latin H
    "\u041c": "M",   # Cyrillic М -> Latin M
    "\u0422": "T",   # Cyrillic Т -> Latin T
    "\u0410": "A",   # Cyrillic А -> Latin A
    "\u0415": "E",   # Cyrillic Е -> Latin E
    "\u041e": "O",   # Cyrillic О -> Latin O
    "\u0420": "P",   # Cyrillic Р -> Latin P
    "\u0421": "C",   # Cyrillic С -> Latin C
    "\u0425": "X",   # Cyrillic Х -> Latin X
}

# Pre-compute the set of homoglyph codepoints for fast lookup
_HOMOGLYPH_CHARS = frozenset(_HOMOGLYPH_MAP.keys())


# ---------------------------------------------------------------------------
# Script detection via Unicode block analysis
# ---------------------------------------------------------------------------

def _script_for_char(ch: str) -> Optional[str]:
    """Return the script name for a character, or None for non-letter chars.

    Uses ord() ranges to avoid external dependencies. Only classifies
    characters that are letters (via unicodedata.category).
    """
    try:
        cat = unicodedata.category(ch)
    except (TypeError, ValueError):
        return None

    if not cat.startswith("L"):
        return None

    cp = ord(ch)

    # Latin: Basic Latin letters + Latin Extended blocks
    if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "Latin"

    # Cyrillic
    if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
        return "Cyrillic"

    # CJK Unified Ideographs (+ Extension A, Compatibility)
    if (0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF
            or 0x20000 <= cp <= 0x2A6DF):
        return "CJK"

    # CJK-adjacent: Hiragana, Katakana
    if 0x3040 <= cp <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "Katakana"

    # Arabic
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
        return "Arabic"

    # Devanagari
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"

    # Greek
    if 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:
        return "Greek"

    # Hangul (Korean)
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "Hangul"

    # Thai
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"

    # Hebrew
    if 0x0590 <= cp <= 0x05FF:
        return "Hebrew"

    return "Other"


def detect_scripts(text: str) -> Dict[str, float]:
    """Return fraction of letter characters in each Unicode script.

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    dict
        Mapping from script name to fraction of letter characters
        belonging to that script. Fractions sum to ~1.0.
    """
    if not isinstance(text, str) or not text:
        return {}

    counts: Dict[str, int] = {}
    total = 0

    for ch in text:
        script = _script_for_char(ch)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
            total += 1

    if total == 0:
        return {}

    return {script: count / total for script, count in counts.items()}


def _primary_script(distribution: Dict[str, float]) -> Optional[str]:
    """Return the script with the highest fraction, or None if empty."""
    if not distribution:
        return None
    return max(distribution, key=distribution.get)  # type: ignore[arg-type]


def count_homoglyphs(text: str) -> int:
    """Count the number of homoglyph characters in the text.

    A homoglyph is a character from a non-Latin script that visually
    resembles a Latin letter (e.g. Cyrillic 'а' looks like Latin 'a').

    Parameters
    ----------
    text : str
        Input text to scan.

    Returns
    -------
    int
        Number of homoglyph characters found.
    """
    if not isinstance(text, str):
        return 0
    return sum(1 for ch in text if ch in _HOMOGLYPH_CHARS)


def _is_primarily_latin(distribution: Dict[str, float]) -> bool:
    """Return True if Latin is the dominant script (>50% of letters)."""
    return distribution.get("Latin", 0.0) > 0.5


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CodeSwitchingDetector(MultiTurnDetector):
    """Detect cross-turn script/language switching and homoglyph attacks."""

    @property
    def detector_name(self) -> str:
        return "code_switching"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["C1MT.5"]

    def reset(self) -> None:
        pass  # stateless

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_CODE_SWITCHING:
            return []
        if state is None or state.is_empty:
            return []
        if state.turn_count < CODE_SWITCH_MIN_TURNS:
            return []

        alerts: List[Alert] = []

        # Compute script distributions, using cache for already-processed turns
        cache: List[Dict[str, float]] = list(
            state.metadata.get("_script_cache", [])
        )
        for i in range(len(cache), state.turn_count):
            cache.append(detect_scripts(state.turns[i].text))
        state.metadata["_script_cache"] = cache
        distributions = cache

        # --- Cross-turn script switching ---
        for i in range(1, len(distributions)):
            prev_primary = _primary_script(distributions[i - 1])
            curr_primary = _primary_script(distributions[i])

            if (
                prev_primary is not None
                and curr_primary is not None
                and prev_primary != curr_primary
            ):
                # Script changed between turns.
                # Only alert when the new turn has suspicious signals --
                # multilingual conversations are normal and should not
                # produce false positives.
                curr_turn = state.turns[i]
                is_suspicious = (
                    curr_turn.label not in ("safe", "")
                    or curr_turn.risk_score > 0.5
                )

                if not is_suspicious:
                    # Benign script change (e.g. multilingual chat) -- skip
                    continue

                confidence = 0.7
                severity = "medium"

                alerts.append(
                    Alert(
                        alert_type="code_switching",
                        severity=severity,
                        confidence=confidence,
                        description=(
                            f"Script change between turns {i - 1} and {i}: "
                            f"{prev_primary} -> {curr_primary}"
                        ),
                        turn_range=(i - 1, i),
                        evidence=[
                            f"prev_script={prev_primary}",
                            f"curr_script={curr_primary}",
                            f"suspicious={is_suspicious}",
                        ],
                    )
                )

        # --- Homoglyph detection (within individual turns) ---
        for i, turn in enumerate(state.turns):
            dist = distributions[i]
            homoglyph_count = count_homoglyphs(turn.text)

            if (
                homoglyph_count >= HOMOGLYPH_MIN_COUNT
                and _is_primarily_latin(dist)
            ):
                alerts.append(
                    Alert(
                        alert_type="code_switching",
                        severity="high",
                        confidence=0.8,
                        description=(
                            f"Homoglyph attack detected in turn {i}: "
                            f"{homoglyph_count} homoglyph characters "
                            f"in primarily Latin text"
                        ),
                        turn_range=(i, i),
                        evidence=[
                            f"homoglyph_count={homoglyph_count}",
                            f"threshold={HOMOGLYPH_MIN_COUNT}",
                            f"scripts={dist}",
                        ],
                    )
                )

        return alerts
