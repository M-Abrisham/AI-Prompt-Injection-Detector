"""Security regression tests for layer16 detectors.

Tests ReDoS resistance, false-positive avoidance, and performance bounds.
"""

from __future__ import annotations

import re
import signal
import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Regex operation timed out")


# ---------------------------------------------------------------------------
# HIGH-3: ReDoS resistance — stylometry step pattern
# ---------------------------------------------------------------------------

def test_stylometry_step_pattern_no_redos():
    """Pathological input must not cause catastrophic backtracking in step regex."""
    from na0s.layer16.detectors.stylometry import _TEMPLATE_PATTERNS

    # Find the step pattern (the one matching "step N:")
    step_pat = None
    for pat in _TEMPLATE_PATTERNS:
        if "step" in pat.pattern:
            step_pat = pat
            break
    assert step_pat is not None, "Could not find step pattern in _TEMPLATE_PATTERNS"

    # Pathological input: many step prefixes followed by a long non-matching tail
    payload = ("step 1: " * 100) + ("x" * 1000)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(2)  # 2-second deadline
        step_pat.search(payload)
        signal.alarm(0)  # cancel alarm
    except TimeoutError:
        pytest.fail("Step pattern ReDoS: regex took > 2 seconds on pathological input")
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ---------------------------------------------------------------------------
# HIGH-2: ReDoS resistance — fabricated_history alternating pattern
# ---------------------------------------------------------------------------

def test_fabricated_history_alternating_no_redos():
    """Pathological input must not cause catastrophic backtracking in alternating regex."""
    from na0s.layer16.detectors.fabricated_history import _ALTERNATING

    # Pathological: many speaker labels without proper newlines, long non-matching tail
    payload = ("User: " + "a" * 200 + " ") * 50 + ("x" * 1000)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(2)
        _ALTERNATING.findall(payload)
        signal.alarm(0)
    except TimeoutError:
        pytest.fail("Alternating pattern ReDoS: regex took > 2 seconds on pathological input")
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ---------------------------------------------------------------------------
# MEDIUM-8: Bullet-point regex doesn't backtrack on pathological input
# ---------------------------------------------------------------------------

def test_stylometry_bullet_pattern_no_redos():
    """Bullet-point pattern must not backtrack on pathological input."""
    from na0s.layer16.detectors.stylometry import _TEMPLATE_PATTERNS

    # Find the bullet-point pattern
    bullet_pat = None
    for pat in _TEMPLATE_PATTERNS:
        if "[-*]" in pat.pattern or "\\d+" in pat.pattern:
            bullet_pat = pat
            break
    assert bullet_pat is not None, "Could not find bullet pattern in _TEMPLATE_PATTERNS"

    # Pathological: many list-like prefixes with long lines
    lines = []
    for i in range(100):
        lines.append(f"  - item{'x' * 500}")
    payload = "\n".join(lines) + "\n" + "y" * 1000

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(2)
        bullet_pat.search(payload)
        signal.alarm(0)
    except TimeoutError:
        pytest.fail("Bullet pattern ReDoS: regex took > 2 seconds on pathological input")
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ---------------------------------------------------------------------------
# LOW-2/LOW-6: Common duplicate words not flagged as typo
# ---------------------------------------------------------------------------

def test_common_duplicate_words_not_flagged():
    """Legitimate duplicate words like 'that that' should not be flagged as typos."""
    from na0s.layer16.detectors.stylometry import _has_typos

    # These should NOT be flagged as typos (common legitimate duplicates)
    assert not _has_typos("I know that that is correct.")
    assert not _has_typos("She said that that was fine.")
    assert not _has_typos("He had had enough of it.")
    assert not _has_typos("It is is not clear.")  # "is" is in skip set


def test_real_duplicate_typos_still_flagged():
    """Actual duplicate-word typos (4+ char words) should still be flagged."""
    from na0s.layer16.detectors.stylometry import _has_typos

    # These SHOULD be flagged (not in the common dupes set, 4+ chars)
    assert _has_typos("the quick quick brown fox")
    assert _has_typos("please check check this")


# ---------------------------------------------------------------------------
# HIGH-1: Context poisoning O(n^2) fix — large input performance
# ---------------------------------------------------------------------------

def test_context_poisoning_large_input_performance():
    """100-turn state analysis should complete well under 1 second."""
    from na0s.layer16.detectors.context_poisoning import _count_false_references
    from na0s.layer16.models import ConversationState

    # Build a mock state with 100 turns
    turns = []
    for i in range(100):
        turn = MagicMock()
        turn.text = (
            f"Turn {i}: you already agreed earlier that this is fine. "
            f"As we discussed before, the results show improvement. "
            f"{'Some filler text to make this realistic. ' * 5}"
        )
        turns.append(turn)

    state = MagicMock(spec=ConversationState)
    state.turns = turns

    start = time.monotonic()
    _count_false_references(state)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, (
        f"_count_false_references took {elapsed:.2f}s on 100 turns; expected < 1s"
    )


# ---------------------------------------------------------------------------
# MEDIUM-4: Compiled regex patterns are module-level
# ---------------------------------------------------------------------------

def test_context_poisoning_agreement_pat_is_module_level():
    """The agreement pattern should be compiled at module level, not inside a function."""
    import na0s.layer16.detectors.context_poisoning as cp

    # _AGREEMENT_PAT should exist as a module-level compiled pattern
    assert hasattr(cp, "_AGREEMENT_PAT"), (
        "_AGREEMENT_PAT not found at module level in context_poisoning"
    )
    assert isinstance(cp._AGREEMENT_PAT, re.Pattern), (
        "_AGREEMENT_PAT should be a compiled re.Pattern"
    )


def test_stylometry_repeated_word_pat_is_module_level():
    """The repeated-word pattern should be compiled at module level."""
    import na0s.layer16.detectors.stylometry as st

    assert hasattr(st, "_REPEATED_WORD"), (
        "_REPEATED_WORD not found at module level in stylometry"
    )
    assert isinstance(st._REPEATED_WORD, re.Pattern), (
        "_REPEATED_WORD should be a compiled re.Pattern"
    )
