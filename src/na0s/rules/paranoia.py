"""Paranoia Level System — controls rule sensitivity filtering.

PL1 (1): Production  — highest confidence, lowest FP risk
PL2 (2): Moderate    — good detection, acceptable FP risk (default)
PL3 (3): High        — aggressive detection, some FP expected
PL4 (4): Audit       — catches everything, high FP rate

Configurable via the RULES_PARANOIA_LEVEL environment variable.
"""

import os

_PARANOIA_LEVEL: int = int(os.environ.get("RULES_PARANOIA_LEVEL", "2"))


def get_paranoia_level() -> int:
    """Return the current module-level paranoia level (1-4)."""
    return _PARANOIA_LEVEL


def set_paranoia_level(level: int) -> None:
    """Set the module-level paranoia level (1-4).

    This is the programmatic equivalent of setting RULES_PARANOIA_LEVEL.
    """
    global _PARANOIA_LEVEL
    if not 1 <= level <= 4:
        raise ValueError("paranoia_level must be 1-4, got {}".format(level))
    _PARANOIA_LEVEL = level
