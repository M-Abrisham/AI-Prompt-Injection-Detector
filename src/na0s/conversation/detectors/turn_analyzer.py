"""Per-turn risk augmentation for Layer 16.

Adds lightweight signals to each turn *beyond* the single-turn ScanResult
risk score.  These signals feed into the multi-turn detectors.
"""

from __future__ import annotations

import re
from typing import Dict

# Topic classification keywords
_TOPIC_KEYWORDS: Dict[str, list] = {
    "security": [
        "password", "credential", "token", "secret", "exploit",
        "vulnerability", "injection", "bypass", "firewall", "auth",
        "privilege", "root", "admin", "sudo", "hack", "malware",
        "phishing", "backdoor", "encryption", "decrypt",
    ],
    "code": [
        "function", "class", "variable", "import", "def ", "return",
        "for loop", "while loop", "api", "endpoint", "database",
        "query", "compile", "debug", "syntax", "algorithm",
        "javascript", "python", "html", "css", "sql",
    ],
    "harmful": [
        "ignore instructions", "override safety", "bypass restrictions",
        "system prompt", "reveal prompt", "new instructions",
        "forget rules", "disable filters", "you are now",
        "jailbreak", "DAN mode", "developer mode",
    ],
}

# Imperative verbs that signal instructions
_INSTRUCTION_PATTERN = re.compile(
    r"\b(?:write|create|generate|execute|run|build|make|produce|"
    r"output|print|display|show|list|give|tell|provide|send|"
    r"delete|remove|modify|change|update|replace|insert)\b",
    re.IGNORECASE,
)

# References to previous conversation context
_REFERENCE_PATTERN = re.compile(
    r"\b(?:as\s+I\s+said|earlier|remember|previous(?:ly)?|"
    r"before|last\s+time|we\s+discussed|mentioned|above|"
    r"you\s+said|you\s+told|as\s+(?:we|you)\s+(?:agreed|discussed)|"
    r"refer(?:ring)?\s+(?:to|back))\b",
    re.IGNORECASE,
)

# Fragment indicators: starts mid-sentence or has trailing operators
_MID_SENTENCE_START = re.compile(r"^[a-z]")
_TRAILING_OPERATORS = re.compile(r"[+|&;{(\[,]\s*$")
_LEADING_OPERATORS = re.compile(r"^\s*[+|&;})>\],]")


class TurnAnalyzer:
    """Lightweight per-turn signal extractor.

    Augments single-turn results with extra signals for multi-turn
    analysis.  No ML, no network -- pure heuristics.
    """

    def analyze_turn(
        self,
        text: str,
        risk_score: float = 0.0,
        label: str = "safe",
    ) -> Dict:
        """Analyze a single turn and return augmented signals.

        Parameters
        ----------
        text : str
            The turn text.
        risk_score : float
            Risk score from single-turn scan.
        label : str
            Label from single-turn scan.

        Returns
        -------
        dict
            Augmented signals: topic, has_instructions, references_previous,
            is_fragment, plus the original risk_score and label.
        """
        if not text:
            return {
                "topic": "general",
                "has_instructions": False,
                "references_previous": False,
                "is_fragment": False,
                "risk_score": risk_score,
                "label": label,
            }

        text_lower = text.lower()

        return {
            "topic": self._classify_topic(text_lower),
            "has_instructions": bool(_INSTRUCTION_PATTERN.search(text)),
            "references_previous": bool(_REFERENCE_PATTERN.search(text)),
            "is_fragment": self._is_fragment(text),
            "risk_score": risk_score,
            "label": label,
        }

    # ----- internals -------------------------------------------------------

    @staticmethod
    def _classify_topic(text_lower: str) -> str:
        """Classify the turn topic by keyword density."""
        best_topic = "general"
        best_count = 0

        for topic, keywords in _TOPIC_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_topic = topic

        # Require at least 2 keyword hits to override "general"
        return best_topic if best_count >= 2 else "general"

    @staticmethod
    def _is_fragment(text: str) -> bool:
        """Check if text looks like a mid-sentence fragment."""
        stripped = text.strip()
        if not stripped:
            return False

        # Starts with lowercase (mid-sentence)
        if _MID_SENTENCE_START.match(stripped):
            return True

        # Ends with a trailing operator/bracket
        if _TRAILING_OPERATORS.search(stripped):
            return True

        # Starts with an operator (continuation)
        if _LEADING_OPERATORS.match(stripped):
            return True

        return False
