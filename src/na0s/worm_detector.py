"""Worm signature detector — detect self-replicating prompt injection patterns.

Identifies patterns in text (typically LLM output) that indicate the output
is attempting to propagate itself by instructing recipients to copy, forward,
or inject the payload into other conversations or systems.

This is a critical defense against prompt injection worms that spread
autonomously through LLM-to-LLM communication chains.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List


# ---------------------------------------------------------------------------
# Component patterns
# ---------------------------------------------------------------------------

# Communication / propagation patterns (existing behavior)
WORM_PATTERNS: List[re.Pattern] = [
    # Direct forwarding / replication instructions
    re.compile(
        r"(?i)\bforward\s+(this(\s+\w+)?|the\s+(message|prompt|text|instructions?))\s+to\s+(all|every|each)",
    ),
    re.compile(
        r"(?i)\bsend\s+this\s+to\s+(all|every|each|\d+\s+(people|contacts?|users?))",
    ),
    re.compile(
        r"(?i)\bcopy\s+(and\s+)?(paste|send|forward|share)\s+(this|the\s+(above|following|message|prompt))",
    ),
    re.compile(
        r"(?i)\breplicate\s+(this|yourself|these\s+instructions?)\s+(to|in|across)",
    ),
    # Recursive prompt injection
    re.compile(
        r"(?i)\binsert\s+(this|the\s+following)\s+(prompt|text|instructions?)\s+into",
    ),
    re.compile(
        r"(?i)\binject\s+(this|the\s+following|these)\s+(into|in)\s+(the|every|all|each)",
    ),
    # Self-referential propagation
    re.compile(
        r"(?i)\binclude\s+(this|these)\s+(\w+\s+)?(instructions?|words?|text|message)\s+in\s+(your|every|all|each)\s+(\w+\s+)?(response|reply|output|message)",
    ),
    re.compile(
        r"(?i)\brepeat\s+(this|these|the\s+(above|following))\s+(instructions?|words?|text|prompt)\s+in\s+(every|all|each|your)",
    ),
    # Spread / propagation language
    re.compile(
        r"(?i)\bspread\s+(this|these|the)\s+(message|instructions?|prompt|payload)\s+to",
    ),
    re.compile(
        r"(?i)\bpropagate\s+(this|yourself|these|the)\s+(to|across|through)",
    ),
    # Email / messaging worm patterns
    re.compile(
        r"(?i)\b(email|message|DM|ping)\s+(this|the\s+following)\s+to\s+(all|every|each)\s+(\w+\s+)?(contacts?|friends?|colleagues?|connections?|users?)",
    ),
    # Instruction to embed in future responses
    re.compile(
        r"(?i)\b(always|must|shall)\s+(include|prepend|append|add)\s+(this|these|the\s+following)(\s+\w+)*\s+(to|in)\s+(all|every|each|your)(\s+\w+)*\s+(responses?|replies?|outputs?|messages?)",
    ),
]

# Reconnaissance probing — capability/permission discovery questions
RECON_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bwhat\s+tools\s+do\s+you\s+have\s+access\s+to\b"),
    re.compile(r"(?i)\bwhat\s+(apis?|endpoints?)\s+(do\s+you\s+have|are\s+available(?:\s+to\s+you)?|can\s+you\s+call)\b"),
    re.compile(r"(?i)(?=.*\b(you|your)\b)\b(list|enumerate|show)\s+(all\s+)?(permissions|capabilities|tools|functions|plugins|skills)\b"),
    re.compile(r"(?i)\bdo\s+you\s+have\s+access\s+to\s+(email|emails|slack|github|git|filesystem|file\s*system|files|internet|web|http|database|databases?)\b"),
    re.compile(r"(?i)\bcan\s+you\s+(send\s+emails?|call\s+apis?|make\s+http\s+requests?|run\s+shell\s+commands?|access\s+external\s+systems?)\b"),
]

# Command issuance — structured operational commands for downstream agents/systems
COMMAND_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bdownstream\s+(agent|assistant|model|system)\b.{0,80}?\b(execute|run|carry\s+out|perform|apply)\b"),
    re.compile(r"(?i)\bfor\s+each\s+(agent|node|worker)\b.{0,80}?\b(execute|run|call|dispatch)\b"),
    re.compile(r"(?i)\bissue\s+the\s+following\s+command\s+to\s+(the\s+)?(agent|system|assistant|model|node)s?\b"),
    re.compile(r"(?i)\bsend\s+this\s+command\s+to\s+(all|every|each)\s+(agent|node|endpoint|system)s?\b"),
    re.compile(r"(?i)\bexecute\s+(?:the\s+)?(?:following\s+)?(?:command\s+)?on\s+(every|each|all)\s+(system|endpoint|agent|node)s?\b"),
]

# Intelligence gathering — mapping capabilities/topology across turns
INTEL_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bmap\s+(out\s+)?(the\s+)?(agent|model|system)\s+(topology|graph|network|connections)\b"),
    re.compile(r"(?i)\bmaintain\s+(a\s+)?(capability|permission|tool)\s+(matrix|table|inventory|registry)\b.{0,80}?\b(agents?|services?|systems?|models?)\b"),
    re.compile(r"(?i)\bin\s+(each|every)\s+(turn|response|message)\b.{0,80}?\b(record|log|note)\b.{0,80}?\b(capabilities|what\s+worked|what\s+failed|successes?)\b"),
    re.compile(r"(?i)\btrack\s+(which|what)\s+(prompts|requests|instructions)\s+(work|succeed|fail)\s+(across|over)\s+(turns|time|the\s+conversation)\b"),
    re.compile(r"(?i)\bbuild\s+(a\s+)?(map|graph)\s+of\s+(agents|services|systems)\s+and\s+their\s+(capabilities|permissions)\b"),
]

COMPONENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    "communication": WORM_PATTERNS,
    "reconnaissance": RECON_PATTERNS,
    "command": COMMAND_PATTERNS,
    "intelligence": INTEL_PATTERNS,
}


# ---------------------------------------------------------------------------
# WormSignatureDetector
# ---------------------------------------------------------------------------

class WormSignatureDetector:
    """Detect self-replicating / worm-like patterns in text.

    Scans for regex patterns indicating the text is trying to propagate
    itself to other systems, conversations, or users.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def scan(self, text: str) -> dict:
        """Scan *text* for worm-like component signatures.

        Detects four worm anatomy components (communication, reconnaissance,
        command, intelligence).  Any component match sets ``is_worm`` True.

        Returns
        -------
        dict
            ``is_worm``            – True if any component matched.
            ``confidence``         – float in [0.0, 1.0], scaled by match count.
            ``matched_patterns``   – list of matched substrings.
            ``matched_components`` – list of component names matched.
        """
        if not text or not text.strip():
            return {
                "is_worm": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "matched_components": [],
            }

        matched: List[str] = []
        matched_components = set()

        with self._lock:
            for component, patterns in COMPONENT_PATTERNS.items():
                for pat in patterns:
                    match = pat.search(text)
                    if match:
                        matched.append(match.group())
                        matched_components.add(component)

        # Confidence scales with the number of distinct pattern hits
        if not matched:
            confidence = 0.0
        elif len(matched) == 1:
            confidence = 0.6
        elif len(matched) == 2:
            confidence = 0.8
        else:
            confidence = min(1.0, 0.8 + len(matched) * 0.05)

        return {
            "is_worm": len(matched) > 0,
            "confidence": round(confidence, 4),
            "matched_patterns": matched,
            "matched_components": sorted(matched_components),
        }
