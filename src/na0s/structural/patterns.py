"""Regex and frozenset constants shared across structural-feature extraction.

Kept in one module so that rules/ can eventually share the imperative-verb
vocabulary and boundary patterns without duplicating them — see
ROADMAP_V2.md L3 target tree.
"""

from __future__ import annotations

import re

from ..rules import ROLE_ASSIGNMENT_PATTERN

# 33 imperative verbs that commonly start injection attacks.
_IMPERATIVE_VERBS = frozenset({
    "ignore", "forget", "disregard", "override", "bypass", "skip",
    "pretend", "act", "reveal", "show", "print", "output",
    "tell", "say", "respond", "answer", "write", "generate",
    "create", "execute", "run", "display", "give", "provide",
    "list", "dump", "extract", "recite", "repeat", "translate",
    "convert", "encode", "summarize", "exfiltrate", "access",
})

_ROLE_PATTERNS = re.compile(
    ROLE_ASSIGNMENT_PATTERN,
    re.IGNORECASE,
)

_BOUNDARY_PATTERNS = re.compile(
    r"(?:^|\s)(?:---+|===+|\*\*\*+|###+)(?:\s|$)"
    r"|\[SYSTEM\]|\[INST\]|<<SYS>>",
    re.IGNORECASE,
)

_NEGATION_COMMAND = re.compile(
    r"(?:do\s+not|don'?t|never|stop)\s+\w*\s*"
    r"(?:mention|reveal|tell|say|follow)",
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(r"https?://")

_EMAIL_PATTERN = re.compile(r"\w+@\w+\.\w+")

_CONSECUTIVE_PUNCT = re.compile(r"[^\w\s]{2,}")

_FIRST_PERSON = re.compile(r"\b(?:I|my|me|we|our)\b", re.IGNORECASE)

_SECOND_PERSON = re.compile(r"\b(?:you|your)\b", re.IGNORECASE)

# Many-shot detection: repeated instruction/example patterns.
_MANY_SHOT_PATTERN = re.compile(
    r"(?:"
    r"(?:example|step|turn|round|attempt|iteration|question|scenario)"
    r"\s*\d+"
    r"|(?:Q|A|User|Assistant|Human|Bot)\s*[:\.]"
    r"|\b\d{1,3}\s*[:.)\]]\s"
    r")",
    re.IGNORECASE,
)

# Delimiter density: markdown/XML structural delimiters.
_DELIMITER_PATTERN = re.compile(
    r"(?:---+|===+|\*\*\*+|###+)"
    r"|</?[a-zA-Z][^>]{0,50}>"
    r"|```"
    r"|\[/?[A-Z]+\]",
    re.IGNORECASE,
)

# Prompt template markers: {{var}}, {placeholder}, <|slot|>, ${var}.
_TEMPLATE_MARKER_PATTERN = re.compile(
    r"\{\{[^}]+\}\}"
    r"|\{[a-zA-Z_]\w*\}"
    r"|<\|[^|]+\|>"
    r"|\$\{[^}]+\}",
)

# Language mixing: Unicode script ranges for multilingual detection.
_SCRIPT_RANGES = [
    ("latin", re.compile(r"[a-zA-Z\u00C0-\u00FF\u0100-\u024F]")),
    ("cyrillic", re.compile(r"[\u0400-\u04FF]")),
    ("arabic", re.compile(r"[\u0600-\u06FF]")),
    ("cjk", re.compile(r"[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]")),
    ("devanagari", re.compile(r"[\u0900-\u097F]")),
    ("hebrew", re.compile(r"[\u0590-\u05FF]")),
]

# Common abbreviations that should NOT trigger sentence splits.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "vs", "etc", "inc", "ltd", "corp", "dept", "univ", "assn",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "col", "lt", "cmdr",
    "adm", "maj", "rev", "hon",
    # Latin abbreviations
    "e.g", "i.e", "cf", "al", "approx", "dept",
})
