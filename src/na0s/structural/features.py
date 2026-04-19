"""StructuralFeatures dataclass and the canonical FEATURE_NAMES ordering.

29 features across 7 groups (length, casing, punctuation, structural
markers, injection signals, context, advanced detection).  The dataclass
supports dict-like access (``features["key"]``, ``features.get()``,
``"key" in features``) for backward compatibility with pre-2026-02-20
callers that used plain dicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


FEATURE_NAMES = [
    # Length features (3)
    "char_count",
    "word_count",
    "avg_word_length",
    # Casing features (3)
    "uppercase_ratio",
    "title_case_words",
    "all_caps_words",
    # Punctuation features (4)
    "exclamation_count",
    "question_count",
    "special_char_ratio",
    "consecutive_punctuation",
    # Structural markers (5)
    "line_count",
    "has_code_block",
    "has_url",
    "has_email",
    "newline_ratio",
    # Injection signal features (6)
    "imperative_start",
    "role_assignment",
    "instruction_boundary",
    "negation_command",
    "quote_depth",
    "text_entropy",
    # Context features (3)
    "question_sentence_ratio",
    "first_person_ratio",
    "second_person_ratio",
    # Advanced detection features (5)
    "many_shot_count",
    "delimiter_density",
    "template_marker_count",
    "language_mixing_score",
    "repetition_score",
]


@dataclass
class StructuralFeatures:
    """Typed container for structural feature extraction results.

    Supports dict-like access for backward compatibility with code that
    uses ``structural.get("key", default)``, ``structural["key"]``,
    or ``"key" in structural``.
    """

    # Length features (3)
    char_count: int = 0
    word_count: int = 0
    avg_word_length: float = 0.0
    # Casing features (3)
    uppercase_ratio: float = 0.0
    title_case_words: int = 0
    all_caps_words: int = 0
    # Punctuation features (4)
    exclamation_count: int = 0
    question_count: int = 0
    special_char_ratio: float = 0.0
    consecutive_punctuation: int = 0
    # Structural markers (5)
    line_count: int = 0
    has_code_block: int = 0
    has_url: int = 0
    has_email: int = 0
    newline_ratio: float = 0.0
    # Injection signal features (6)
    imperative_start: int = 0
    role_assignment: int = 0
    instruction_boundary: int = 0
    negation_command: int = 0
    quote_depth: int = 0
    text_entropy: float = 0.0
    # Context features (3)
    question_sentence_ratio: float = 0.0
    first_person_ratio: float = 0.0
    second_person_ratio: float = 0.0
    # Advanced detection features (5)
    many_shot_count: int = 0
    delimiter_density: float = 0.0
    template_marker_count: int = 0
    language_mixing_score: float = 0.0
    repetition_score: float = 0.0


    # ---- dict-like interface for backward compatibility ----

    def __getitem__(self, key):
        """Allow ``structural["key"]`` access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key):
        """Allow ``"key" in structural``."""
        return hasattr(self, key) and key in self.keys()

    def get(self, key, default=None):
        """Allow ``structural.get("key", default)``."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def keys(self):
        """Return feature names (matches FEATURE_NAMES order)."""
        return [f.name for f in fields(self)]

    def values(self):
        """Return feature values in FEATURE_NAMES order."""
        return [getattr(self, f.name) for f in fields(self)]

    def items(self):
        """Return (name, value) pairs in FEATURE_NAMES order."""
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def to_dict(self):
        """Convert to a plain dict."""
        return asdict(self)
