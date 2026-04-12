"""Layer 4: Lightweight perplexity-based signal for adversarial prompt detection.

Computes a pseudo-perplexity score using character-level Shannon entropy
and word-level out-of-vocabulary (OOV) ratio.  Adversarial prompts tend to
have unusual character distributions and many non-standard words.

Combined score:  ``0.4 * char_entropy_deviation + 0.6 * oov_ratio``

Score range:  [0.0, 1.0] where 1.0 = highly unnatural text.

This module has ZERO external dependencies beyond the Python stdlib and numpy.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ── Constants ──────────────────────────────────────────────────────────────

#: Above this threshold the text is suspiciously unnatural.
PERPLEXITY_THRESHOLD = 0.7

#: Minimum character count required to produce a meaningful signal.
_MIN_CHARS = 10

#: Expected Shannon entropy (bits) for typical English prose.
#: Empirical range for natural English is roughly 3.5-4.5 bits/char.
_ENGLISH_ENTROPY_BASELINE = 4.0

#: Maximum plausible entropy for any character distribution (log2(95)
#: for printable ASCII).  Used to normalize the deviation score.
_MAX_ENTROPY = math.log2(95)  # ~6.57

#: Minimum ratio of alphabetic characters to total characters for text
#: to be considered "word-bearing".  Below this, OOV ratio is set to 1.0
#: because the text is mostly non-alphabetic (encoded payloads, hex, etc.).
_MIN_ALPHA_RATIO = 0.4

# ── Common English word list (top 500, frozenset for O(1) lookup) ─────────

COMMON_WORDS: frozenset = frozenset({
    # Function words and pronouns
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "is", "are", "was", "were", "been", "being", "has", "had", "did", "does",
    "doing", "am", "having",
    # Common verbs
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "need", "should", "here", "still",
    "part", "hand", "high", "keep", "last", "point", "same", "more", "much",
    "why", "ask", "men", "went", "read", "may", "world", "each", "made",
    "find", "where", "before", "many", "those", "tell", "thing", "too",
    "right", "left", "life", "never", "old", "let", "while", "mean", "end",
    "might", "call", "under", "begin", "seem", "turn", "down", "side",
    "show", "every", "few", "next", "must", "big", "group", "such",
    "place", "again", "case", "number", "both", "during", "run", "feel",
    "between", "however", "another", "off", "always", "play", "move", "put",
    "set", "leave", "try", "far", "start", "line", "become", "stop",
    "hold", "hear", "bring", "carry", "eat", "cut", "sit", "stand",
    "lose", "pay", "meet", "grow", "lead", "live", "believe", "happen",
    "include", "continue", "follow", "create", "speak", "allow", "add",
    "spend", "return", "fall", "drive", "break", "receive", "agree",
    "support", "hit", "produce", "remember", "pass", "reach", "kill",
    "remain", "suggest", "raise", "cover", "describe", "develop",
    "pull", "pick", "build", "offer", "consider", "appear", "buy",
    "wait", "serve", "send", "expect", "stay", "drop", "plan",
    "draw", "decide", "reduce", "note", "enjoy", "love", "report",
    # Common nouns
    "children", "man", "woman", "school", "state", "family", "student",
    "country", "problem", "system", "city", "program", "question",
    "change", "government", "company", "night", "story", "room",
    "mother", "area", "money", "power", "home", "water", "door",
    "house", "child", "head", "body", "information", "word", "business",
    "eye", "face", "fact", "idea", "morning", "book", "month", "land",
    "food", "table", "war", "class", "field", "level", "paper",
    "force", "experience", "reason", "team", "mind", "type", "job",
    "age", "voice", "care", "town", "community", "sort", "position",
    "model", "rate", "process", "cost", "death", "girl", "view",
    "order", "price", "office", "issue", "wall", "picture", "party",
    "data", "heart", "color", "million", "friend", "law", "bed",
    "center", "figure", "hour", "member", "street", "art", "game",
    "music", "court", "result", "example", "interest", "person",
    "market", "name", "test", "step", "car", "list", "space",
    "boy", "rest", "oil", "air", "lot", "light", "service",
    # Common adjectives
    "great", "long", "very", "own", "small", "large", "different", "kind",
    "little", "hard", "important", "young", "real", "possible", "less",
    "able", "free", "strong", "sure", "true", "clear", "simple", "easy",
    "better", "best", "bad", "worst", "whole", "single", "major", "main",
    "special", "general", "political", "national", "local", "public",
    "social", "human", "available", "necessary", "certain", "similar",
    "ready", "full", "short", "low", "late", "early", "happy", "common",
    "open", "quick", "brown", "fast", "slow", "near", "close", "dark",
    "hot", "cold", "deep", "wide", "thin", "thick", "flat", "round",
    "soft", "sharp", "fresh", "quiet", "loud", "bright", "clean", "dry",
    "wet", "safe", "rich", "poor", "wrong", "fair", "rough", "nice",
    # Common adverbs
    "really", "actually", "simply", "either", "usually", "likely",
    "quite", "probably", "certainly", "particularly", "especially",
    "currently", "recently", "exactly", "basically", "directly",
    "else", "perhaps", "enough", "often", "already", "rather",
    "almost", "sometimes", "together", "ago", "today", "away",
    # Common everyday words
    "please", "thank", "yes", "okay", "sorry", "hello", "language",
    "himself", "herself", "itself", "myself", "yourself", "themselves",
    "anything", "something", "nothing", "everything", "someone",
    "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "hundred", "thousand", "since", "above", "below", "along", "across",
    "through", "behind", "against", "without", "among", "until",
    "whether", "though", "second", "later", "once", "half",
    # Animals, nature, everyday objects
    "dog", "cat", "bird", "fish", "horse", "tree", "flower", "sun",
    "moon", "star", "river", "sea", "mountain", "rain", "snow", "wind",
    "fire", "stone", "road", "bridge", "garden", "grass", "sky",
    # Additional common words for better coverage
    "fox", "lazy", "jumps", "jumped", "walking", "running", "going",
    "coming", "looking", "making", "working", "taking", "getting",
    "saying", "knowing", "thinking", "seeing", "wanting", "using",
    "finding", "giving", "telling", "asking", "trying", "leaving",
    "calling", "feeling", "becoming", "keeping", "beginning", "showing",
    "hearing", "playing", "moving", "living", "believing", "bringing",
    "writing", "sitting", "standing", "losing", "paying", "meeting",
    "learning", "leading", "understanding", "watching", "following",
    "turning", "talking", "helping", "starting", "reading", "singing",
    "walking", "shining", "science", "article", "summary", "capital",
    "weather", "today", "tomorrow", "yesterday",
    "write", "study", "learn", "teach", "explain", "describe",
    "understand", "watch", "listen", "answer", "question", "talk",
    "send", "receive", "check", "control", "protect", "save",
    "white", "black", "blue", "green", "red", "yellow",
    "north", "south", "east", "west",
})


# ── Core Functions ─────────────────────────────────────────────────────────


def _char_entropy(text: str) -> float:
    """Compute Shannon entropy (bits) of the character distribution in *text*.

    Uses ``math.log2`` per requirement.  Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _oov_ratio(text: str) -> float:
    """Return the fraction of words in *text* NOT in :data:`COMMON_WORDS`.

    Words are extracted by splitting on non-alpha characters and lowercasing.
    Single-character tokens and empty strings are excluded to avoid noise.

    If the text has very few alphabetic characters relative to its length
    (below ``_MIN_ALPHA_RATIO``), returns 1.0 because the text is mostly
    non-alphabetic content (hex, encoded payloads, special characters, etc.)
    which is inherently unnatural.

    Returns 0.0 when there are no valid words and the text is short enough
    that the alpha-ratio check doesn't trigger.
    """
    # Check alpha ratio — long non-alphabetic text is unnatural
    alpha_count = sum(1 for c in text if c.isalpha())
    if len(text) >= _MIN_CHARS and alpha_count / len(text) < _MIN_ALPHA_RATIO:
        return 1.0

    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text) if len(w) > 1]
    if not words:
        # Long text with no recognizable words (e.g. hex strings, encoded
        # payloads) is inherently unnatural.
        return 1.0 if len(text) >= _MIN_CHARS else 0.0
    oov_count = sum(1 for w in words if w not in COMMON_WORDS)
    return oov_count / len(words)


def compute_perplexity(text: str) -> float:
    """Return a pseudo-perplexity score in [0.0, 1.0].

    * 0.0 = perfectly natural English
    * 1.0 = highly unnatural / adversarial

    Returns 0.0 for empty strings or text shorter than ``_MIN_CHARS``
    (not enough signal to judge).

    Combined formula::

        score = 0.4 * char_entropy_deviation + 0.6 * oov_ratio

    where ``char_entropy_deviation`` is the absolute difference between the
    observed character entropy and the English baseline, normalized to [0, 1].
    """
    if not text or len(text) < _MIN_CHARS:
        return 0.0

    # --- Character entropy deviation ---
    entropy = _char_entropy(text)
    deviation = abs(entropy - _ENGLISH_ENTROPY_BASELINE)
    # Normalize: max possible deviation is _MAX_ENTROPY (if entropy is 0 or
    # at theoretical max).  We use _MAX_ENTROPY as the normalizer.
    char_score = min(deviation / _MAX_ENTROPY, 1.0)

    # --- Word-level OOV ratio ---
    oov = _oov_ratio(text)

    # --- Combined score ---
    score = 0.4 * char_score + 0.6 * oov
    return min(max(score, 0.0), 1.0)
