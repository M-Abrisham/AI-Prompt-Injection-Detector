"""Main structural-feature extraction pipeline.

Exposes :func:`extract_structural_features` (single text) and
:func:`extract_structural_features_batch` (vectorised).  Small scoring
helpers (entropy, script-family count, repetition n-gram ratio) live in
this module because they're only used by the extract functions.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .features import FEATURE_NAMES, StructuralFeatures
from .patterns import (
    _BOUNDARY_PATTERNS,
    _CONSECUTIVE_PUNCT,
    _DELIMITER_PATTERN,
    _EMAIL_PATTERN,
    _FIRST_PERSON,
    _IMPERATIVE_VERBS,
    _MANY_SHOT_PATTERN,
    _NEGATION_COMMAND,
    _ROLE_PATTERNS,
    _SCRIPT_RANGES,
    _SECOND_PERSON,
    _TEMPLATE_MARKER_PATTERN,
    _URL_PATTERN,
)
from .quotes import _compute_quote_depth
from .sentences import _split_sentences


def _count_script_families(text):
    """Count distinct Unicode script families with significant presence.

    Returns the number of script families that have >= 3 characters
    in the text.  A score > 1 indicates multiple scripts are mixed
    (potential multilingual bypass).  Returns 0 for empty text.
    """
    if not text:
        return 0
    count = 0
    for _name, pattern in _SCRIPT_RANGES:
        if len(pattern.findall(text)) >= 3:
            count += 1
    return count


def _compute_repetition_score(words, n=3):
    """Compute word-level n-gram repetition ratio.

    Returns the fraction of unique n-grams that appear more than once.
    High values indicate repetitive patterns (resource exhaustion,
    crescendo attacks, many-shot jailbreaking).
    """
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


def _compute_entropy(text):
    """Shannon entropy of the character distribution in *text*."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_structural_features(text):
    """Extract structural (non-lexical) features from *text*.

    Parameters
    ----------
    text : str or None
        The input prompt.  ``None`` is treated as an empty string.

    Returns
    -------
    StructuralFeatures
        A dataclass with typed fields matching :data:`FEATURE_NAMES`.
        Supports dict-like access (``result["key"]``, ``result.get()``,
        ``"key" in result``) for backward compatibility.
    """
    if text is None:
        text = ""

    # ------------------------------------------------------------------
    # Pre-compute shared quantities
    # ------------------------------------------------------------------
    chars = list(text)
    words = text.split()               # whitespace-separated tokens
    lines = text.split("\n")
    alpha_chars = [c for c in chars if c.isalpha()]
    sentences = _split_sentences(text)

    char_count = len(text)
    word_count = len(words)
    line_count = len(lines)
    alpha_count = len(alpha_chars)
    sentence_count = len(sentences) if sentences else 0

    # ------------------------------------------------------------------
    # 1. Length features
    # ------------------------------------------------------------------
    avg_word_length = (char_count / word_count) if word_count else 0.0

    # ------------------------------------------------------------------
    # 2. Casing features
    # ------------------------------------------------------------------
    upper_alpha = sum(1 for c in alpha_chars if c.isupper())
    uppercase_ratio = (upper_alpha / alpha_count) if alpha_count else 0.0

    title_case_words = sum(
        1 for w in words if len(w) >= 2 and w[0].isupper() and w[1:].islower()
    )
    all_caps_words = sum(
        1 for w in words if len(w) >= 2 and w.isupper()
    )

    # ------------------------------------------------------------------
    # 3. Punctuation features
    # ------------------------------------------------------------------
    exclamation_count = text.count("!")
    question_count = text.count("?")

    non_alnum_non_space = sum(
        1 for c in chars if not c.isalnum() and not c.isspace()
    )
    special_char_ratio = (non_alnum_non_space / char_count) if char_count else 0.0

    consecutive_punctuation = len(_CONSECUTIVE_PUNCT.findall(text))

    # ------------------------------------------------------------------
    # 4. Structural markers
    # ------------------------------------------------------------------
    has_code_block = 1 if "```" in text else 0
    has_url = 1 if _URL_PATTERN.search(text) else 0
    has_email = 1 if _EMAIL_PATTERN.search(text) else 0
    newline_ratio = (line_count / word_count) if word_count else 0.0

    # ------------------------------------------------------------------
    # 5. Injection signal features
    # ------------------------------------------------------------------
    first_word = words[0].lower().strip("\"'`([{") if words else ""
    imperative_start = 1 if first_word in _IMPERATIVE_VERBS else 0

    role_assignment = 1 if _ROLE_PATTERNS.search(text) else 0
    instruction_boundary = 1 if _BOUNDARY_PATTERNS.search(text) else 0
    negation_command = 1 if _NEGATION_COMMAND.search(text) else 0

    quote_depth = _compute_quote_depth(text)
    text_entropy = _compute_entropy(text)

    # ------------------------------------------------------------------
    # 6. Context features
    # ------------------------------------------------------------------
    if sentence_count > 0:
        question_sentences = sum(
            1 for s in sentences if s.rstrip().endswith("?")
        )
        first_person_sentences = sum(
            1 for s in sentences if _FIRST_PERSON.search(s)
        )
        second_person_sentences = sum(
            1 for s in sentences if _SECOND_PERSON.search(s)
        )
        question_sentence_ratio = question_sentences / sentence_count
        first_person_ratio = first_person_sentences / sentence_count
        second_person_ratio = second_person_sentences / sentence_count
    else:
        question_sentence_ratio = 0.0
        first_person_ratio = 0.0
        second_person_ratio = 0.0

    # ------------------------------------------------------------------
    # 7. Advanced detection features
    # ------------------------------------------------------------------
    # Many-shot detection: count repeated instruction/example patterns
    many_shot_count = len(_MANY_SHOT_PATTERN.findall(text))

    # Delimiter density: ratio of markdown/XML delimiters per line
    delimiter_matches = len(_DELIMITER_PATTERN.findall(text))
    delimiter_density = (delimiter_matches / line_count) if line_count else 0.0

    # Prompt template markers: count {{var}}, {placeholder}, <|slot|>
    template_marker_count = len(_TEMPLATE_MARKER_PATTERN.findall(text))

    # Language mixing score: number of distinct script families
    script_families = _count_script_families(text)
    language_mixing_score = float(script_families) if script_families > 1 else 0.0

    # Repetition score: word-level trigram repetition ratio
    repetition_score = _compute_repetition_score(words, n=3)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return StructuralFeatures(
        char_count=char_count,
        word_count=word_count,
        avg_word_length=avg_word_length,
        uppercase_ratio=uppercase_ratio,
        title_case_words=title_case_words,
        all_caps_words=all_caps_words,
        exclamation_count=exclamation_count,
        question_count=question_count,
        special_char_ratio=special_char_ratio,
        consecutive_punctuation=consecutive_punctuation,
        line_count=line_count,
        has_code_block=has_code_block,
        has_url=has_url,
        has_email=has_email,
        newline_ratio=newline_ratio,
        imperative_start=imperative_start,
        role_assignment=role_assignment,
        instruction_boundary=instruction_boundary,
        negation_command=negation_command,
        quote_depth=quote_depth,
        text_entropy=text_entropy,
        question_sentence_ratio=question_sentence_ratio,
        first_person_ratio=first_person_ratio,
        second_person_ratio=second_person_ratio,
        many_shot_count=many_shot_count,
        delimiter_density=delimiter_density,
        template_marker_count=template_marker_count,
        language_mixing_score=language_mixing_score,
        repetition_score=repetition_score,
    )


def extract_structural_features_batch(texts, normalize=False):
    """Extract structural features for a list of texts.

    Parameters
    ----------
    texts : list[str]
        Input prompts.  ``None`` entries are treated as empty strings.
    normalize : bool, optional
        If ``True``, unbounded features are scaled to [0, 1] using soft
        caps (see :func:`na0s.structural.normalize_features`).  Default
        is ``False`` to preserve backward compatibility with
        ``predict.py``'s raw-value thresholds.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(texts), len(FEATURE_NAMES))``.
    """
    rows = []
    for text in texts:
        feat = extract_structural_features(text)
        rows.append([feat[name] for name in FEATURE_NAMES])
    arr = np.array(rows, dtype=np.float64)
    if normalize:
        # Local import avoids a circular dependency at module load time:
        # normalize.py imports FEATURE_NAMES from features.py, which
        # is already imported here.
        from .normalize import normalize_features
        arr = normalize_features(arr)
    return arr
