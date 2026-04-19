"""Sentence splitting with abbreviation awareness.

Used by :func:`extract_structural_features` to compute
``question_sentence_ratio``, ``first_person_ratio``, and
``second_person_ratio``.
"""

from __future__ import annotations

import re

from .patterns import _ABBREVIATIONS


def _split_sentences(text):
    """Split text into sentences.  Returns a list of non-empty strings.

    Uses a heuristic that avoids splitting on common abbreviations
    (e.g. "Dr.", "Mr.", "e.g.", "U.S.A.") while correctly splitting
    on sentence-ending periods.
    """
    if not text or not text.strip():
        return []

    # Strategy: first split on unambiguous terminators (! ?), then
    # handle period-based splits with abbreviation awareness.
    #
    # We use a single-pass approach: find all ". " positions and
    # decide whether each is a sentence boundary or abbreviation.
    # Find candidate split points: period followed by optional closing
    # quotes/parens, then whitespace
    result_parts = []
    last = 0

    for m in re.finditer(r'([.!?])["\')]*\s+', text):
        punct = m.group(1)
        split_pos = m.end()

        if punct == '.':
            # Check if the word before the period is an abbreviation
            # or a single uppercase letter (initial)
            before = text[last:m.start()].rstrip()
            # Extract the last "word" before the period
            last_word_match = re.search(r'(\S+)$', before)
            if last_word_match:
                last_word = last_word_match.group(1).lower().rstrip('.')
                # Skip if it's a known abbreviation
                if last_word in _ABBREVIATIONS:
                    continue
                # Skip if it's a single letter (initial like "U." in U.S.A.)
                if len(last_word) == 1 and last_word.isalpha():
                    continue

        # This is a real sentence boundary
        result_parts.append(text[last:m.start() + 1])  # include the punct
        last = split_pos

    # Add the remaining text
    if last < len(text):
        result_parts.append(text[last:])

    return [s.strip() for s in result_parts if s.strip()]
