"""Quote-depth computation with apostrophe heuristic.

A single-quote preceded by an alphanumeric character (e.g. "it's",
"don't") is treated as an apostrophe rather than a quote delimiter,
so contractions don't inflate the nesting count.
"""

from __future__ import annotations


def _compute_quote_depth(text):
    """Return the maximum nesting depth of quotes (single, double, backtick).

    Handles apostrophes correctly: a single quote preceded by a word
    character (e.g. "it's", "don't") is treated as an apostrophe,
    NOT a quote delimiter.
    """
    max_depth = 0
    stack = []
    for i, ch in enumerate(text):
        if ch not in ('"', "'", '`'):
            continue
        # Apostrophe heuristic: single quote preceded by a word char
        # (letter/digit) is an apostrophe, not a quote delimiter,
        # UNLESS it matches the innermost open quote (closing it).
        if ch == "'" and i > 0 and text[i - 1].isalnum():
            # It's an apostrophe -- skip it, unless it's closing
            # a previously opened single-quote on the stack.
            if stack and stack[-1] == "'":
                # Check if it looks like a closing quote: the char
                # after it must be non-alphanumeric or end-of-string.
                next_idx = i + 1
                if next_idx >= len(text) or not text[next_idx].isalnum():
                    stack.pop()  # closing single-quote
                # else: apostrophe in the middle of a word, skip
            continue
        # Normal quote character
        if stack and stack[-1] == ch:
            stack.pop()          # closing quote
        else:
            stack.append(ch)     # opening quote
            if len(stack) > max_depth:
                max_depth = len(stack)
    return max_depth
