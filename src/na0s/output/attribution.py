"""RAG attribution verification — flag LLM output not grounded in retrieved context.

If the LLM's response includes instructions or content NOT present in the
retrieved context documents, that's a sign the injection succeeded: the LLM
is following injected instructions rather than the legitimate context.

Gated by ``NA0S_RAG_ATTRIBUTION=1`` env var (default: disabled).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List


# Patterns that suggest injected instructions when found in ungrounded segments
_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(all\s+)?previous\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\byou\s+must\b", re.IGNORECASE),
    re.compile(r"\byou\s+should\s+now\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+follow\b", re.IGNORECASE),
    re.compile(r"\bdisregard\b", re.IGNORECASE),
    re.compile(r"\boverride\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(all\s+)?previous\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
]


def _tokenise(text: str) -> set[str]:
    """Return lowercased word tokens, stripping punctuation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class RAGAttributionChecker:
    """Check whether LLM output is grounded in the retrieved context."""

    def __init__(self, overlap_threshold: float = 0.3) -> None:
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def is_enabled() -> bool:
        """Return True when ``NA0S_RAG_ATTRIBUTION`` is set to a truthy value."""
        return os.environ.get("NA0S_RAG_ATTRIBUTION", "").strip() in ("1", "true", "yes")

    def check(
        self,
        output_text: str,
        context_documents: list[str],
    ) -> Dict[str, object]:
        """Verify that *output_text* is grounded in *context_documents*.

        Returns
        -------
        dict
            ``is_grounded``, ``attribution_score``, ``ungrounded_segments``,
            ``flags``.
        """
        # Edge cases
        if not output_text or not output_text.strip():
            return {
                "is_grounded": True,
                "attribution_score": 1.0,
                "ungrounded_segments": [],
                "flags": [],
            }

        # Build combined context word set
        context_words: set[str] = set()
        for doc in (context_documents or []):
            context_words |= _tokenise(doc)

        sentences = _split_sentences(output_text)
        if not sentences:
            return {
                "is_grounded": True,
                "attribution_score": 1.0,
                "ungrounded_segments": [],
                "flags": [],
            }

        ungrounded: List[str] = []
        grounded_count = 0

        for sentence in sentences:
            sent_words = _tokenise(sentence)
            if not sent_words:
                grounded_count += 1
                continue

            if not context_words:
                # No context at all — everything is ungrounded
                ungrounded.append(sentence)
                continue

            overlap = len(sent_words & context_words) / len(sent_words)
            if overlap >= self.overlap_threshold:
                grounded_count += 1
            else:
                ungrounded.append(sentence)

        attribution_score = grounded_count / len(sentences) if sentences else 1.0

        # Detect injection patterns in ungrounded segments
        flags: List[str] = []
        for segment in ungrounded:
            for pat in _INJECTION_PATTERNS:
                if pat.search(segment):
                    flag_msg = f"injection_pattern_in_ungrounded: {pat.pattern!r}"
                    if flag_msg not in flags:
                        flags.append(flag_msg)

        return {
            "is_grounded": attribution_score >= self.overlap_threshold,
            "attribution_score": round(attribution_score, 4),
            "ungrounded_segments": ungrounded,
            "flags": flags,
        }


def verify_attribution(output: str, context: str) -> dict:
    """Convenience function: check if *output* is grounded in *context*.

    Uses token overlap ratio and novel content detection.

    Returns
    -------
    dict
        ``{"grounded": bool, "grounding_score": float, "novel_segments": list}``
    """
    if not output or not output.strip():
        return {
            "grounded": True,
            "grounding_score": 1.0,
            "novel_segments": [],
        }

    context_words = _tokenise(context) if context else set()
    sentences = _split_sentences(output)

    if not sentences:
        return {
            "grounded": True,
            "grounding_score": 1.0,
            "novel_segments": [],
        }

    novel_segments: list[str] = []
    grounded_count = 0
    threshold = 0.3

    for sentence in sentences:
        sent_words = _tokenise(sentence)
        if not sent_words:
            grounded_count += 1
            continue

        if not context_words:
            novel_segments.append(sentence)
            continue

        overlap = len(sent_words & context_words) / len(sent_words)
        if overlap >= threshold:
            grounded_count += 1
        else:
            novel_segments.append(sentence)

    grounding_score = grounded_count / len(sentences) if sentences else 1.0

    return {
        "grounded": grounding_score >= threshold,
        "grounding_score": round(grounding_score, 4),
        "novel_segments": novel_segments,
    }
