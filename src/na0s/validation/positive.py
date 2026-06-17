"""PositiveValidator -- verify input IS a legitimate prompt.

Instead of only checking if input looks malicious (blocklisting),
also verify it looks like what a legitimate user would send
(allowlisting). This dramatically reduces false positives because
benign prompts about security topics PASS positive validation even
though they FAIL blocklist checks.

Inspired by the Snyk Fetch the Flag 2026 "AI WAF" challenge which
used task validation and sandwich defense as two of its six defense
layers.  Academic research (SaTML 2024 LLM CTF) showed that ALL 44
pure-blocklist defenses were eventually bypassed, but multi-layer
defense with positive validation was the most resilient.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..layer1 import PERSONA_OVERRIDE_PATTERNS


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of positive validation."""

    is_valid: bool
    confidence: float          # 0.0 .. 1.0
    reason: str
    task_match: float          # 0.0 .. 1.0 -- how well input fits expected task
    technique_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "shall",
    "has", "have", "had", "may", "might",
}

_COMMON_VERBS = {
    "explain", "describe", "write", "create", "generate", "make",
    "list", "tell", "show", "give", "find", "help", "translate",
    "summarize", "analyze", "compare", "define", "calculate", "solve",
    "convert", "fix", "debug", "implement", "build", "design",
    "outline", "clarify", "elaborate", "suggest", "recommend",
    "improve", "rewrite", "edit", "check", "review", "test",
    "run", "execute", "deploy", "install", "configure", "set",
    "get", "fetch", "search", "read", "open", "close", "start",
    "stop", "update", "delete", "add", "remove", "change",
    "know", "think", "need", "want", "like", "use", "try",
    "provide", "include", "consider", "continue", "understand",
}

# FIX BUG-L8-5: Persona override patterns consolidated in rules.py.
# Imported as PERSONA_OVERRIDE_PATTERNS above -- single source of truth.
_PERSONA_OVERRIDE_PATTERNS = PERSONA_OVERRIDE_PATTERNS

# Shared with validate_output() in trust_boundary.py -- both the input
# validator and the output validator flag these chat-template markers.
_SYSTEM_PROMPT_MARKERS = [
    "[SYSTEM]", "<<SYS>>", "[INST]", "</s>", "<|im_start|>",
    "<|im_end|>", "[/INST]", "<</SYS>>", "### System:",
    "### Human:", "### Assistant:", "<|system|>", "<|user|>",
    "<|assistant|>",
]

_INSTRUCTION_BOUNDARIES = ["---", "===", "***", "###", "```"]

# ---------------------------------------------------------------------------
# Taxonomy mapping: validation failure -> technique ID
# ---------------------------------------------------------------------------

VALIDATION_TAXONOMY_MAP: Dict[str, str] = {
    "persona_override": "D2",       # persona hijack
    "system_prompt_markers": "D3",  # structural boundary
    "low_coherence": "D4",          # obfuscation
    "contradiction": "D1",          # instruction override
    "boundary_count": "D3",         # structural boundary
}

# ---------------------------------------------------------------------------
# Default check weights (P1: configurable)
# ---------------------------------------------------------------------------

DEFAULT_VALIDATION_WEIGHTS: Dict[str, float] = {
    "coherence": 0.15,
    "intent": 0.15,
    "scope": 0.20,
    "persona": 0.30,
    "task": 0.20,
}

# ---------------------------------------------------------------------------
# Task-match keyword sets (used by _check_task_match)
# ---------------------------------------------------------------------------

_SUMMARIZATION_KEYWORDS = {
    "summarize", "summary", "summarise", "key points", "tl;dr",
    "tldr", "main points", "brief overview", "condense", "recap",
    "gist", "abstract", "synopsis",
}

_CODING_KEYWORDS = {
    "function", "class", "bug", "error", "code", "implement",
    "fix", "debug", "compile", "runtime", "syntax", "variable",
    "loop", "array", "string", "integer", "exception", "stack",
    "trace", "refactor", "algorithm", "api", "endpoint", "module",
    "import", "library", "framework", "test", "unit test", "deploy",
    "script", "program", "method", "object", "inheritance",
    "python", "javascript", "java", "typescript", "rust", "go",
    "c++", "html", "css", "sql", "react", "django", "flask",
}


# ---------------------------------------------------------------------------
# PositiveValidator
# ---------------------------------------------------------------------------

class PositiveValidator:
    """Verify that input IS a legitimate prompt (allowlisting).

    Parameters
    ----------
    task_type : str
        One of ``"general"``, ``"summarization"``, ``"qa"``, ``"coding"``.
    """

    VALID_TASK_TYPES = {"general", "summarization", "qa", "coding"}

    def __init__(
        self,
        task_type: str = "general",
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if task_type not in self.VALID_TASK_TYPES:
            raise ValueError(
                f"Unknown task_type {task_type!r}. "
                f"Choose from {sorted(self.VALID_TASK_TYPES)}."
            )
        self.task_type = task_type

        # Resolve weights: explicit param > env var > defaults
        if weights is not None:
            self.weights = dict(weights)
        else:
            env_weights = os.environ.get("NA0S_VALIDATION_WEIGHTS")
            if env_weights:
                try:
                    self.weights = json.loads(env_weights)
                except (json.JSONDecodeError, TypeError):
                    self.weights = dict(DEFAULT_VALIDATION_WEIGHTS)
            else:
                self.weights = dict(DEFAULT_VALIDATION_WEIGHTS)

    # ---- public API -------------------------------------------------------

    def validate(self, text: str, sanitized_text: Optional[str] = None) -> ValidationResult:
        """Run all positive-validation checks and return an aggregate result.

        Parameters
        ----------
        text : str
            The input text to validate.  When *sanitized_text* is provided
            this parameter is ignored in favour of the sanitized version.
        sanitized_text : str or None
            L0-sanitized text.  When provided, validation runs on this
            instead of *text* so that the positive validator sees the same
            normalized form as the rest of the pipeline (BUG-L8-2 fix).
        """
        # BUG-L8-7 fix: guard against non-string input to prevent
        # AttributeError on None / int / list etc.
        effective = sanitized_text if sanitized_text is not None else text
        if not isinstance(effective, str):
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                reason="Non-string input.",
                task_match=0.0,
            )
        text = effective

        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                reason="Empty input.",
                task_match=0.0,
            )

        text = text.strip()
        issues: List[str] = []
        technique_ids: List[str] = []
        weighted_scores: List[tuple] = []  # (weight_key, score)

        # 1. Coherence
        coh_ok, coh_score, coh_reason = self._check_coherence(text)
        weighted_scores.append(("coherence", coh_score))
        if not coh_ok:
            issues.append(coh_reason)
            tid = VALIDATION_TAXONOMY_MAP.get("low_coherence")
            if tid and tid not in technique_ids:
                technique_ids.append(tid)

        # 2. Intent
        int_ok, int_score, int_reason = self._check_intent(text)
        weighted_scores.append(("intent", int_score))
        if not int_ok:
            issues.append(int_reason)

        # 3. Scope
        scp_ok, scp_score, scp_reason = self._check_scope(text)
        weighted_scores.append(("scope", scp_score))
        if not scp_ok:
            issues.append(scp_reason)
            # Map specific scope failures to technique IDs
            if "Contradictory" in scp_reason:
                tid = VALIDATION_TAXONOMY_MAP.get("contradiction")
                if tid and tid not in technique_ids:
                    technique_ids.append(tid)
            if "instruction boundaries" in scp_reason:
                tid = VALIDATION_TAXONOMY_MAP.get("boundary_count")
                if tid and tid not in technique_ids:
                    technique_ids.append(tid)

        # 4. Persona boundary
        per_ok, per_score, per_reason = self._check_persona_boundary(text)
        weighted_scores.append(("persona", per_score))
        if not per_ok:
            issues.append(per_reason)
            if "Persona override" in per_reason:
                tid = VALIDATION_TAXONOMY_MAP.get("persona_override")
                if tid and tid not in technique_ids:
                    technique_ids.append(tid)
            if "System prompt marker" in per_reason:
                tid = VALIDATION_TAXONOMY_MAP.get("system_prompt_markers")
                if tid and tid not in technique_ids:
                    technique_ids.append(tid)

        # 5. Task match
        task_match = self._check_task_match(text)
        weighted_scores.append(("task", task_match))

        # Weighted confidence instead of simple mean
        total_weight = 0.0
        weighted_sum = 0.0
        for key, score in weighted_scores:
            w = self.weights.get(key, 0.0)
            weighted_sum += w * score
            total_weight += w
        confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        is_valid = len(issues) == 0
        reason = "All checks passed." if is_valid else " | ".join(issues)

        return ValidationResult(
            is_valid=is_valid,
            confidence=round(confidence, 4),
            reason=reason,
            task_match=round(task_match, 4),
            technique_ids=technique_ids,
        )

    # ---- coherence --------------------------------------------------------

    # Per-task alpha_ratio thresholds (BUG-L8-3).
    # Code, JSON, URLs, and log output contain many symbols/punctuation,
    # so coding tasks need a lower threshold to avoid false rejections.
    _ALPHA_RATIO_THRESHOLDS = {
        "coding": 0.15,
        "general": 0.30,
        "summarization": 0.30,
        "qa": 0.30,
    }

    # Per-task avg_word_len thresholds (BUG-L8-6).
    # English average word length is ~5 chars; even long technical words
    # (e.g., "internationalization" = 20 chars, "deinstitutionalization"
    # = 22 chars) are well under 25.  A threshold of 25 catches encoded
    # or concatenated blobs while allowing all legitimate vocabulary.
    # Coding tasks allow slightly longer because identifiers like
    # `AbstractSingletonProxyFactoryBean` (35 chars) are real.
    _AVG_WORD_LEN_THRESHOLDS = {
        "coding": 35,
        "general": 25,
        "summarization": 25,
        "qa": 25,
    }

    def _check_coherence(self, text: str) -> tuple:
        """Text is readable natural language, not gibberish or encoded."""
        words = text.split()
        num_words = len(words)
        num_chars = len(text)

        # Word-to-character ratio check
        if num_words == 0:
            return (False, 0.0, "No words detected.")
        avg_word_len = num_chars / num_words

        # BUG-L8-6: Use per-task avg_word_len threshold instead of
        # hard-coded 45.  See _AVG_WORD_LEN_THRESHOLDS for rationale.
        avg_word_len_limit = self._AVG_WORD_LEN_THRESHOLDS.get(
            self.task_type, 25,
        )
        if avg_word_len > avg_word_len_limit:
            return (False, 0.1, "Text appears encoded or lacks word boundaries.")

        # At least some words with length > 2 (recognizable words)
        long_words = [w for w in words if len(w) > 2]
        long_ratio = len(long_words) / num_words if num_words else 0.0
        if long_ratio < 0.15:
            return (False, 0.2, "Text lacks recognizable words (too many single/two-char tokens).")

        # BUG-L8-3: Use per-task alpha_ratio threshold instead of
        # hard-coded 0.30.  See _ALPHA_RATIO_THRESHOLDS for rationale.
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / num_chars if num_chars else 0.0
        alpha_threshold = self._ALPHA_RATIO_THRESHOLDS.get(
            self.task_type, 0.30,
        )
        if alpha_ratio < alpha_threshold:
            return (False, 0.2, "Text is mostly special characters or numbers.")

        # Score: higher is more coherent
        score = min(1.0, 0.4 + long_ratio * 0.3 + alpha_ratio * 0.3)
        return (True, round(score, 4), "")

    # ---- intent -----------------------------------------------------------

    def _check_intent(self, text: str) -> tuple:
        """Text expresses a clear user intent."""
        lower = text.lower()
        words_set = set(re.findall(r"[a-z]+", lower))

        has_question_word = bool(words_set & _QUESTION_WORDS)
        has_verb = bool(words_set & _COMMON_VERBS)
        ends_with_question = text.rstrip().endswith("?")

        # A question or a command both count as intent
        if has_question_word or has_verb or ends_with_question:
            score = 0.6
            if has_verb:
                score += 0.2
            if has_question_word or ends_with_question:
                score += 0.2
            return (True, min(1.0, score), "")

        return (False, 0.2, "No clear intent detected (missing verb or question).")

    # ---- scope ------------------------------------------------------------

    def _check_scope(self, text: str) -> tuple:
        """Text is a single, bounded request."""
        issues: List[str] = []
        score = 1.0

        # Length check
        max_length = {
            "general": 2000,
            "summarization": 10000,  # may include long text to summarize
            "qa": 1000,
            "coding": 5000,
        }.get(self.task_type, 2000)

        if len(text) > max_length:
            issues.append(f"Input exceeds max length ({len(text)} > {max_length}).")
            score -= 0.4

        # Multiple instruction boundaries
        boundary_count = sum(text.count(b) for b in _INSTRUCTION_BOUNDARIES)
        if boundary_count >= 3:
            issues.append(
                f"Multiple instruction boundaries detected ({boundary_count})."
            )
            score -= 0.3

        # Contradictory instructions heuristic (BUG-L8-4 fix)
        #
        # Window size rationale: {1,500} covers ~80-100 words of typical
        # English text (avg ~5 chars/word + space).  Attackers commonly
        # insert 50-80 words of benign filler between contradictory
        # phrases to evade narrow-window detection.  500 chars is wide
        # enough to catch realistic evasion payloads while staying safely
        # bounded against ReDoS (all quantifiers are finite).
        # re.DOTALL ensures newlines within the gap are matched too.
        contradiction_patterns = [
            re.compile(r"\bdo\b.{1,500}\bbut\s+(also\s+)?ignore\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bfollow\b.{1,500}\bbut\s+(also\s+)?disregard\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bobey\b.{1,500}\bbut\s+(also\s+)?override\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bmust\b.{1,500}\bbut\s+(actually\s+)?ignore\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bcomply\b.{1,500}\bbut\s+(also\s+)?skip\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\bfollow\b.{1,500}\bbut\s+(actually\s+)?forget\b", re.IGNORECASE | re.DOTALL),
        ]
        for pat in contradiction_patterns:
            if pat.search(text):
                issues.append("Contradictory instructions detected.")
                score -= 0.3
                break

        # Sentence-level contradiction detection: look for contradictory
        # intent across any two sentences, regardless of distance.  This
        # catches payloads where the "setup" and "reversal" sentences are
        # far apart or separated by paragraph breaks.
        if "Contradictory instructions detected." not in issues:
            if self._has_sentence_level_contradiction(text):
                issues.append("Contradictory instructions detected.")
                score -= 0.3

        score = max(0.0, score)
        if issues:
            return (False, round(score, 4), " ".join(issues))
        return (True, 1.0, "")

    # ---- sentence-level contradiction (BUG-L8-4) -------------------------

    # Keyword sets for sentence-level contradiction detection.
    # "setup" words establish a directive; "reversal" words negate it.
    _SETUP_KEYWORDS = re.compile(
        r"\b(do|follow|obey|must|comply|always|ensure|stick\s+to"
        r"|adhere\s+to|respect|observe|execute|perform)\b",
        re.IGNORECASE,
    )
    _REVERSAL_KEYWORDS = re.compile(
        r"\b(ignore|disregard|override|forget|skip|bypass|neglect"
        r"|disobey|circumvent|never\s+mind|actually\s+ignore"
        r"|actually\s+disregard|instead\s+ignore)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences on terminal punctuation or newlines."""
        # Split on .!? followed by whitespace/end, or on newlines
        parts = re.split(r"[.!?]+(?:\s|$)|\n+", text)
        return [p.strip() for p in parts if p and p.strip()]

    def _has_sentence_level_contradiction(self, text: str) -> bool:
        """Detect contradictory intent across any pair of sentences.

        Returns True if one sentence contains a setup keyword (e.g.
        "follow", "must", "obey") and a different sentence contains a
        reversal keyword (e.g. "ignore", "disregard", "override").
        Both must be present in distinct sentences for a contradiction.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return False

        has_setup = False
        has_reversal = False
        setup_idx = -1
        reversal_idx = -1

        for i, sent in enumerate(sentences):
            if not has_setup and self._SETUP_KEYWORDS.search(sent):
                has_setup = True
                setup_idx = i
            if not has_reversal and self._REVERSAL_KEYWORDS.search(sent):
                has_reversal = True
                reversal_idx = i

        # Both must be present AND in different sentences
        if has_setup and has_reversal and setup_idx != reversal_idx:
            return True

        return False

    # ---- persona boundary -------------------------------------------------

    def _check_persona_boundary(self, text: str) -> tuple:
        """Text does not try to redefine the assistant."""
        for pat in _PERSONA_OVERRIDE_PATTERNS:
            if pat.search(text):
                return (False, 0.1, f"Persona override attempt detected: {pat.pattern!r}.")

        upper = text.upper()
        for marker in _SYSTEM_PROMPT_MARKERS:
            if marker.upper() in upper:
                return (False, 0.1, f"System prompt marker detected: {marker!r}.")

        return (True, 1.0, "")

    # ---- task match -------------------------------------------------------

    def _check_task_match(self, text: str) -> float:
        """How well does input match the expected task type?"""
        if self.task_type == "general":
            # General always gives moderate match -- anything plausible is fine
            return 0.7

        lower = text.lower()
        words_set = set(re.findall(r"[a-z]+", lower))

        if self.task_type == "summarization":
            hits = sum(1 for kw in _SUMMARIZATION_KEYWORDS if kw in lower)
            # Summarization prompts usually also include a block of text
            has_text_block = len(text) > 100
            score = min(1.0, hits * 0.25 + (0.3 if has_text_block else 0.0))
            return round(score, 4)

        if self.task_type == "qa":
            ends_q = text.rstrip().endswith("?")
            has_qword = bool(words_set & _QUESTION_WORDS)
            score = 0.0
            if ends_q:
                score += 0.5
            if has_qword:
                score += 0.5
            return round(min(1.0, score), 4)

        if self.task_type == "coding":
            hits = sum(1 for kw in _CODING_KEYWORDS if kw in lower)
            score = min(1.0, hits * 0.15)
            return round(score, 4)

        return 0.0


__all__ = [
    "DEFAULT_VALIDATION_WEIGHTS",
    "PositiveValidator",
    "VALIDATION_TAXONOMY_MAP",
    "ValidationResult",
]
