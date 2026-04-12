"""Diffusion-based Denoising Defense (DiffuseDef) for adversarial robustness.

This module implements a text denoising layer that neutralizes adversarial
perturbations (typos, character swaps, Unicode tricks, leetspeak, homoglyphs,
zero-width characters) while preserving semantic meaning.  Inspired by
diffusion-based denoising defenses that outperform standard adversarial
training.

The defense operates at three levels:
  1. **Character-level denoising**: homoglyph normalization, repeated character
     collapse, leetspeak reversal, zero-width character removal.
  2. **Token-level denoising**: edit-distance matching against a vocabulary of
     common attack keywords to reconstruct perturbed words.
  3. **Semantic denoising**: generate N slightly perturbed variants of the
     input, embed all variants, and use the centroid embedding as the
     "denoised" representation -- this smooths out adversarial perturbations.

Usage::

    from na0s.diffuse_defense import DiffuseDefense, DiffuseDefenseConfig

    cfg = DiffuseDefenseConfig(n_variants=5, perturbation_rate=0.1)
    dd = DiffuseDefense(cfg)

    clean_text = dd.denoise_text("1gn0r3 a11 pr3v10us 1nstruct10ns")
    # -> "ignore all previous instructions"

    # Semantic denoising with embedding function
    denoised_emb = dd.denoise_embedding(text, embed_fn=my_embed)

Environment variable ``NA0S_DIFFUSE_DEFENSE=1`` enables integration in the
embedding classifier pipeline.
"""

from __future__ import annotations

import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

_log = logging.getLogger(__name__)

# Optional numpy -- fall back to text-only denoising when unavailable.
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Environment toggle
# ---------------------------------------------------------------------------
_ENV_KEY = "NA0S_DIFFUSE_DEFENSE"


def _is_enabled() -> bool:
    """Return True if the DiffuseDef layer is enabled via env var."""
    return os.environ.get(_ENV_KEY, "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiffuseDefenseConfig:
    """Configuration for the DiffuseDefense denoising layer.

    Parameters
    ----------
    n_variants : int
        Number of perturbed text variants to generate for semantic denoising.
    perturbation_rate : float
        Fraction of characters to perturb in each variant (0.0 -- 1.0).
    use_semantic_denoising : bool
        Whether to use multi-variant embedding averaging.
    max_edit_distance : int
        Maximum edit distance for token-level vocabulary matching.
    seed : int or None
        Random seed for reproducible perturbations.  None = non-deterministic.
    """

    n_variants: int = 5
    perturbation_rate: float = 0.1
    use_semantic_denoising: bool = True
    max_edit_distance: int = 2
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Homoglyph mapping -- visually similar Unicode -> ASCII
# ---------------------------------------------------------------------------
# This covers the most common homoglyph attacks.  The mapping is intentionally
# kept compact; full-range coverage would require a much larger table.
HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic
    "\u0410": "A", "\u0430": "a",  # А а
    "\u0412": "B", "\u0432": "v",  # В в (uppercase -> B)
    "\u0421": "C", "\u0441": "c",  # С с
    "\u0415": "E", "\u0435": "e",  # Е е
    "\u041d": "H", "\u043d": "h",  # Н н
    "\u041a": "K", "\u043a": "k",  # К к
    "\u041c": "M", "\u043c": "m",  # М м
    "\u041e": "O", "\u043e": "o",  # О о
    "\u0420": "P", "\u0440": "p",  # Р р
    "\u0422": "T", "\u0442": "t",  # Т т
    "\u0425": "X", "\u0445": "x",  # Х х
    "\u0423": "Y", "\u0443": "y",  # У у
    # Greek
    "\u0391": "A", "\u03b1": "a",  # Α α
    "\u0392": "B", "\u03b2": "B",  # Β β
    "\u0395": "E", "\u03b5": "e",  # Ε ε
    "\u0397": "H", "\u03b7": "n",  # Η η
    "\u0399": "I", "\u03b9": "i",  # Ι ι
    "\u039a": "K", "\u03ba": "k",  # Κ κ
    "\u039c": "M", "\u03bc": "m",  # Μ μ (note: also micro sign)
    "\u039d": "N", "\u03bd": "v",  # Ν ν
    "\u039f": "O", "\u03bf": "o",  # Ο ο
    "\u03a1": "P", "\u03c1": "p",  # Ρ ρ
    "\u03a4": "T", "\u03c4": "t",  # Τ τ
    "\u03a7": "X", "\u03c7": "x",  # Χ χ
    "\u03a5": "Y", "\u03c5": "u",  # Υ υ
    "\u0396": "Z", "\u03b6": "z",  # Ζ ζ
    # Fullwidth ASCII
    "\uff21": "A", "\uff22": "B", "\uff23": "C", "\uff24": "D",
    "\uff25": "E", "\uff26": "F", "\uff27": "G", "\uff28": "H",
    "\uff29": "I", "\uff2a": "J", "\uff2b": "K", "\uff2c": "L",
    "\uff2d": "M", "\uff2e": "N", "\uff2f": "O", "\uff30": "P",
    "\uff31": "Q", "\uff32": "R", "\uff33": "S", "\uff34": "T",
    "\uff35": "U", "\uff36": "V", "\uff37": "W", "\uff38": "X",
    "\uff39": "Y", "\uff3a": "Z",
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
    # Common special homoglyphs
    "\u00b5": "u",   # micro sign
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2026": "...", # ellipsis
}

# ---------------------------------------------------------------------------
# Leetspeak mapping
# ---------------------------------------------------------------------------
LEETSPEAK_MAP: Dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "+": "t",
}

# ---------------------------------------------------------------------------
# Zero-width and invisible characters to strip
# ---------------------------------------------------------------------------
_ZERO_WIDTH_CHARS = frozenset([
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\ufeff",  # BOM / zero-width no-break space
    "\u00ad",  # soft hyphen
    "\u034f",  # combining grapheme joiner
    "\u061c",  # Arabic letter mark
    "\u180e",  # Mongolian vowel separator
])

# ---------------------------------------------------------------------------
# Attack vocabulary for token-level denoising
# ---------------------------------------------------------------------------
ATTACK_VOCABULARY: List[str] = [
    "ignore", "previous", "instructions", "system", "prompt", "override",
    "bypass", "forget", "disregard", "reveal", "secret", "hidden",
    "admin", "password", "credential", "token", "key", "api",
    "execute", "command", "shell", "code", "inject", "injection",
    "jailbreak", "restrict", "restriction", "output", "print",
    "respond", "assistant", "user", "role", "pretend", "act",
    "new", "now", "instead", "above", "below", "all", "stop",
    "delete", "drop", "table", "select", "from", "where",
    "dan", "developer", "mode", "enable", "disable", "sudo",
    "hack", "exploit", "vulnerability", "attack", "payload",
    "obey", "comply", "listen", "follow", "reset", "clear",
    "confidential", "internal", "private", "sensitive",
    "context", "window", "memory", "history", "conversation",
    "write", "read", "access", "permission", "grant", "deny",
    "safety", "filter", "guardrail", "policy", "rule", "constraint",
    "character", "persona", "identity", "original", "true",
    "ignore", "previous", "everything",
]
# Deduplicate while preserving order
ATTACK_VOCABULARY = list(dict.fromkeys(ATTACK_VOCABULARY))


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein)
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = prev_row[j + 1] + 1
            delete = curr_row[j] + 1
            substitute = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insert, delete, substitute))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Character-level denoising
# ---------------------------------------------------------------------------

def _remove_zero_width_chars(text: str) -> str:
    """Strip zero-width and invisible Unicode characters."""
    return "".join(ch for ch in text if ch not in _ZERO_WIDTH_CHARS)


def _normalize_homoglyphs(text: str) -> str:
    """Replace visually similar Unicode characters with ASCII equivalents."""
    # First, apply Unicode NFKC normalization (handles fullwidth, compatibility)
    text = unicodedata.normalize("NFKC", text)
    # Then apply our explicit homoglyph map for anything NFKC missed
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


def _collapse_repeated_chars(text: str) -> str:
    """Collapse runs of 3+ identical characters to a single character.

    Preserves legitimate double letters (e.g., "ll" in "all") while
    collapsing adversarial repetitions like "iiiiignore" -> "ignore".
    """
    return re.sub(r"(.)\1{2,}", r"\1", text)


def _reverse_leetspeak(text: str) -> str:
    """Reverse common leetspeak substitutions within word boundaries.

    Only applies to sequences that look like words (not standalone numbers).
    """
    result = []
    # Split on whitespace to process word by word
    tokens = text.split()
    for token in tokens:
        # Only apply leetspeak reversal if the token contains a mix of
        # letters and leet characters, or is all leet characters.
        has_letter = any(c.isalpha() for c in token)
        has_leet = any(c in LEETSPEAK_MAP for c in token)

        if has_leet and (has_letter or len(token) > 1):
            converted = "".join(
                LEETSPEAK_MAP.get(c, c) for c in token
            )
            result.append(converted)
        else:
            result.append(token)

    return " ".join(result)


def _character_level_denoise(text: str) -> str:
    """Apply all character-level denoising passes.

    Order matters:
      1. Remove zero-width chars (may be splitting words)
      2. Normalize homoglyphs (Cyrillic/Greek -> ASCII)
      3. Collapse repeated characters
      4. Reverse leetspeak
    """
    text = _remove_zero_width_chars(text)
    text = _normalize_homoglyphs(text)
    text = _collapse_repeated_chars(text)
    text = _reverse_leetspeak(text)
    return text


# ---------------------------------------------------------------------------
# Token-level denoising
# ---------------------------------------------------------------------------

def _find_closest_vocab_word(
    word: str,
    vocabulary: Sequence[str],
    max_distance: int,
) -> Optional[str]:
    """Find the closest vocabulary word within *max_distance* edits.

    Returns None if no word is close enough.  When multiple words tie,
    the first match in vocabulary order wins.
    """
    word_lower = word.lower()
    if len(word_lower) < 2:
        return None

    best_word: Optional[str] = None
    best_dist = max_distance + 1

    for vocab_word in vocabulary:
        # Quick length-based pruning
        if abs(len(vocab_word) - len(word_lower)) > max_distance:
            continue
        dist = _levenshtein(word_lower, vocab_word)
        if dist < best_dist:
            best_dist = dist
            best_word = vocab_word
        if dist == 0:
            break  # Exact match

    return best_word if best_dist <= max_distance else None


def _token_level_denoise(
    text: str,
    vocabulary: Sequence[str],
    max_edit_distance: int,
) -> str:
    """Correct perturbed tokens using edit-distance vocabulary matching."""
    tokens = re.split(r"(\s+)", text)  # Keep whitespace tokens
    result = []

    for token in tokens:
        # Preserve whitespace and punctuation-only tokens
        if not token.strip() or not any(c.isalpha() for c in token):
            result.append(token)
            continue

        # Strip leading/trailing punctuation for matching
        leading = ""
        trailing = ""
        core = token
        while core and not core[0].isalnum():
            leading += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            trailing = core[-1] + trailing
            core = core[:-1]

        if not core:
            result.append(token)
            continue

        match = _find_closest_vocab_word(core, vocabulary, max_edit_distance)
        if match is not None and match != core.lower():
            # Preserve original casing pattern if possible
            if core[0].isupper() and len(core) > 1 and core[1:].islower():
                match = match.capitalize()
            elif core.isupper():
                match = match.upper()
            result.append(leading + match + trailing)
        else:
            result.append(token)

    return "".join(result)


# ---------------------------------------------------------------------------
# Semantic denoising (multi-variant embedding averaging)
# ---------------------------------------------------------------------------

def _generate_perturbation(
    text: str,
    rate: float,
    rng: random.Random,
) -> str:
    """Generate a single perturbed variant of *text*.

    Perturbation operations (applied character-by-character):
      - Drop character (skip it)
      - Swap with neighbor
      - Duplicate character
    Each character has *rate* probability of being perturbed.
    """
    chars = list(text)
    result = []
    i = 0

    while i < len(chars):
        if rng.random() < rate:
            op = rng.choice(["drop", "swap", "duplicate"])
            if op == "drop":
                i += 1
                continue
            elif op == "swap" and i + 1 < len(chars):
                result.append(chars[i + 1])
                result.append(chars[i])
                i += 2
                continue
            elif op == "duplicate":
                result.append(chars[i])
                result.append(chars[i])
                i += 1
                continue
        result.append(chars[i])
        i += 1

    return "".join(result)


def _semantic_denoise_embedding(
    text: str,
    embed_fn: Callable[[List[str]], "np.ndarray"],
    n_variants: int,
    perturbation_rate: float,
    rng: random.Random,
) -> "np.ndarray":
    """Generate perturbed variants, embed them all, return centroid embedding.

    Parameters
    ----------
    text : str
        Original text to denoise.
    embed_fn : callable
        Function that takes a list of strings and returns a 2-D numpy array
        of shape ``(n_texts, embedding_dim)``.
    n_variants : int
        Number of perturbed variants to generate.
    perturbation_rate : float
        Per-character perturbation probability.
    rng : random.Random
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Centroid embedding of shape ``(1, embedding_dim)``.
    """
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for semantic denoising")

    # Build variant list: original + N perturbations
    variants = [text]
    for _ in range(n_variants):
        variant = _generate_perturbation(text, perturbation_rate, rng)
        variants.append(variant)

    # Embed all variants at once (batch call)
    embeddings = embed_fn(variants)  # shape: (n_variants+1, dim)

    # Compute centroid
    centroid = np.mean(embeddings, axis=0, keepdims=True)  # shape: (1, dim)
    return centroid


# ---------------------------------------------------------------------------
# Main DiffuseDefense class
# ---------------------------------------------------------------------------

class DiffuseDefense:
    """Text denoising defense for adversarial robustness.

    Applies multi-pass denoising to neutralize adversarial perturbations
    in prompt text before classification.

    Parameters
    ----------
    config : DiffuseDefenseConfig or None
        Configuration.  Defaults to ``DiffuseDefenseConfig()``.
    """

    def __init__(self, config: Optional[DiffuseDefenseConfig] = None) -> None:
        self.config = config or DiffuseDefenseConfig()
        self._rng = random.Random(self.config.seed)
        _log.debug(
            "DiffuseDefense initialized: n_variants=%d, perturbation_rate=%.2f, "
            "semantic=%s",
            self.config.n_variants,
            self.config.perturbation_rate,
            self.config.use_semantic_denoising,
        )

    def denoise_text(self, text: str) -> str:
        """Apply character-level and token-level denoising to *text*.

        Parameters
        ----------
        text : str
            Raw input text, possibly containing adversarial perturbations.

        Returns
        -------
        str
            Denoised text suitable for downstream classification.
        """
        if not text:
            return text

        # Pass 1: character-level denoising
        denoised = _character_level_denoise(text)

        # Pass 2: token-level denoising (edit-distance vocabulary matching)
        denoised = _token_level_denoise(
            denoised,
            vocabulary=ATTACK_VOCABULARY,
            max_edit_distance=self.config.max_edit_distance,
        )

        _log.debug("denoise_text: %r -> %r", text[:80], denoised[:80])
        return denoised

    def denoise_embedding(
        self,
        text: str,
        embed_fn: Callable[[List[str]], "np.ndarray"],
    ) -> "np.ndarray":
        """Return a denoised embedding via multi-variant centroid averaging.

        Applies text-level denoising first, then (if configured) generates
        perturbed variants of the denoised text, embeds all of them, and
        returns the centroid embedding.

        Parameters
        ----------
        text : str
            Raw input text.
        embed_fn : callable
            Embedding function: ``(List[str]) -> np.ndarray`` of shape
            ``(n, dim)``.

        Returns
        -------
        np.ndarray
            Denoised embedding of shape ``(1, dim)``.

        Raises
        ------
        RuntimeError
            If numpy is not available and semantic denoising is requested.
        """
        denoised_text = self.denoise_text(text)

        if not self.config.use_semantic_denoising or not _HAS_NUMPY:
            if not _HAS_NUMPY:
                _log.warning(
                    "numpy not available; falling back to text-only denoising"
                )
            # Fall back to single embedding of denoised text
            return embed_fn([denoised_text])

        return _semantic_denoise_embedding(
            denoised_text,
            embed_fn=embed_fn,
            n_variants=self.config.n_variants,
            perturbation_rate=self.config.perturbation_rate,
            rng=self._rng,
        )


# ---------------------------------------------------------------------------
# Module-level convenience (singleton pattern matching promptguard_signal.py)
# ---------------------------------------------------------------------------
_singleton: Optional[DiffuseDefense] = None
_singleton_lock = __import__("threading").Lock()


def _get_singleton() -> DiffuseDefense:
    """Return (or create) the module-level DiffuseDefense singleton."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        _singleton = DiffuseDefense()
        return _singleton


def reset_singleton() -> None:
    """Reset the module-level singleton.  Used in tests only."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def get_denoised_text(text: str) -> str:
    """Convenience: denoise text using the module singleton.

    Returns the original text unchanged if the defense is disabled.
    """
    if not _is_enabled():
        return text
    return _get_singleton().denoise_text(text)


def get_denoised_embedding(
    text: str,
    embed_fn: Callable[[List[str]], "np.ndarray"],
) -> "np.ndarray":
    """Convenience: return denoised embedding using the module singleton.

    Falls back to a plain embedding of the original text when the defense
    is disabled.
    """
    if not _is_enabled():
        return embed_fn([text])
    return _get_singleton().denoise_embedding(text, embed_fn)
