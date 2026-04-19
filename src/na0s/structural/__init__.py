"""Layer 3 — structural feature extraction.

Computes 29 non-lexical features that capture the STRUCTURE and formatting
of prompts — things TF-IDF misses entirely.  Features span 7 groups:
length, casing, punctuation, structural markers, injection signals,
context, and advanced detection (many-shot, delimiter density, template
markers, language mixing, repetition).

Public API:
    extract_structural_features        -> StructuralFeatures
    extract_structural_features_batch  -> numpy.ndarray
    normalize_features                 -> numpy.ndarray (0..1 scaled)
    StructuralFeatures                 -- dataclass, 29 fields, dict-like
    FEATURE_NAMES                      -- canonical ordering (list of str)
    UNBOUNDED_FEATURE_CAPS             -- per-feature normalisation caps
"""

from __future__ import annotations

from .features import FEATURE_NAMES, StructuralFeatures
from .extractors import (
    extract_structural_features,
    extract_structural_features_batch,
)
from .normalize import UNBOUNDED_FEATURE_CAPS, normalize_features

# Back-compat re-exports for callers that reached into the old
# ``structural_features`` module for private symbols.  Private helpers
# and regex constants stay accessible via the package root so the
# existing test suite and any out-of-tree consumers keep working.
from .patterns import (  # noqa: F401
    _ROLE_PATTERNS,
    _IMPERATIVE_VERBS,
)
from .sentences import _split_sentences  # noqa: F401
from .quotes import _compute_quote_depth  # noqa: F401
from .extractors import (  # noqa: F401
    _compute_entropy,
    _compute_repetition_score,
    _count_script_families,
)

__all__ = [
    "FEATURE_NAMES",
    "StructuralFeatures",
    "UNBOUNDED_FEATURE_CAPS",
    "extract_structural_features",
    "extract_structural_features_batch",
    "normalize_features",
]
