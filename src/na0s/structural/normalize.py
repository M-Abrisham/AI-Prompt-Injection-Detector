"""Soft-cap normalisation for unbounded structural features.

Used when feeding structural features into ML classifiers that expect
scaled inputs.  Binary flags and ratios are left unchanged; unbounded
counts are divided by per-feature soft maximums and clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES


# Features that are unbounded (not naturally in [0, 1]) and benefit from
# normalization when used with ML classifiers expecting scaled inputs.
# The caps are *soft* maximums chosen from empirical analysis of prompt
# injection datasets; values above the cap are clipped to 1.0.
UNBOUNDED_FEATURE_CAPS = {
    "char_count": 5000.0,
    "word_count": 1000.0,
    "avg_word_length": 20.0,
    "exclamation_count": 20.0,
    "question_count": 20.0,
    "consecutive_punctuation": 20.0,
    "line_count": 100.0,
    "title_case_words": 50.0,
    "all_caps_words": 50.0,
    "newline_ratio": 5.0,
    "quote_depth": 10.0,
    "text_entropy": 8.0,    # Shannon entropy of ASCII text maxes at ~6.6
    "many_shot_count": 50.0,
    "delimiter_density": 10.0,
    "template_marker_count": 20.0,
}


def normalize_features(feature_array):
    """Min-max normalize unbounded features to [0, 1] using soft caps.

    Features that are already ratios or binary flags (0/1) are left
    unchanged.  Unbounded features (``char_count``, ``word_count``,
    ``quote_depth``, ``text_entropy``, etc.) are divided by a soft
    maximum and clipped to [0, 1].

    Parameters
    ----------
    feature_array : numpy.ndarray
        Array of shape ``(n, len(FEATURE_NAMES))`` from
        :func:`na0s.structural.extract_structural_features_batch`.

    Returns
    -------
    numpy.ndarray
        Normalized copy of the input array (same shape, float64).

    Notes
    -----
    This is intentionally a *separate* function rather than being built
    into ``extract_structural_features()`` because ``predict.py`` relies
    on raw, un-normalized feature values for threshold-based decisions
    (e.g. ``quote_depth >= 3``, ``text_entropy > 5.0``).
    """
    out = feature_array.copy()
    for feat_name, cap in UNBOUNDED_FEATURE_CAPS.items():
        idx = FEATURE_NAMES.index(feat_name)
        out[:, idx] = np.clip(out[:, idx] / cap, 0.0, 1.0)
    return out
