"""Layer 6: value-weighted reciprocal-rank signal combination.

NOTE (GAP-11): classic Reciprocal Rank Fusion (Cormack et al. 2009) fuses
MULTIPLE ranked lists from independent retrievers.  Applying it to a single
vector of signal *values* — as an alternative to weighted voting — is a
misuse: every input produces the rank set {1..n}, so the pure-rank sum
`Σ 1/(k+rank)` depends ONLY on the number of signals, not their magnitudes.
That made the old `rrf_score` flag MALICIOUS on a handful of trivially-weak
signals (3 signals of value 0.01 scored the same as 0.9).

This implementation therefore WEIGHTS each reciprocal-rank term by the signal's
value, so magnitude drives the score (weak signals -> low score) while a mild
rank tilt keeps any single huge-scale signal from dominating.  It remains an
env-gated, non-default alternative; the canonical combiner is fusion/voting.py.
"""

from __future__ import annotations


def rrf_score(signals: dict[str, float], k: int = 60) -> float:
    """Compute a value-weighted reciprocal-rank score from named signals.

    Signals are ranked by value (highest = rank 1) and each contributes
    ``value_i / (k + rank_i)`` — so MAGNITUDE drives the score while the
    ``1/(k+rank)`` factor gives a mild rank tilt.  Normalized to [0, 1] by the
    max attainable when every signal equals 1.0, so weak signals (all near 0)
    yield a low score and cannot inflate the verdict on count alone (GAP-11).

    Parameters
    ----------
    signals : dict[str, float]
        Named signal values, each expected in [0, 1] (e.g. ``{"ml": 0.8}``).
    k : int
        Reciprocal-rank smoothing constant (default 60).

    Returns
    -------
    float
        Normalized score in [0, 1].
    """
    if not signals:
        return 0.0

    # Sort signal names by value descending → rank 1 = highest value
    sorted_names = sorted(signals, key=lambda s: signals[s], reverse=True)

    raw = 0.0
    max_raw = 0.0
    for rank_0, name in enumerate(sorted_names):
        rank = rank_0 + 1  # 1-based rank
        weight = 1.0 / (k + rank)
        # Clamp each value into [0,1] so an out-of-range signal can't distort.
        value = min(max(float(signals[name]), 0.0), 1.0)
        raw += value * weight        # magnitude-weighted contribution
        max_raw += weight            # max when value == 1.0 at this rank

    if max_raw == 0.0:
        return 0.0

    normalized = raw / max_raw
    return round(min(max(normalized, 0.0), 1.0), 6)


def rrf_decision(
    signals: dict[str, float],
    threshold: float = 0.55,
    k: int = 60,
) -> tuple[str, float]:
    """Compute RRF score and return a classification decision.

    Parameters
    ----------
    signals : dict[str, float]
        Named signal values.
    threshold : float
        Decision threshold above which the verdict is MALICIOUS.
    k : int
        RRF smoothing constant.

    Returns
    -------
    tuple[str, float]
        ``(label, confidence)`` where label is "SAFE" or "MALICIOUS".
    """
    score = rrf_score(signals, k=k)
    if score >= threshold:
        return "MALICIOUS", score
    return "SAFE", score
