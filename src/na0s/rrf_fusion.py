"""Layer 6: Reciprocal Rank Fusion (RRF) — scale-invariant signal combination.

RRF converts heterogeneous signal magnitudes into rank-based scores so that
no single signal can dominate purely because of its scale.  This provides a
robust alternative to linear weighted sums when signal distributions are
unknown or non-stationary.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""

from __future__ import annotations


def rrf_score(signals: dict[str, float], k: int = 60) -> float:
    """Compute Reciprocal Rank Fusion score from named signals.

    Each signal value is ranked (highest value = rank 1).  The RRF
    score is the sum of ``1 / (k + rank_i)`` for each signal,
    normalized to the [0, 1] range.

    Parameters
    ----------
    signals : dict[str, float]
        Named signal values (e.g. ``{"ml": 0.8, "rules": 0.3}``).
    k : int
        RRF smoothing constant (default 60, standard value from the
        original paper).

    Returns
    -------
    float
        Normalized RRF score in [0, 1].
    """
    if not signals:
        return 0.0

    n = len(signals)

    # Sort signal names by value descending → rank 1 = highest value
    sorted_names = sorted(signals, key=lambda s: signals[s], reverse=True)

    raw = 0.0
    for rank_0, _name in enumerate(sorted_names):
        rank = rank_0 + 1  # 1-based rank
        raw += 1.0 / (k + rank)

    # Theoretical max: all at rank 1 → n * 1/(k+1)
    # Theoretical min: 0 (no signals)
    # Normalize to [0, 1] by dividing by the max possible sum
    max_raw = n * (1.0 / (k + 1))
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
