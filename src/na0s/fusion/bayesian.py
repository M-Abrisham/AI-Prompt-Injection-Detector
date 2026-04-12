"""Bayesian decision fusion for combining heterogeneous detection signals.

Instead of a linear weighted sum, this module treats each detection signal
(ML confidence, rule hits, obfuscation flags) as evidence and updates a
posterior probability of injection using Bayes' rule.

Enabled via the ``NA0S_BAYESIAN_FUSION=1`` environment variable as an
alternative to the default linear voting in ``_voting.py``.
"""

from __future__ import annotations

import threading

# Default likelihood ratios for common signal types.
DEFAULT_LIKELIHOOD_RATIOS: dict[str, float] = {
    "ml_high_confidence": 10.0,
    "ml_medium_confidence": 4.0,
    "rule_critical": 8.0,
    "rule_high": 6.0,
    "rule_medium": 3.0,
    "rule_low": 1.5,
    "obfuscation": 5.0,
    "structural": 3.0,
    "embedding": 4.0,
}


class BayesianFusion:
    """Bayesian evidence fusion for prompt injection detection.

    Maintains a running posterior probability of the input being malicious,
    updated incrementally as each detection signal is observed.

    Parameters
    ----------
    prior : float
        Prior probability P(malicious) before any evidence (default 0.1).
    """

    def __init__(self, prior: float = 0.1):
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1), got {}".format(prior))
        self._prior = prior
        self._posterior = prior
        self._evidence: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    # -- Evidence accumulation -----------------------------------------------

    def update(self, signal_name: str, likelihood_ratio: float) -> None:
        """Update the posterior with a new piece of evidence.

        Parameters
        ----------
        signal_name : str
            Human-readable label for the signal (e.g. ``"ml_high_confidence"``).
        likelihood_ratio : float
            Ratio P(signal | malicious) / P(signal | benign).
            Values > 1 increase P(malicious); values < 1 decrease it.
        """
        if likelihood_ratio <= 0:
            raise ValueError("likelihood_ratio must be positive, got {}".format(
                likelihood_ratio,
            ))
        with self._lock:
            # Bayes update:  posterior_odds = prior_odds * LR
            # Clamp posterior away from 0 and 1 to avoid division by zero.
            p = max(min(self._posterior, 1.0 - 1e-15), 1e-15)
            odds = (p / (1.0 - p)) * likelihood_ratio
            self._posterior = odds / (1.0 + odds)
            self._evidence.append((signal_name, likelihood_ratio))

    # -- Querying ------------------------------------------------------------

    def get_posterior(self) -> float:
        """Return current P(malicious | all evidence so far)."""
        with self._lock:
            return self._posterior

    def decide(self, threshold: float = 0.55) -> tuple[str, float]:
        """Make a label decision based on current posterior.

        Returns
        -------
        tuple[str, float]
            ``(label, confidence)`` where label is ``"MALICIOUS"`` or ``"SAFE"``
            and confidence is P(label correct).
        """
        with self._lock:
            p = self._posterior
        if p >= threshold:
            return "MALICIOUS", p
        return "SAFE", 1.0 - p

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Reset posterior to the prior and clear evidence history."""
        with self._lock:
            self._posterior = self._prior
            self._evidence.clear()

    @property
    def prior(self) -> float:
        return self._prior

    @property
    def evidence(self) -> list[tuple[str, float]]:
        """Return a copy of the accumulated evidence list."""
        with self._lock:
            return list(self._evidence)
