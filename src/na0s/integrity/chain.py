"""Layer 6: Chain Integrity Tracker — trust score propagation across multi-LLM pipeline stages.

In multi-hop pipelines where LLM output feeds into the next LLM, trust
must decay when injection signals are detected at any stage.  This module
tracks per-stage scan results and maintains a running trust score that
downstream consumers can use to decide whether to escalate scrutiny.
"""

from __future__ import annotations

from na0s.scan_result import ScanResult


class ChainIntegrityTracker:
    """Track trust across a multi-stage LLM pipeline.

    Trust starts at *initial_trust* (default 1.0) and decays whenever a
    pipeline stage produces injection signals.  The decay compounds
    multiplicatively so that multiple suspicious stages erode trust
    quickly.

    Parameters
    ----------
    initial_trust : float
        Starting trust level in [0.0, 1.0].
    decay_rate : float
        Base decay factor applied per suspicious signal category.
    """

    ESCALATION_THRESHOLD = 0.5

    def __init__(self, initial_trust: float = 1.0, decay_rate: float = 0.15):
        if not 0.0 <= initial_trust <= 1.0:
            raise ValueError("initial_trust must be in [0.0, 1.0]")
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay_rate must be in [0.0, 1.0]")
        self._initial_trust = initial_trust
        self._decay_rate = decay_rate
        self._trust: float = initial_trust
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_stage(self, stage_name: str, result: ScanResult) -> None:
        """Record a pipeline stage result and decay trust if warranted.

        Trust decays when any of the following are true:
        * ``len(result.rule_hits) > 0``
        * ``len(result.anomaly_flags) > 0``
        * ``result.risk_score > 0.5``

        Each condition that fires applies one multiplicative decay of
        ``(1 - decay_rate)`` to the current trust score.
        """
        decay_reasons: list[str] = []

        if len(result.rule_hits) > 0:
            decay_reasons.append("rule_hits")
        if len(result.anomaly_flags) > 0:
            decay_reasons.append("anomaly_flags")
        if result.risk_score > 0.5:
            decay_reasons.append("high_risk_score")

        for _reason in decay_reasons:
            self._trust *= (1.0 - self._decay_rate)

        # Clamp to [0.0, 1.0]
        self._trust = max(0.0, min(1.0, self._trust))

        self._history.append({
            "stage_name": stage_name,
            "label": result.label,
            "risk_score": result.risk_score,
            "rule_hits": list(result.rule_hits),
            "anomaly_flags": list(result.anomaly_flags),
            "decay_reasons": decay_reasons,
            "trust_after": round(self._trust, 6),
        })

    def get_trust_score(self) -> float:
        """Return the current trust level in [0.0, 1.0]."""
        return round(self._trust, 6)

    def get_history(self) -> list[dict]:
        """Return the list of recorded stage results with trust scores."""
        return list(self._history)

    def should_escalate(self) -> bool:
        """Return True if trust has fallen below the escalation threshold."""
        return self._trust < self.ESCALATION_THRESHOLD

    def reset(self) -> None:
        """Reset trust and history for a new input."""
        self._trust = self._initial_trust
        self._history = []
