"""Unified scan result — clean, machine-readable output from the detector."""

import dataclasses
import json
from dataclasses import dataclass, field


def _clamp_unit(value) -> float:
    """Clamp a score to the public [0.0, 1.0] contract.

    Casts to a plain float (normalizing numpy scalars), maps NaN/inf to 0.0,
    and bounds to [0, 1].  Used to enforce the risk-score invariant at the
    output boundary (GAP-07).
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):  # NaN / +-inf
        return 0.0
    return min(1.0, max(0.0, v))


@dataclass
class ScanResult:
    sanitized_text: str = ""
    is_malicious: bool = False
    risk_score: float = 0.0
    label: str = "safe"            # "safe", "malicious", "blocked"
    technique_tags: list = field(default_factory=list)
    rule_hits: list = field(default_factory=list)
    ml_confidence: float = 0.0
    ml_label: str = ""             # what the model alone predicted
    anomaly_flags: list = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str = ""
    cascade_stage: str = ""        # which cascade stage decided: "whitelist",
                                   # "weighted", "embedding", "judge",
                                   # "positive_validation", "blocked", or ""
                                   # (empty when produced by predict.scan())
    embedding_score: float = 0.0   # Layer 5: centroid-based semantic similarity
                                   # score in [0.0, 0.20].  0.0 when the
                                   # embedding classifier is not available.
    embedding_available: bool = True  # Layer 5: False when the embedding signal
                                   # was degraded for this scan (env-disabled or
                                   # running on a fallback backend).  Defaults to
                                   # True (observability-safe) so callers /
                                   # telemetry can see when detection ran without
                                   # the live semantic model.
    model_version: str = ""        # Layer 4: first 8 chars of model.pkl SHA-256
    perplexity_score: float = 0.0  # Layer 4: pseudo-perplexity score [0.0, 1.0]
    judge_reasoning: str = ""      # Layer 7: LLM judge reasoning (CoT or summary)
    output_scan_flags: list = field(default_factory=list)   # Layer 9: output scan flags
    output_scan_risk: float = 0.0                           # Layer 9: output scan risk score
    canary_triggered: bool = False                          # Layer 10: canary token triggered
    canary_leaks: list = field(default_factory=list)        # Layer 10: canary leak details
    elapsed_ms: float = 0.0

    # Layer 16: Multi-turn detection  # LAYER16
    multi_turn_alerts: list = field(default_factory=list)
    multi_turn_risk_trend: list = field(default_factory=list)
    escalation_detected: bool = False
    session_id: str = ""
    multi_turn_threat_level: str = ""   # "" | normal | watch | suspect | flagged | blocked
    multi_turn_recommendation: str = ""  # "" | continue_monitoring | flag | block
    cumulative_risk: float = 0.0         # EMA session risk that drove the verdict

    # GAP-12: low-margin / signal-disagreement abstain band.  `abstained` marks a
    # borderline verdict (risk near the threshold, or the detectors disagree) that
    # the embedding application SHOULD escalate (human review / LLM judge) rather
    # than trust the coin-flip.  `uncertainty` in [0,1] quantifies how borderline.
    abstained: bool = False
    uncertainty: float = 0.0

    def __post_init__(self):
        # GAP-07: enforce the public [0,1] risk-score contract at the single
        # output boundary.  Internal scoring can transiently go negative (e.g.
        # safe-content deductions subtract without a lower clamp) and a raw
        # np.float64 can leak through, so clamp + NaN/inf-guard + normalize to a
        # plain float here.  This covers EVERY current and future ScanResult
        # construction site (predict.scan(), cascade, ensemble, ...) in one
        # place, and also stops the Layer-16 fold from receiving an
        # out-of-range score that add_turn() would reject.
        self.risk_score = _clamp_unit(self.risk_score)
        self.cumulative_risk = _clamp_unit(self.cumulative_risk)
        self.uncertainty = _clamp_unit(self.uncertainty)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)
