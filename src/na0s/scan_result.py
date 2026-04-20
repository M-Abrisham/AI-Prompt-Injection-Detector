"""Unified scan result — clean, machine-readable output from the detector."""

import dataclasses
import json
from dataclasses import dataclass, field


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

    # Confidence band: {"safe", "uncertain", "malicious"} — derived from
    # risk_score in __post_init__. "uncertain" absorbs ~0.05 hardware drift
    # near the threshold boundary. See config.classify_band.
    confidence_band: str = "safe"

    def __post_init__(self):
        from na0s.config import classify_band
        self.confidence_band = classify_band(self.risk_score)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)
