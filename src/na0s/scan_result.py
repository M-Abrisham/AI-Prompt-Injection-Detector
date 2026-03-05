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
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)
