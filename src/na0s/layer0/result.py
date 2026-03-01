from dataclasses import dataclass, field


@dataclass
class Layer0Result:
    sanitized_text: str = ""
    original_length: int = 0
    chars_stripped: int = 0
    anomaly_flags: list = field(default_factory=list)
    token_char_ratio: float = 0.0
    fingerprint: dict = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""
    source_metadata: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
