"""Layer 16 multi-turn detectors."""

from .base_detector import MultiTurnDetector
from .context_poisoning import ContextPoisoningDetector
from .escalation import EscalationDetector
from .fabricated_history import FabricatedHistoryDetector
from .payload_splitting import PayloadSplittingDetector
from .stylometry import BehavioralStylometryDetector
from .turn_analyzer import TurnAnalyzer

try:
    from .embedding_drift import EmbeddingDriftDetector
except ImportError:
    EmbeddingDriftDetector = None  # type: ignore[assignment,misc]

__all__ = [
    "MultiTurnDetector",
    "BehavioralStylometryDetector",
    "ContextPoisoningDetector",
    "EmbeddingDriftDetector",
    "EscalationDetector",
    "FabricatedHistoryDetector",
    "PayloadSplittingDetector",
    "TurnAnalyzer",
]
