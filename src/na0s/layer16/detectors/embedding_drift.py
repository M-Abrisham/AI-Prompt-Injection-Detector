"""Embedding Drift detector (D1.23) -- sudden semantic drift between turns.

Uses sentence embeddings (all-MiniLM-L6-v2) to compute cosine similarity
between consecutive conversation turns.  A sudden drop in similarity signals
a topic-pivot attack (e.g., benign conversation about weather followed by an
injection attempt).

Two detection signals:
1. Sharp Pivot -- cosine similarity between consecutive turns drops below
   DRIFT_SHARP_THRESHOLD (default 0.3).
2. Sustained Drift -- average cosine similarity over the analysis window
   falls below DRIFT_AVG_THRESHOLD (default 0.5).

Graceful degradation: if sentence-transformers is not installed, analyze()
returns an empty list.  The model is loaded lazily on first analyze() call
and cached as a class-level attribute to avoid repeated loading.

Reference: ZEDD (arXiv 2601.12359), DeepContext (arXiv 2602.16935).
"""

from __future__ import annotations

import math
import threading
from typing import List, Optional

from ..config import (
    DRIFT_AVG_THRESHOLD,
    DRIFT_CONFIDENCE_MIN,
    DRIFT_MIN_TURNS,
    DRIFT_SHARP_THRESHOLD,
    DRIFT_WINDOW,
    ENABLE_EMBEDDING_DRIFT,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# ---------------------------------------------------------------------------
# Optional dependency: sentence-transformers
# ---------------------------------------------------------------------------

_HAS_EMBEDDINGS = False
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _HAS_EMBEDDINGS = True
except ImportError:
    pass

# Sentinel so tests can check / override
_MODEL_NAME = "all-MiniLM-L6-v2"


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors.

    Handles zero-magnitude vectors gracefully (returns 0.0).
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi
    if math.isnan(dot) or math.isnan(norm_a) or math.isnan(norm_b):
        return 0.0
    if math.isinf(dot) or math.isinf(norm_a) or math.isinf(norm_b):
        return 0.0
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


class EmbeddingDriftDetector(MultiTurnDetector):
    """Detects sudden semantic drift between consecutive conversation turns.

    Uses sentence embeddings (MiniLM-L6-v2) to compute cosine similarity
    between turns. A sudden drop in similarity signals a topic-pivot attack
    (e.g., benign conversation about weather -> injection attempt).

    Taxonomy IDs: D1.23 (Semantic Drift Attack)
    """

    # Class-level model cache -- loaded once, shared across instances.
    _model = None
    _model_lock = threading.Lock()

    # ----- MultiTurnDetector interface ----------------------------------------

    @property
    def detector_name(self) -> str:
        return "embedding_drift"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.23"]

    def reset(self) -> None:
        pass  # stateless -- all data comes from ConversationState

    # ----- Embedding helpers --------------------------------------------------

    @classmethod
    def _load_model(cls) -> None:
        """Lazily load the sentence-transformer model (once per process)."""
        if cls._model is None and _HAS_EMBEDDINGS:
            with cls._model_lock:
                if cls._model is None:  # double-check under lock
                    from sentence_transformers import SentenceTransformer

                    cls._model = SentenceTransformer(_MODEL_NAME)

    def _get_embeddings(self, texts: List[str]) -> Optional[list]:
        """Batch-embed *texts*.  Returns list of vectors, or None if unavailable."""
        if not _HAS_EMBEDDINGS:
            return None
        self._load_model()
        if self._model is None:
            return None
        # SentenceTransformer.encode returns np.ndarray; convert rows to lists
        # so downstream code stays numpy-free.
        raw = self._model.encode(texts)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in raw]

    # ----- Main detection logic -----------------------------------------------

    def analyze(self, state: ConversationState) -> List[Alert]:  # noqa: C901
        if not ENABLE_EMBEDDING_DRIFT:
            return []

        if not _HAS_EMBEDDINGS:
            return []

        if state is None or state.is_empty:
            return []

        if len(state.turns) < DRIFT_MIN_TURNS:
            return []

        # Take last DRIFT_WINDOW turns
        window_turns = state.turns[-DRIFT_WINDOW:]
        texts = [t.text for t in window_turns]

        if len(texts) < 2:
            return []

        embeddings = self._get_embeddings(texts)
        if embeddings is None:
            return []

        # Compute consecutive cosine similarities
        similarities: List[float] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Filter out NaN values that could propagate from bad embeddings
        similarities = [s for s in similarities if not math.isnan(s)]

        if not similarities:
            return []

        alerts: List[Alert] = []

        # --- Signal 1: Sharp pivot -------------------------------------------
        min_sim = min(similarities)
        min_idx = similarities.index(min_sim)
        if min_sim < DRIFT_SHARP_THRESHOLD:
            # Confidence: how far below the threshold
            # At threshold -> DRIFT_CONFIDENCE_MIN; at 0.0 -> ~1.0
            raw_conf = 1.0 - (min_sim / DRIFT_SHARP_THRESHOLD)
            confidence = max(DRIFT_CONFIDENCE_MIN, min(1.0, raw_conf))
            confidence = round(confidence, 4)

            # Turn indices relative to full state
            offset = len(state.turns) - len(window_turns)
            alerts.append(
                Alert(
                    alert_type="embedding_drift",
                    severity="high" if confidence >= 0.7 else "medium",
                    confidence=confidence,
                    description=(
                        f"Sharp topic pivot detected: cosine similarity "
                        f"dropped to {min_sim:.3f} between turns "
                        f"{offset + min_idx} and {offset + min_idx + 1}"
                    ),
                    turn_range=(offset + min_idx, offset + min_idx + 1),
                    evidence=[
                        f"min_similarity={min_sim:.4f}",
                        f"threshold={DRIFT_SHARP_THRESHOLD}",
                        f"similarities={[round(s, 4) for s in similarities]}",
                        "signal=sharp_pivot",
                    ],
                )
            )

        # --- Signal 2: Sustained drift ---------------------------------------
        avg_sim = sum(similarities) / len(similarities)
        if avg_sim < DRIFT_AVG_THRESHOLD:
            raw_conf = 1.0 - (avg_sim / DRIFT_AVG_THRESHOLD)
            confidence = max(DRIFT_CONFIDENCE_MIN, min(1.0, raw_conf))
            confidence = round(confidence, 4)

            offset = len(state.turns) - len(window_turns)
            alerts.append(
                Alert(
                    alert_type="embedding_drift",
                    severity="high" if confidence >= 0.7 else "medium",
                    confidence=confidence,
                    description=(
                        f"Sustained semantic drift: average cosine similarity "
                        f"{avg_sim:.3f} over {len(similarities)} consecutive pairs"
                    ),
                    turn_range=(offset, offset + len(window_turns) - 1),
                    evidence=[
                        f"avg_similarity={avg_sim:.4f}",
                        f"threshold={DRIFT_AVG_THRESHOLD}",
                        f"similarities={[round(s, 4) for s in similarities]}",
                        "signal=sustained_drift",
                    ],
                )
            )

        return alerts
