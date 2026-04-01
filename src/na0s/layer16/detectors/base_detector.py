"""Abstract base class for all Layer 16 multi-turn detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..models import Alert, ConversationState


class MultiTurnDetector(ABC):
    """Base class for multi-turn detection algorithms.

    Every Layer 16 detector inherits from this and implements the three
    required methods.  Detectors must be FAST (<10 ms) -- no ML model
    loading, no network calls.
    """

    @abstractmethod
    def analyze(self, state: ConversationState) -> List[Alert]:
        """Run the detector on the accumulated conversation state.

        Parameters
        ----------
        state : ConversationState
            The full (or windowed) conversation history.

        Returns
        -------
        list[Alert]
            Zero or more alerts produced by this detector.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any internal state so the detector can be reused."""

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Human-readable name for logging and metrics."""

    @property
    @abstractmethod
    def taxonomy_ids(self) -> List[str]:
        """Na0S taxonomy IDs this detector covers."""
