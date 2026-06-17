"""Detection quality metrics for Layer 16 multi-turn testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from na0s.conversation.models import Alert


@dataclass
class _Record:
    scenario_name: str
    expected: bool
    actual_detected: bool
    alerts: List[Alert]


class DetectionMetrics:
    """Accumulates detection results and computes quality metrics."""

    def __init__(self) -> None:
        self._records: List[_Record] = []

    def record(
        self,
        scenario_name: str,
        expected: bool,
        actual_detected: bool,
        alerts: List[Alert],
    ) -> None:
        """Record the outcome of a single scenario.

        Args:
            scenario_name: Human-readable name of the scenario.
            expected: Whether detection was expected (ground truth).
            actual_detected: Whether detection actually occurred.
            alerts: The list of alerts produced.
        """
        self._records.append(
            _Record(
                scenario_name=scenario_name,
                expected=expected,
                actual_detected=actual_detected,
                alerts=alerts,
            )
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def detection_rate(self) -> float:
        """True positive rate: fraction of expected-positive scenarios detected.

        Returns 0.0 if there are no expected-positive scenarios.
        """
        positives = [r for r in self._records if r.expected]
        if not positives:
            return 0.0
        detected = sum(1 for r in positives if r.actual_detected)
        return detected / len(positives)

    def false_positive_rate(self) -> float:
        """False positive rate: fraction of expected-negative scenarios that fired.

        Returns 0.0 if there are no expected-negative scenarios.
        """
        negatives = [r for r in self._records if not r.expected]
        if not negatives:
            return 0.0
        false_pos = sum(1 for r in negatives if r.actual_detected)
        return false_pos / len(negatives)

    def report(self) -> str:
        """Return a human-readable summary of all recorded results."""
        total = len(self._records)
        if total == 0:
            return "No scenarios recorded."

        tp = sum(1 for r in self._records if r.expected and r.actual_detected)
        fn = sum(1 for r in self._records if r.expected and not r.actual_detected)
        fp = sum(1 for r in self._records if not r.expected and r.actual_detected)
        tn = sum(1 for r in self._records if not r.expected and not r.actual_detected)

        lines = [
            f"Detection Metrics ({total} scenarios)",
            f"  True positives:  {tp}",
            f"  False negatives: {fn}",
            f"  False positives: {fp}",
            f"  True negatives:  {tn}",
            f"  Detection rate:       {self.detection_rate():.2%}",
            f"  False positive rate:  {self.false_positive_rate():.2%}",
        ]

        # Per-scenario detail
        mismatches = [
            r for r in self._records if r.expected != r.actual_detected
        ]
        if mismatches:
            lines.append("")
            lines.append("Mismatches:")
            for r in mismatches:
                tag = "FN" if r.expected else "FP"
                lines.append(f"  [{tag}] {r.scenario_name}")

        return "\n".join(lines)
