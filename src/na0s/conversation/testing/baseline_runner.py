"""Run all Layer 16 test scenarios and measure detection metrics."""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from na0s.conversation.conversation_monitor import ConversationSecurityMonitor
from na0s.conversation.models import SessionConfig
from na0s.conversation.testing.conversation_harness import ConversationTestHarness
from na0s.conversation.testing.metrics import DetectionMetrics
from na0s.conversation.testing.scenario_loader import TestScenario

# Default fixture directory (src/na0s/conversation/testing -> repo root -> tests/...)
_DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "tests"
    / "conversation"
    / "fixtures"
)


def _normalise_entry(entry: Dict[str, Any]) -> TestScenario:
    """Normalise a single fixture entry into a TestScenario.

    Handles three known fixture formats:
    1. Standard: has ``name`` and ``turns`` keys.
    2. Single-turn (fabricated_history): has ``text`` and ``risk_score`` at
       the top level with no ``turns``.
    3. Rescan format: uses ``id`` instead of ``name`` and turns lack
       ``risk_score``/``label``.
    """
    name = entry.get("name") or entry.get("id", "unnamed")

    if "turns" in entry:
        # Normalise each turn dict to always have risk_score and label
        turns = []
        for t in entry["turns"]:
            turns.append({
                "text": t.get("text", ""),
                "risk_score": t.get("risk_score", 0.1),
                "label": t.get("label", "safe"),
            })
    else:
        # Single-turn format (e.g. fabricated_history_samples.json)
        turns = [{
            "text": entry.get("text", ""),
            "risk_score": entry.get("risk_score", 0.1),
            "label": entry.get("label", "safe"),
        }]

    return TestScenario(
        name=name,
        description=entry.get("description", ""),
        turns=turns,
        expected_detection=entry["expected_detection"],
        expected_alert_type=entry.get("expected_alert_type", ""),
        expected_min_confidence=entry.get("expected_min_confidence", 0.0),
    )


def _load_fixture(path: Path) -> List[TestScenario]:
    """Load and normalise all scenarios from a single JSON fixture file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_normalise_entry(e) for e in data]


class BaselineRunner:
    """Runs all Layer 16 test scenarios and records detection metrics."""

    def __init__(
        self,
        fixture_dir: Optional[Path] = None,
        monitor: Optional[ConversationSecurityMonitor] = None,
    ) -> None:
        self._fixture_dir = Path(fixture_dir) if fixture_dir else _DEFAULT_FIXTURES
        self._monitor = monitor or ConversationSecurityMonitor(config=SessionConfig())
        self._metrics = DetectionMetrics()

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def load_all_scenarios(self) -> List[TestScenario]:
        """Load every ``*.json`` fixture from the fixture directory.

        Returns:
            Flat list of all scenarios from all fixture files.

        Raises:
            FileNotFoundError: If the fixture directory does not exist.
        """
        if not self._fixture_dir.is_dir():
            raise FileNotFoundError(
                f"Fixture directory not found: {self._fixture_dir}"
            )

        fixture_files = sorted(self._fixture_dir.glob("*.json"))
        all_scenarios: List[TestScenario] = []
        for fpath in fixture_files:
            scenarios = _load_fixture(fpath)
            all_scenarios.extend(scenarios)
        return all_scenarios

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_full_suite(self) -> Dict[str, Any]:
        """Run every scenario and return structured results.

        Returns:
            Dict with keys: ``aggregate_metrics``, ``per_scenario_results``,
            ``timing_stats``, ``timestamp``, ``scenario_count``.
        """
        scenarios = self.load_all_scenarios()
        if not scenarios:
            return self._empty_results()

        per_scenario: List[Dict[str, Any]] = []
        all_turn_times: List[float] = []

        for scenario in scenarios:
            result = self._run_scenario(scenario)
            per_scenario.append(result)
            all_turn_times.extend(result["turn_times"])

        # Build aggregate metrics from the DetectionMetrics instance
        aggregate = self._compute_aggregate()
        timing = self._compute_timing_stats(all_turn_times)

        return {
            "aggregate_metrics": aggregate,
            "per_scenario_results": per_scenario,
            "timing_stats": timing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario_count": len(scenarios),
        }

    def _run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run a single scenario through the harness."""
        harness = ConversationTestHarness(monitor=self._monitor)
        turn_times: List[float] = []

        for turn in scenario.turns:
            t0 = time.perf_counter()
            harness.send(
                text=turn["text"],
                risk_score=turn.get("risk_score", 0.1),
                label=turn.get("label", "safe"),
            )
            elapsed = time.perf_counter() - t0
            turn_times.append(elapsed)

        actual_detected = harness.alerts_triggered()
        alerts = harness.all_alerts()

        # Record into metrics accumulator
        self._metrics.record(
            scenario_name=scenario.name,
            expected=scenario.expected_detection,
            actual_detected=actual_detected,
            alerts=alerts,
        )

        return {
            "name": scenario.name,
            "description": scenario.description,
            "expected_detection": scenario.expected_detection,
            "actual_detected": actual_detected,
            "alert_count": len(alerts),
            "alerts": [
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "confidence": a.confidence,
                    "description": a.description,
                }
                for a in alerts
            ],
            "turn_count": len(scenario.turns),
            "turn_times": turn_times,
            "total_time": sum(turn_times),
            "match": scenario.expected_detection == actual_detected,
        }

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_aggregate(self) -> Dict[str, Any]:
        """Compute aggregate detection quality metrics."""
        dr = self._metrics.detection_rate()
        fpr = self._metrics.false_positive_rate()

        precision = self._safe_precision()
        recall = dr  # detection_rate == recall
        f1 = self._safe_f1(precision, recall)

        return {
            "detection_rate": dr,
            "false_positive_rate": fpr,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "report": self._metrics.report(),
        }

    def _safe_precision(self) -> float:
        """Precision = TP / (TP + FP).  Returns 0.0 when undefined."""
        records = self._metrics._records
        tp = sum(1 for r in records if r.expected and r.actual_detected)
        fp = sum(1 for r in records if not r.expected and r.actual_detected)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def _safe_f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def _compute_timing_stats(times: List[float]) -> Dict[str, float]:
        """Compute p50 / p95 / p99 / mean / total from a list of durations."""
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "total": 0.0}
        sorted_times = sorted(times)
        n = len(sorted_times)
        return {
            "p50": sorted_times[int(n * 0.50)],
            "p95": sorted_times[min(int(n * 0.95), n - 1)],
            "p99": sorted_times[min(int(n * 0.99), n - 1)],
            "mean": statistics.mean(sorted_times),
            "total": sum(sorted_times),
        }

    @staticmethod
    def _empty_results() -> Dict[str, Any]:
        return {
            "aggregate_metrics": {
                "detection_rate": 0.0,
                "false_positive_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "report": "No scenarios recorded.",
            },
            "per_scenario_results": [],
            "timing_stats": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "total": 0.0},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario_count": 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_baseline(self, results: Dict[str, Any], path: Path) -> None:
        """Save results as JSON for future comparison.

        Creates parent directories if they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strip non-serialisable fields (turn_times lists are fine as floats)
        path.write_text(
            json.dumps(results, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def load_baseline(path: Path) -> Dict[str, Any]:
        """Load a previously saved baseline from JSON."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(self, results: Dict[str, Any]) -> str:
        """Return (and print) a human-readable summary table."""
        agg = results["aggregate_metrics"]
        timing = results["timing_stats"]
        lines = [
            "=" * 60,
            "Layer 16 Detection Baseline",
            "=" * 60,
            f"Scenarios:          {results['scenario_count']}",
            f"Detection rate:     {agg['detection_rate']:.2%}",
            f"False positive rate: {agg['false_positive_rate']:.2%}",
            f"Precision:          {agg['precision']:.2%}",
            f"Recall:             {agg['recall']:.2%}",
            f"F1 score:           {agg['f1_score']:.2%}",
            "",
            "Timing",
            f"  p50:  {timing['p50']*1000:.2f} ms",
            f"  p95:  {timing['p95']*1000:.2f} ms",
            f"  p99:  {timing['p99']*1000:.2f} ms",
            f"  mean: {timing['mean']*1000:.2f} ms",
            f"  total: {timing['total']:.3f} s",
            "",
        ]

        # Mismatches
        mismatches = [s for s in results["per_scenario_results"] if not s["match"]]
        if mismatches:
            lines.append("Mismatches:")
            for m in mismatches:
                tag = "FN" if m["expected_detection"] else "FP"
                lines.append(f"  [{tag}] {m['name']}")
        else:
            lines.append("All scenarios matched expected outcomes.")

        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text
