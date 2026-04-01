"""Load multi-turn test scenarios from JSON fixture files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TestScenario:
    """A single multi-turn test scenario loaded from a fixture file."""

    __test__ = False  # prevent pytest collection

    name: str
    description: str
    turns: List[Dict[str, Any]]  # [{"text": ..., "risk_score": float, "label": str}]
    expected_detection: bool
    expected_alert_type: str = ""
    expected_min_confidence: float = 0.0


def load_scenarios(fixture_path: Path) -> List[TestScenario]:
    """Load test scenarios from a JSON file.

    Args:
        fixture_path: Path to a JSON file containing a list of scenario dicts.

    Returns:
        List of TestScenario instances.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    scenarios = []
    for entry in data:
        scenarios.append(
            TestScenario(
                name=entry["name"],
                description=entry.get("description", ""),
                turns=entry["turns"],
                expected_detection=entry["expected_detection"],
                expected_alert_type=entry.get("expected_alert_type", ""),
                expected_min_confidence=entry.get("expected_min_confidence", 0.0),
            )
        )
    return scenarios
