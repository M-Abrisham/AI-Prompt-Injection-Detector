"""ScenarioLoader — reads YAML scenario files from a directory.

YAML is chosen over JSON because scenarios are human-authored and reviewed
in PRs; YAML's multi-line string support makes multi-turn payloads readable
in diffs. Each ``.yaml`` file contains a list of scenario dicts; a single
file may hold many scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import (
    EvaluatorType,
    Scenario,
    ScenarioEvaluator,
    ScenarioTurn,
    ScenarioType,
)


class ScenarioLoader:
    """Load :class:`Scenario` objects from YAML files.

    Parameters
    ----------
    root : Path
        Directory containing ``*.yaml`` scenario files. Directory is
        walked non-recursively; nested scenario packs should have their
        own loader instances.

    Usage
    -----
    >>> loader = ScenarioLoader(Path("data/eval/scenarios/v0.1"))
    >>> scenarios = loader.load_all()
    >>> assert all(s.stable_id for s in scenarios)
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def load_all(self) -> list[Scenario]:
        """Load every ``*.yaml`` file in ``self.root``.

        Returns
        -------
        list[Scenario]
            Scenarios sorted by ``name`` for deterministic ordering.

        Raises
        ------
        FileNotFoundError
            If ``self.root`` does not exist.
        ValueError
            If any scenario dict is malformed; the error message cites
            the source file + scenario name.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Scenario root not found: {self.root}")

        scenarios: list[Scenario] = []
        for yaml_path in sorted(self.root.glob("*.yaml")):
            scenarios.extend(self._load_file(yaml_path))
        return sorted(scenarios, key=lambda s: s.name)

    def _load_file(self, path: Path) -> Iterable[Scenario]:
        """Parse one YAML file; yield Scenario instances."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError(
                f"Scenario file {path} must contain a list of scenarios, "
                f"got {type(raw).__name__}"
            )

        for idx, entry in enumerate(raw):
            try:
                yield self._scenario_from_dict(entry)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid scenario at {path}[{idx}]: {exc}"
                ) from exc

    @staticmethod
    def _scenario_from_dict(entry: dict[str, Any]) -> Scenario:
        """Build a Scenario from a parsed YAML dict."""
        if not isinstance(entry, dict):
            raise TypeError(f"expected dict, got {type(entry).__name__}")

        scenario_type = ScenarioType(entry["type"])

        # Build turns if MULTI_TURN
        turns: list[ScenarioTurn] = []
        if scenario_type == ScenarioType.MULTI_TURN:
            for turn_idx, turn_data in enumerate(entry.get("turns", [])):
                try:
                    turns.append(
                        ScenarioTurn(
                            text=turn_data["text"],
                            expected_label=turn_data["expected_label"],
                            risk_score=turn_data.get("risk_score"),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"turn[{turn_idx}] missing required field {exc}"
                    ) from exc

        # Build evaluator (defaults apply when missing)
        eval_data = entry.get("evaluator") or {}
        evaluator = ScenarioEvaluator(
            type=EvaluatorType(eval_data.get("type", "deterministic")),
            check=eval_data.get("check", "label == MALICIOUS"),
            threshold=eval_data.get("threshold"),
        )

        return Scenario(
            name=entry["name"],
            type=scenario_type,
            expected_verdict=entry["expected_verdict"],
            severity=entry["severity"],
            attack_category=entry["attack_category"],
            description=entry.get("description", ""),
            payload=entry.get("payload"),
            turns=turns,
            customer_archetype=entry.get("customer_archetype"),
            evaluator=evaluator,
            source=entry.get("source", "manual"),
            tags=list(entry.get("tags", [])),
            difficulty=entry.get("difficulty"),
            compliance_tags=list(entry.get("compliance_tags", [])),
            stable_id=entry.get("stable_id"),  # re-computed if None
            paired_benign_id=entry.get("paired_benign_id"),
        )


def load_scenarios_dir(root: str | Path) -> list[Scenario]:
    """Convenience wrapper: load every scenario from a directory.

    Equivalent to ``ScenarioLoader(root).load_all()``.
    """
    return ScenarioLoader(Path(root)).load_all()
