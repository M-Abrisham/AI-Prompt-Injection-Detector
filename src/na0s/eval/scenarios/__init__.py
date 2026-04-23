"""F14 scenario-based promotion gate — scenario library.

Scenarios are realistic attack/benign test cases that candidate models
must handle correctly before promotion. A scenario can be a single
prompt (``type: single_prompt``) or a multi-turn conversation
(``type: multi_turn``). Each scenario carries an expected verdict and
is scored per-category / per-severity / per-customer-archetype by the
F14 promotion gate.

**Public surface:**

- :class:`Scenario` — dataclass representing one scenario
- :class:`ScenarioTurn` — one turn within a multi-turn scenario
- :class:`ScenarioEvaluator` — how to score a scenario's outcome
- :class:`ScenarioLoader` — reads YAML files from ``data/eval/scenarios/``

**Cross-references:**

- A3 schema fields (``stable_id``, ``paired_benign_id``,
  ``compliance_tags``) from ``na0s.dataset.schema`` are reused here
- F14 ROADMAP entry (Layer 13)
- B1 sandboxed playground (scenarios run inside the sandbox)
- M3 secret-canary randomizer (private scenarios pattern)
"""

from __future__ import annotations

from .schema import (
    Scenario,
    ScenarioEvaluator,
    ScenarioTurn,
    ScenarioType,
    EvaluatorType,
)
from .loader import ScenarioLoader, load_scenarios_dir

__all__ = [
    "Scenario",
    "ScenarioTurn",
    "ScenarioEvaluator",
    "ScenarioType",
    "EvaluatorType",
    "ScenarioLoader",
    "load_scenarios_dir",
]
