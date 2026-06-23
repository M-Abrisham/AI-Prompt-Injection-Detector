"""Per-taxonomy-code coverage of the F14 eval-scenario corpus.

Answers "which of the canonical attack classes does the live eval set actually
have scenario data for, and which are zero-data?" The code universe is derived
AT RUNTIME from :class:`na0s.eval.harvest.taxonomy.TaxonomyValidator` (the
single source of truth) — never hardcoded — so the report stays truthful if the
taxonomy grows or shrinks.

This is the *eval-corpus acquisition* axis: it reads the SCENARIO corpus
(``data/eval/scenarios/v0.1/``), counting ``attack_category`` per code. It is
orthogonal to the registry-provenance axis (``data/datasets.yaml``
``taxonomy_codes``); a forward-compat ``datasets.yaml`` union hook is exposed by
the CLI but is a documented no-op today.

A scenario whose ``attack_category`` is NOT canonical (e.g. the legacy
``E1_benign``) is never silently dropped — it is surfaced in a dedicated
``non_canonical`` section so a future taxonomy drift is visible, not swallowed.

Pure / local / keyless: no network, no LLM. Deterministic given the scenario
corpus + the committed taxonomy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from na0s.eval.harvest.taxonomy import TaxonomyValidator
from na0s.eval.scenarios.loader import load_scenarios_dir

#: Status for a code that has at least one scenario.
STATUS_COVERED = "COVERED"
#: Status for a canonical code with zero scenario data.
STATUS_ZERO_DATA = "ZERO-DATA"


@dataclass(frozen=True)
class CoverageRow:
    """Coverage of one canonical top-level taxonomy code.

    Attributes
    ----------
    code : str
        The canonical top-level code (e.g. ``"D1"``, ``"E"``).
    severity : str
        The taxonomy's severity for the code ("" if the taxonomy records none).
    scenario_count : int
        Total scenarios whose ``attack_category`` equals this code.
    attack_count : int
        Of those, scenarios with ``expected_verdict != "allowed"``.
    benign_count : int
        Of those, scenarios with ``expected_verdict == "allowed"``.
    status : str
        :data:`STATUS_COVERED` or :data:`STATUS_ZERO_DATA`.
    """

    code: str
    severity: str
    scenario_count: int
    attack_count: int
    benign_count: int
    status: str


@dataclass(frozen=True)
class CoverageReport:
    """Per-code coverage of the scenario corpus.

    Attributes
    ----------
    rows : list[CoverageRow]
        One row per canonical TOP-LEVEL code, sorted by code.
    non_canonical : list[tuple[str, int]]
        ``(observed_code, count)`` for every ``attack_category`` seen in the
        corpus that is NOT a canonical code (e.g. ``("E1_benign", 2)``).
        Surfaced, never dropped.
    total_scenarios : int
        Total scenarios loaded across all input dirs.
    """

    rows: list[CoverageRow] = field(default_factory=list)
    non_canonical: list[tuple[str, int]] = field(default_factory=list)
    total_scenarios: int = 0

    @property
    def covered_codes(self) -> list[str]:
        return [r.code for r in self.rows if r.status == STATUS_COVERED]

    @property
    def zero_data_codes(self) -> list[str]:
        return [r.code for r in self.rows if r.status == STATUS_ZERO_DATA]

    @property
    def total_codes(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict:
        """JSON-serializable form (the source-of-truth artifact shape)."""
        return {
            "summary": {
                "total_scenarios": self.total_scenarios,
                "total_codes": self.total_codes,
                "covered": len(self.covered_codes),
                "zero_data": len(self.zero_data_codes),
                "non_canonical_codes": [c for c, _ in self.non_canonical],
            },
            "rows": [
                {
                    "code": r.code,
                    "severity": r.severity,
                    "scenario_count": r.scenario_count,
                    "attack_count": r.attack_count,
                    "benign_count": r.benign_count,
                    "status": r.status,
                }
                for r in self.rows
            ],
            "non_canonical": [
                {"code": code, "count": count}
                for code, count in self.non_canonical
            ],
        }


def _top_level_codes(taxonomy: TaxonomyValidator) -> list[str]:
    """The canonical TOP-LEVEL codes (no dotted technique leaves), sorted.

    Derived at runtime from the validator's known codes — never hardcoded — so
    a taxonomy edit is reflected automatically.
    """
    return sorted(c for c in taxonomy.known_codes() if "." not in c)


def compute_taxonomy_coverage(
    scenario_dirs: list[Path],
    taxonomy: Optional[TaxonomyValidator] = None,
) -> CoverageReport:
    """Compute per-code coverage of the scenario corpus.

    Parameters
    ----------
    scenario_dirs : list[Path]
        Directories of ``*.yaml`` scenario files to union (e.g.
        ``[data/eval/scenarios/v0.1]``).
    taxonomy : TaxonomyValidator | None
        The canonical-code source of truth. Defaults to ``TaxonomyValidator()``.

    Returns
    -------
    CoverageReport
        One row per canonical top-level code (covered or zero-data), plus a
        ``non_canonical`` section for any observed non-canonical category.

    Raises
    ------
    FileNotFoundError
        If a scenario dir does not exist (surfaced by ``load_scenarios_dir``).
    """
    taxonomy = taxonomy or TaxonomyValidator()
    codes = _top_level_codes(taxonomy)
    code_set = set(codes)

    scenarios = []
    for d in scenario_dirs:
        scenarios.extend(load_scenarios_dir(d))

    # Tally per attack_category, split attack vs benign by verdict.
    total = Counter()  # type: Counter[str]
    benign = Counter()  # type: Counter[str]
    for s in scenarios:
        cat = s.attack_category
        total[cat] += 1
        if s.expected_verdict == "allowed":
            benign[cat] += 1

    rows: list[CoverageRow] = []
    for code in codes:
        n = total.get(code, 0)
        b = benign.get(code, 0)
        rows.append(
            CoverageRow(
                code=code,
                severity=taxonomy.get_severity(code) or "",
                scenario_count=n,
                attack_count=n - b,
                benign_count=b,
                status=STATUS_COVERED if n > 0 else STATUS_ZERO_DATA,
            )
        )

    non_canonical = sorted(
        (cat, cnt) for cat, cnt in total.items() if cat not in code_set
    )

    return CoverageReport(
        rows=rows,
        non_canonical=non_canonical,
        total_scenarios=len(scenarios),
    )
