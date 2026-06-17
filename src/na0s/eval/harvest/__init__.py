"""Deterministic intel -> DRAFT F14 scenario extraction.

Turns REAL attack strings synced from threat intel into VALIDATED,
provenance-traced DRAFT eval scenarios written to
``data/eval/scenarios/_drafts/`` for human review. It NEVER auto-promotes and
NEVER fabricates attack payloads from technique descriptions.

This is the deterministic, offline backbone of the ``threat-intel-harvester``
capability — no network, no external LLM (that would itself be an injection
surface this project defends against).

Public surface
--------------
- :class:`TaxonomyValidator` — validate ``attack_category`` against the taxonomy.
- :class:`IntelScenarioExtractor` — build + write DRAFT scenarios.
- :class:`IntelProvenance` — origin/retrieval metadata folded into descriptions.
- :class:`HarvestReport` / :class:`SkippedInput` — emitted + skipped accounting.
"""

from __future__ import annotations

from na0s.eval.harvest.taxonomy import TaxonomyValidator
from na0s.eval.harvest.extractor import (
    DEFAULT_DRAFTS_DIR,
    HarvestReport,
    IntelProvenance,
    IntelScenarioExtractor,
    SkippedInput,
)

__all__ = [
    "TaxonomyValidator",
    "IntelScenarioExtractor",
    "IntelProvenance",
    "HarvestReport",
    "SkippedInput",
    "DEFAULT_DRAFTS_DIR",
]
