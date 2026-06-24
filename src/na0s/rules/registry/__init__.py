"""Per-track rule registries (v1.0.0 Step 10 — semantic rule organization).

Each module here owns one attack-track's rule pack, aggregated into the global
``RULES`` list by :mod:`na0s.rules.rules_registry`.  Top-level scattered rule
extensions migrate here so all signature rule sets live under one package.

Residents:
  compliance_evasion -- C1 (moved from top-level ``na0s.compliance_evasion_rules``)
  privacy_probe      -- P (privacy / data leakage) detector + PRIVACY_RULES
"""

from . import privacy_probe  # noqa: F401
