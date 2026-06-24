"""Per-track rule registries.

Each module here owns one attack-track's rule pack (a list of
:class:`na0s.rules.result.Rule`), aggregated into the global ``RULES`` list by
:mod:`na0s.rules.rules_registry`.  Created for v1.0.0 Step 10 (semantic rule
organization); the first resident is ``compliance_evasion`` (C1, moved from the
top-level ``na0s.compliance_evasion_rules`` module — no back-compat shim).
"""
