"""Dataset-registry provenance: per-source taxonomy-code labelling.

The training-data registry ``data/datasets.yaml`` declares every corpus the
ML classifier trains on. Each source is the *provenance* of a slice of
``data/processed/combined_data.csv``; tagging each source with a CANONICAL
``data/taxonomy.yaml`` code lets a downstream consumer answer "which attack
classes does our training corpus actually have provenance for?" without
re-reading 70+ raw CSVs.

A fabricated code in the registry would be an injection vector into the
training corpus exactly like a fabricated ``attack_category`` is into the eval
library (it would silently mis-attribute the provenance of training rows). So
this package REUSES :class:`na0s.eval.harvest.taxonomy.TaxonomyValidator` — the
same ATLAS-anchored gate the harvest pipeline uses — and never invents a code.

Public surface
--------------
- :func:`load_registry` — read ``data/datasets.yaml`` (size-guarded loader).
- :func:`iter_source_codes` — ``(source_name, codes_list)`` for every source.
- :func:`validate_registry_codes` — flag every non-canonical code (the gate).
- :class:`RegistryCodeError` — one ``(source, code, reason)`` finding.
- :func:`serialize_codes` / :func:`parse_codes` — the ONE owner of the
  ``;``-joined CSV-cell format used to carry codes through process_data.
"""

from __future__ import annotations

from na0s.eval.registry.taxonomy_labels import (
    CODE_SEPARATOR,
    RegistryCodeError,
    iter_source_codes,
    load_registry,
    parse_codes,
    serialize_codes,
    validate_registry_codes,
)

__all__ = [
    "CODE_SEPARATOR",
    "RegistryCodeError",
    "iter_source_codes",
    "load_registry",
    "parse_codes",
    "serialize_codes",
    "validate_registry_codes",
]
