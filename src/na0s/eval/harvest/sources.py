"""L15-sync / AIID reader — Layer-15 snapshot -> DRAFT F14 scenarios (offline).

This is the harvest wiring that completes the ``Layer 15 -> drafts`` path: it
reads a Layer-15 threat-intel *snapshot* off disk (the JSON a
:class:`~na0s.threat_intel.base.ThreatIntelSource` persists via ``save_snapshot``),
runs its ``techniques`` through the deterministic template path of
:class:`~na0s.threat_intel.incident_to_sample.IncidentToSamplePipeline`, and hands the
resulting real attack strings to :class:`~na0s.eval.harvest.extractor.IntelScenarioExtractor`
to produce provenance-traced, taxonomy-validated DRAFT scenarios.

Design guarantees (inherited from the extractor — this module only wires, it
never relaxes them):

1. **Offline — no network, no external LLM.** The pipeline is always built with
   ``llm_client=None``, forcing the deterministic ``_match_templates`` path. The
   template path is keyword-matching only; it would itself be an injection
   surface to route untrusted intel text through an external model, which is a
   thing this project defends against.
2. **Never fabricate a payload from a description.** A technique whose
   description matches no template yields no samples, hence no scenarios — a
   bare description is never promoted to an attack payload. Samples the
   extractor cannot place (empty text / unknown taxonomy code) are SKIPPED and
   surfaced in the :class:`~na0s.eval.harvest.extractor.HarvestReport`, never
   silently dropped.
3. **Nothing is auto-promoted.** The output is DRAFT scenarios for human review.

NOTE: imports use the canonical ``na0s.threat_intel.*`` path (the v1.0.0 semantic
name). The old ``na0s.layer15`` remains a deprecated ``sys.modules`` alias shim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from na0s.eval.harvest.extractor import (
    HarvestReport,
    IntelProvenance,
    IntelScenarioExtractor,
)
from na0s.threat_intel.base import SchemaValidationError, SourceSnapshot
from na0s.threat_intel.incident_to_sample import IncidentToSamplePipeline


def snapshot_to_scenarios(
    snapshot_path: Path,
    provenance: IntelProvenance,
    *,
    extractor: Optional[IntelScenarioExtractor] = None,
    pipeline: Optional[IncidentToSamplePipeline] = None,
) -> HarvestReport:
    """Read a Layer-15 snapshot from disk and draft F14 scenarios from it.

    The full offline path::

        json.loads(snapshot_path) -> SourceSnapshot.from_dict
            -> .techniques
            -> IncidentToSamplePipeline(llm_client=None).generate(techniques)
            -> IntelScenarioExtractor().scenarios_from_generated_samples(...)

    Parameters
    ----------
    snapshot_path : Path
        Path to a ``{source}_snapshot.json`` file, as written by
        :meth:`~na0s.threat_intel.base.ThreatIntelSource.save_snapshot`. The file is
        deserialized via :meth:`~na0s.threat_intel.base.SourceSnapshot.from_dict`.
    provenance : IntelProvenance
        Origin / retrieval metadata folded into every emitted scenario's
        description and used as the draft filename stem.
    extractor : IntelScenarioExtractor | None
        Override the extractor (e.g. with a custom :class:`TaxonomyValidator`).
        Defaults to a fresh :class:`IntelScenarioExtractor`.
    pipeline : IncidentToSamplePipeline | None
        Override the incident-to-sample pipeline. Defaults to a template-only
        ``IncidentToSamplePipeline(llm_client=None)`` — the OFFLINE path. A
        provided pipeline is used as-is; callers are responsible for keeping it
        offline.

    Returns
    -------
    HarvestReport
        Emitted scenarios + skipped inputs with reasons. An empty snapshot (no
        techniques) yields an empty report, not an error. Skips are always
        accounted for by the extractor — nothing is silently dropped.

    Raises
    ------
    FileNotFoundError
        If ``snapshot_path`` does not exist.
    ValueError
        If the file is not valid JSON, or does not match the snapshot schema
        (re-raised from :class:`~na0s.threat_intel.base.SchemaValidationError`).
    """
    path = Path(snapshot_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Layer-15 snapshot not found: {path}. Expected a "
            f"'{{source}}_snapshot.json' file written by a ThreatIntelSource."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Layer-15 snapshot {path} is not valid JSON: {exc}"
        ) from exc

    try:
        snapshot = SourceSnapshot.from_dict(data)
    except SchemaValidationError as exc:
        raise ValueError(
            f"Layer-15 snapshot {path} does not match the snapshot schema: {exc}"
        ) from exc

    techniques = snapshot.techniques
    if not techniques:
        # Empty snapshot is a valid, expected state (a first sync with no data,
        # or a source with no techniques) — return an empty report, never crash.
        return HarvestReport()

    pipe = pipeline or IncidentToSamplePipeline(llm_client=None)
    samples = pipe.generate(techniques)

    intel_extractor = extractor or IntelScenarioExtractor()
    return intel_extractor.scenarios_from_generated_samples(samples, provenance)
