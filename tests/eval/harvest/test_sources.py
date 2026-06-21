"""Tests for snapshot_to_scenarios — Layer-15 snapshot -> DRAFT F14 scenarios.

These tests have teeth: each asserts a behavior that breaks if the wiring
breaks. They use the REAL :class:`IncidentToSamplePipeline` template path
(``llm_client=None``) and the REAL :class:`IntelScenarioExtractor` — no network,
no external LLM, nothing mocked.

Invariants under test:

- a snapshot of techniques whose descriptions hit the deterministic template
  keywords produces real scenarios that carry ``source="harvest_pipeline"`` and
  provenance in the description, and round-trip through ``ScenarioLoader`` after
  ``write_drafts``;
- a technique whose description matches NO template yields NO scenario — a bare
  description is never fabricated into an attack payload;
- the extractor's SKIP accounting is surfaced (not silently dropped) when a
  sample cannot be placed;
- an empty snapshot returns an empty :class:`HarvestReport` rather than crashing;
- a missing file raises ``FileNotFoundError`` and malformed JSON raises
  ``ValueError``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from na0s.eval.harvest import (
    HarvestReport,
    IntelProvenance,
    IntelScenarioExtractor,
    snapshot_to_scenarios,
)
from na0s.eval.scenarios import ScenarioLoader
from na0s.threat_intel.base import SourceSnapshot, TechniqueEntry
from na0s.threat_intel.incident_to_sample import GeneratedSample, IncidentToSamplePipeline


@pytest.fixture
def provenance() -> IntelProvenance:
    return IntelProvenance(
        source_slug="l15-unit-source",
        origin="https://example.test/threat-intel/snapshot",
        retrieved="2026-06-17",
    )


def _write_snapshot(path: Path, techniques: list[TechniqueEntry]) -> Path:
    """Write a SourceSnapshot to ``path`` as the on-disk JSON the reader expects."""
    snapshot = SourceSnapshot(
        source_name="unit-source",
        fetched_at=datetime(2026, 6, 17),
        version="v1",
        techniques=techniques,
    )
    path.write_text(json.dumps(snapshot.to_dict()), encoding="utf-8")
    return path


def test_snapshot_with_matching_techniques_emits_and_roundtrips(
    provenance: IntelProvenance, tmp_path: Path
):
    """Techniques that hit template keywords -> real scenarios that round-trip.

    Descriptions are crafted to hit the deterministic template keyword groups
    (override/ignore -> D1, exfiltrate/extract/data -> E, persona/roleplay -> D2),
    so the offline pipeline emits real attack strings.
    """
    techniques = [
        TechniqueEntry(
            id="T-override",
            name="Instruction override",
            description=(
                "Attackers ignore and override the system instructions to "
                "bypass guardrails and jailbreak the assistant."
            ),
        ),
        TechniqueEntry(
            id="T-exfil",
            name="Data exfiltration",
            description=(
                "The attack tries to leak and extract the system prompt and "
                "exfiltrate sensitive data."
            ),
        ),
    ]
    snap = _write_snapshot(tmp_path / "unit_snapshot.json", techniques)

    report = snapshot_to_scenarios(snap, provenance)

    # Real scenarios were emitted from real (template) attack strings.
    assert report.emitted_count > 0
    for scenario in report.scenarios:
        assert scenario.source == "harvest_pipeline"
        # Provenance origin + date folded into the description.
        assert "https://example.test/threat-intel/snapshot" in scenario.description
        assert "2026-06-17" in scenario.description
        assert scenario.payload and scenario.payload.strip()

    # Round-trip through ScenarioLoader after write_drafts.
    out_path = IntelScenarioExtractor().write_drafts(
        report.scenarios, output_dir=tmp_path, source_slug="l15-unit-source"
    )
    assert out_path == tmp_path / "l15-unit-source.yaml"
    loaded = ScenarioLoader(tmp_path).load_all()
    assert len(loaded) == report.emitted_count
    assert all(s.source == "harvest_pipeline" for s in loaded)
    # stable_ids survive the round-trip.
    assert {s.stable_id for s in loaded} == {s.stable_id for s in report.scenarios}


def test_no_match_technique_is_not_fabricated(
    provenance: IntelProvenance, tmp_path: Path
):
    """A description with NO template keyword never becomes an attack payload.

    The real template pipeline emits zero samples for a keyword-free technique,
    so it contributes zero scenarios. This proves a bare description is never
    promoted to a payload (the #1 quality rule), via the REAL pipeline.
    """
    techniques = [
        TechniqueEntry(
            id="T-override",
            name="Instruction override",
            description="Attackers ignore and override system instructions.",
        ),
        TechniqueEntry(
            id="T-nomatch",
            name="Benign forecast",
            description="A plain weather forecast with no attack keywords at all.",
        ),
    ]
    snap = _write_snapshot(tmp_path / "nomatch_snapshot.json", techniques)

    report = snapshot_to_scenarios(snap, provenance)

    # Something was emitted (from the override technique)...
    assert report.emitted_count > 0
    # ...but NOTHING traces back to the no-match technique, and its description
    # text is nowhere in any emitted payload.
    emitted_ids = {s.name for s in report.scenarios}
    assert not any("T-nomatch" in name for name in emitted_ids)
    for scenario in report.scenarios:
        assert "weather forecast" not in (scenario.payload or "")


def test_unplaceable_sample_is_skipped_and_reported(
    provenance: IntelProvenance, tmp_path: Path
):
    """A sample the extractor cannot place is SKIPPED and surfaced, not dropped.

    The real template pipeline only ever emits valid-category samples, so to
    exercise the extractor's skip-accounting we inject a tiny offline pipeline
    (a real object, not a mock of any external service) that returns a
    GeneratedSample with an unknown taxonomy code. The REAL extractor does the
    skipping; we assert the skip is reported, proving nothing is silently lost.
    """

    class _UnplaceablePipeline:
        """Offline pipeline stub returning a sample with an invalid category."""

        def generate(self, incidents):  # noqa: ANN001 — mirrors real signature
            return [
                GeneratedSample(
                    text="a real-looking attack string",
                    source_incident_id=incidents[0].id if incidents else "x",
                    category_hint="ZZ9.9",  # not in the taxonomy
                )
            ]

    techniques = [
        TechniqueEntry(id="T1", name="x", description="anything")
    ]
    snap = _write_snapshot(tmp_path / "skip_snapshot.json", techniques)

    report = snapshot_to_scenarios(
        snap, provenance, pipeline=_UnplaceablePipeline()
    )
    assert report.emitted_count == 0
    assert report.skipped_count == 1
    assert "not in taxonomy" in report.skipped[0].reason
    assert report.scenarios == []


def test_empty_snapshot_returns_empty_report(
    provenance: IntelProvenance, tmp_path: Path
):
    """A snapshot with no techniques returns an empty report, never crashes."""
    snap = _write_snapshot(tmp_path / "empty_snapshot.json", [])

    report = snapshot_to_scenarios(snap, provenance)

    assert isinstance(report, HarvestReport)
    assert report.emitted_count == 0
    assert report.skipped_count == 0


def test_missing_file_raises_file_not_found(
    provenance: IntelProvenance, tmp_path: Path
):
    with pytest.raises(FileNotFoundError, match="snapshot not found"):
        snapshot_to_scenarios(tmp_path / "does_not_exist.json", provenance)


def test_malformed_json_raises_value_error(
    provenance: IntelProvenance, tmp_path: Path
):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        snapshot_to_scenarios(bad, provenance)


def test_schema_violation_raises_value_error(
    provenance: IntelProvenance, tmp_path: Path
):
    """Valid JSON but missing required snapshot keys -> ValueError (not a crash)."""
    bad = tmp_path / "schema_bad.json"
    # Valid JSON, but no 'source_name'/'version'/'fetched_at' -> SchemaValidationError.
    bad.write_text(json.dumps({"techniques": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the snapshot schema"):
        snapshot_to_scenarios(bad, provenance)


def test_default_pipeline_is_offline_template_only(
    provenance: IntelProvenance, tmp_path: Path
):
    """The default path uses IncidentToSamplePipeline(llm_client=None).

    Sanity-check that passing a default and passing an explicit offline pipeline
    produce identical emitted counts — i.e. the default really is the offline
    template path, not something that would otherwise reach for an LLM.
    """
    techniques = [
        TechniqueEntry(
            id="T-persona",
            name="Persona hijack",
            description="Use a persona / roleplay character (DAN) to pretend.",
        )
    ]
    snap = _write_snapshot(tmp_path / "persona_snapshot.json", techniques)

    default_report = snapshot_to_scenarios(snap, provenance)
    explicit_report = snapshot_to_scenarios(
        snap, provenance, pipeline=IncidentToSamplePipeline(llm_client=None)
    )
    assert default_report.emitted_count == explicit_report.emitted_count
    assert default_report.emitted_count > 0
