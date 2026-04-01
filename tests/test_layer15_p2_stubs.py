"""Tests for Layer 15 P2 stubs: Incident-to-Sample pipeline and TAP/PAIR.

Covers:
- Incident-to-sample: template matching, generation, edge cases
- TAP/PAIR: stub behavior, evaluate() with mock detector
"""

import pytest

from na0s.layer15.base import TechniqueEntry
from na0s.layer15.incident_to_sample import GeneratedSample, IncidentToSamplePipeline
from na0s.layer15.red_teaming import (
    PAIRStub,
    RedTeamProbe,
    RedTeamResult,
    TAPStub,
)


# ===================================================================
# Incident-to-Sample Tests
# ===================================================================


class TestIncidentToSamplePipeline:

    def setup_method(self):
        self.pipeline = IncidentToSamplePipeline()

    def test_generates_samples_from_override_incident(self):
        """Incident mentioning 'jailbreak' matches D1 override templates."""
        incidents = [
            TechniqueEntry(
                id="AIID-42",
                name="Jailbreak attack",
                description="Users found a way to jailbreak the chatbot by telling it to ignore instructions",
            )
        ]
        samples = self.pipeline.generate(incidents)
        assert len(samples) > 0
        assert all(isinstance(s, GeneratedSample) for s in samples)
        assert all(s.source_incident_id == "AIID-42" for s in samples)
        # Should match D1 (instruction override) category
        categories = {s.category_hint for s in samples}
        assert "D1" in categories

    def test_generates_samples_from_exfiltration_incident(self):
        incidents = [
            TechniqueEntry(
                id="AIID-43",
                name="Data leak",
                description="Attacker used prompt injection to exfiltrate sensitive data from the model",
            )
        ]
        samples = self.pipeline.generate(incidents)
        assert len(samples) > 0
        categories = {s.category_hint for s in samples}
        assert "E" in categories

    def test_generates_samples_from_persona_incident(self):
        incidents = [
            TechniqueEntry(
                id="AIID-44",
                name="DAN exploit",
                description="Users tricked the AI into a DAN persona roleplay",
            )
        ]
        samples = self.pipeline.generate(incidents)
        assert len(samples) > 0
        categories = {s.category_hint for s in samples}
        assert "D2" in categories

    def test_no_match_returns_empty(self):
        """Incident with no matching keywords produces no samples."""
        incidents = [
            TechniqueEntry(
                id="AIID-99",
                name="Unrelated",
                description="The system had a performance degradation issue",
            )
        ]
        samples = self.pipeline.generate(incidents)
        assert samples == []

    def test_empty_incidents_returns_empty(self):
        samples = self.pipeline.generate([])
        assert samples == []

    def test_confidence_increases_with_more_keyword_matches(self):
        # This incident matches multiple override keywords
        incidents = [
            TechniqueEntry(
                id="AIID-50",
                name="Multi-keyword",
                description="User tried to override and bypass and ignore the safety jailbreak",
            )
        ]
        samples = self.pipeline.generate(incidents)
        assert len(samples) > 0
        # With 4/4 keywords matching, confidence should be high
        max_confidence = max(s.confidence for s in samples)
        assert max_confidence >= 0.75

    def test_sample_metadata_includes_method(self):
        incidents = [
            TechniqueEntry(
                id="AIID-51",
                name="Test",
                description="Attacker tried to bypass the system",
            )
        ]
        samples = self.pipeline.generate(incidents)
        for sample in samples:
            assert sample.metadata["generation_method"] == "template"
            assert "matched_keywords" in sample.metadata


# ===================================================================
# TAP/PAIR Stub Tests
# ===================================================================


class TestTAPStub:

    def test_generate_returns_empty(self):
        tap = TAPStub()
        probes = tap.generate("provide harmful instructions")
        assert probes == []

    def test_name_is_tap(self):
        assert TAPStub().name == "tap"


class TestPAIRStub:

    def test_generate_returns_empty(self):
        pair = PAIRStub()
        probes = pair.generate("provide harmful instructions")
        assert probes == []

    def test_name_is_pair(self):
        assert PAIRStub().name == "pair"


class TestRedTeamEvaluate:

    def test_evaluate_with_perfect_detector(self):
        """Detector catches everything → 100% detection rate."""
        tap = TAPStub()
        probes = [
            RedTeamProbe(text="attack 1", algorithm="tap"),
            RedTeamProbe(text="attack 2", algorithm="tap"),
        ]
        result = tap.evaluate(probes, detector_fn=lambda x: True)
        assert result.detection_rate == 1.0
        assert result.probes_successful == 0

    def test_evaluate_with_no_detection(self):
        """Detector catches nothing → 0% detection rate."""
        tap = TAPStub()
        probes = [
            RedTeamProbe(text="attack 1", algorithm="tap"),
            RedTeamProbe(text="attack 2", algorithm="tap"),
        ]
        result = tap.evaluate(probes, detector_fn=lambda x: False)
        assert result.detection_rate == 0.0
        assert result.probes_successful == 2

    def test_evaluate_empty_probes(self):
        tap = TAPStub()
        result = tap.evaluate([], detector_fn=lambda x: True)
        assert result.detection_rate == 1.0
        assert result.probes_generated == 0

    def test_evaluate_partial_detection(self):
        pair = PAIRStub()
        probes = [
            RedTeamProbe(text="caught", algorithm="pair"),
            RedTeamProbe(text="missed", algorithm="pair"),
        ]
        # Detect only probes containing "caught"
        result = pair.evaluate(
            probes, detector_fn=lambda x: "caught" in x
        )
        assert result.detection_rate == 0.5
        assert result.probes_successful == 1
