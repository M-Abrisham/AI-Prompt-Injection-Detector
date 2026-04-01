"""Tests for the Layer 15 cross-benchmark validation analyzer and dashboard.

All tests use mock taxonomy data and mock snapshots -- no real files or
network access required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from na0s.layer15.base import SourceSnapshot, TechniqueEntry
from na0s.layer15.benchmark_analyzer import (
    BenchmarkAnalysis,
    BenchmarkAnalyzer,
    CategoryCoverage,
    _jaccard,
    _tokenize,
)
from na0s.layer15.dashboard_generator import DashboardGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TAXONOMY = {
    "version": "1.0",
    "categories": {
        "D1": {
            "name": "Instruction Override",
            "description": "Attempts to override or ignore the system prompt instructions.",
            "type": "direct",
            "severity": "critical",
            "techniques": {
                "D1.1": {"name": "Ignore-previous", "severity": "critical"},
                "D1.2": {"name": "New-instruction injection", "severity": "critical"},
                "D1.3": {"name": "Priority-override", "severity": "high"},
            },
        },
        "D2": {
            "name": "Persona/Roleplay Hijack",
            "description": "Tricks the LLM into adopting an unrestricted persona.",
            "type": "direct",
            "severity": "high",
            "techniques": {
                "D2.1": {"name": "DAN-variants", "severity": "high"},
                "D2.2": {"name": "Amoral-character", "severity": "high"},
            },
        },
        "M": {
            "name": "Multimodal Injection",
            "description": "Attacks delivered through non-text modalities like images and audio.",
            "type": "multimodal",
            "severity": "high",
            "techniques": {
                "M1.1": {"name": "Hidden text in images", "severity": "high"},
            },
        },
    },
}


def _write_taxonomy(tmp_path: Path) -> Path:
    """Write a sample taxonomy YAML and return its path."""
    import yaml

    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(yaml.dump(SAMPLE_TAXONOMY), encoding="utf-8")
    return taxonomy_path


def _write_snapshot(
    snapshots_dir: Path,
    source_name: str = "jailbreakbench",
    techniques: list[TechniqueEntry] | None = None,
) -> Path:
    """Write a snapshot JSON file and return its path."""
    if techniques is None:
        techniques = [
            TechniqueEntry(
                id="jailbreakbench.data/jailbreaks.csv",
                name="jailbreaks",
                description="Jailbreak prompt dataset, instruction override techniques",
            ),
            TechniqueEntry(
                id="harmbench.data/harmful_behaviors.csv",
                name="harmful_behaviors",
                description="Harmful behavior generation requests, persona hijack",
            ),
        ]
    snap = SourceSnapshot(
        source_name=source_name,
        fetched_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        version="abc123",
        techniques=techniques,
    )
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshots_dir / f"{source_name}_snapshot.json"
    path.write_text(json.dumps(snap.to_dict(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# TestBenchmarkAnalyzer
# ---------------------------------------------------------------------------


class TestBenchmarkAnalyzer:
    """Tests for BenchmarkAnalyzer."""

    def test_load_taxonomy(self, tmp_path: Path) -> None:
        """Taxonomy YAML is parsed into a categories dict."""
        taxonomy_path = _write_taxonomy(tmp_path)
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path,
            snapshots_dir=tmp_path / "snaps",
        )
        cats = analyzer.load_taxonomy()
        assert "D1" in cats
        assert "D2" in cats
        assert cats["D1"]["name"] == "Instruction Override"
        assert "D1.1" in cats["D1"]["techniques"]

    def test_analyze_with_overlapping_data(self, tmp_path: Path) -> None:
        """Analysis finds overlap between taxonomy and benchmark items."""
        taxonomy_path = _write_taxonomy(tmp_path)
        snaps_dir = tmp_path / "snaps"
        _write_snapshot(
            snaps_dir,
            techniques=[
                TechniqueEntry(
                    id="jailbreakbench.jailbreaks",
                    name="instruction override jailbreak prompts",
                    description="Ignore previous instruction override attempts",
                ),
                TechniqueEntry(
                    id="harmbench.harmful",
                    name="persona roleplay hijack",
                    description="DAN variants amoral character persona tricks",
                ),
            ],
        )

        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path,
            snapshots_dir=snaps_dir,
        )
        result = analyzer.analyze()

        assert result.na0s_categories == 3
        assert result.na0s_techniques == 6
        assert result.overall_overlap_pct > 0
        # At least one category should have matches.
        matched_cats = [
            c for c in result.coverage if c.coverage_level != "none"
        ]
        assert len(matched_cats) > 0

    def test_analyze_with_no_benchmarks(self, tmp_path: Path) -> None:
        """Analysis works when no benchmark snapshots exist."""
        taxonomy_path = _write_taxonomy(tmp_path)
        snaps_dir = tmp_path / "empty_snaps"
        # Do NOT create any snapshot files.

        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path,
            snapshots_dir=snaps_dir,
        )
        result = analyzer.analyze()

        assert result.na0s_categories == 3
        assert result.na0s_techniques == 6
        assert result.benchmarks == {}
        assert result.overall_overlap_pct == 0.0
        assert result.benchmark_unique == []
        # All categories should be Na0S-unique when no benchmarks.
        assert len(result.na0s_unique) == 3
        # All coverage levels should be "none".
        for cov in result.coverage:
            assert cov.coverage_level == "none"

    def test_fuzzy_matching_finds_related_categories(self) -> None:
        """Jaccard matching finds related items above threshold."""
        # "instruction override" and "ignore previous instruction"
        # share tokens.
        a = _tokenize("instruction override ignore previous")
        b = _tokenize("instruction override jailbreak prompts")
        score = _jaccard(a, b)
        assert score >= BenchmarkAnalyzer.MATCH_THRESHOLD
        assert score > 0.15  # should share "instruction", "override"

    def test_fuzzy_matching_rejects_unrelated(self) -> None:
        """Jaccard matching rejects items below threshold."""
        a = _tokenize("multimodal audio injection ultrasonic")
        b = _tokenize("supply chain dependency compromise")
        score = _jaccard(a, b)
        assert score < BenchmarkAnalyzer.MATCH_THRESHOLD

    def test_json_serialization(self, tmp_path: Path) -> None:
        """Analysis can be serialized to valid JSON."""
        taxonomy_path = _write_taxonomy(tmp_path)
        snaps_dir = tmp_path / "snaps"
        _write_snapshot(snaps_dir)

        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path,
            snapshots_dir=snaps_dir,
        )
        result = analyzer.analyze()
        json_str = analyzer.to_json(result)

        # Must be valid JSON.
        data = json.loads(json_str)
        assert "timestamp" in data
        assert "na0s_categories" in data
        assert "coverage" in data
        assert isinstance(data["coverage"], list)
        assert len(data["coverage"]) == 3

        # Round-trip: values match.
        assert data["na0s_categories"] == result.na0s_categories
        assert data["na0s_techniques"] == result.na0s_techniques
        assert data["overall_overlap_pct"] == result.overall_overlap_pct

    def test_load_benchmark_snapshots_empty_dir(self, tmp_path: Path) -> None:
        """Loading snapshots from a non-existent directory returns empty."""
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=tmp_path / "taxonomy.yaml",
            snapshots_dir=tmp_path / "does_not_exist",
        )
        result = analyzer.load_benchmark_snapshots()
        assert result == {}

    def test_load_benchmark_snapshots_skips_invalid(self, tmp_path: Path) -> None:
        """Invalid snapshot files are skipped without crashing."""
        snaps_dir = tmp_path / "snaps"
        snaps_dir.mkdir()
        bad_file = snaps_dir / "bad_snapshot.json"
        bad_file.write_text("{invalid json", encoding="utf-8")

        analyzer = BenchmarkAnalyzer(
            taxonomy_path=tmp_path / "taxonomy.yaml",
            snapshots_dir=snaps_dir,
        )
        result = analyzer.load_benchmark_snapshots()
        assert result == {}


# ---------------------------------------------------------------------------
# TestDashboardGenerator
# ---------------------------------------------------------------------------


class TestDashboardGenerator:
    """Tests for DashboardGenerator."""

    def _make_analysis(self) -> BenchmarkAnalysis:
        """Create a minimal BenchmarkAnalysis for testing."""
        return BenchmarkAnalysis(
            timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
            na0s_categories=3,
            na0s_techniques=6,
            benchmarks={"jailbreakbench": 5, "harmbench": 3},
            coverage=[
                CategoryCoverage(
                    category_id="D1",
                    category_name="Instruction Override",
                    na0s_technique_count=3,
                    benchmark_matches={"jailbreakbench": 2, "harmbench": 1},
                    coverage_level="strong",
                    gaps=[],
                ),
                CategoryCoverage(
                    category_id="D2",
                    category_name="Persona/Roleplay Hijack",
                    na0s_technique_count=2,
                    benchmark_matches={"jailbreakbench": 1, "harmbench": 0},
                    coverage_level="partial",
                    gaps=["[harmbench] adversarial suffix"],
                ),
                CategoryCoverage(
                    category_id="M",
                    category_name="Multimodal Injection",
                    na0s_technique_count=1,
                    benchmark_matches={"jailbreakbench": 0, "harmbench": 0},
                    coverage_level="none",
                    gaps=[],
                ),
            ],
            na0s_unique=["M"],
            benchmark_unique=["[harmbench] novel_technique"],
            overall_overlap_pct=66.7,
        )

    def test_generates_html_file(self, tmp_path: Path) -> None:
        """Dashboard generator creates an HTML file."""
        analysis = self._make_analysis()
        gen = DashboardGenerator()
        html_path = gen.generate(analysis, tmp_path / "out")

        assert html_path.exists()
        assert html_path.name == "dashboard.html"
        assert html_path.stat().st_size > 0

    def test_generates_json_file(self, tmp_path: Path) -> None:
        """Dashboard generator creates a companion JSON data file."""
        analysis = self._make_analysis()
        gen = DashboardGenerator()
        gen.generate(analysis, tmp_path / "out")

        json_path = tmp_path / "out" / "dashboard_data.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["na0s_categories"] == 3
        assert "coverage" in data

    def test_html_contains_key_sections(self, tmp_path: Path) -> None:
        """Generated HTML contains all required dashboard sections."""
        analysis = self._make_analysis()
        gen = DashboardGenerator()
        html_path = gen.generate(analysis, tmp_path / "out")
        html = html_path.read_text(encoding="utf-8")

        # Header section.
        assert "Na0S Cross-Benchmark Validation" in html
        assert "2026-03-24" in html

        # Stats box.
        assert "Na0S Categories" in html
        assert "Na0S Techniques" in html
        assert "Overall Overlap" in html
        assert "66.7%" in html

        # Coverage table.
        assert "Coverage Heatmap" in html
        assert "Instruction Override" in html
        assert "Persona/Roleplay Hijack" in html
        assert "Multimodal Injection" in html
        assert "jailbreakbench" in html
        assert "harmbench" in html

        # Gap summary.
        assert "Benchmark Items Na0S Should Add" in html
        assert "Na0S-Unique Coverage" in html
        assert "novel_technique" in html

        # Priority gaps section.
        assert "Priority Gaps" in html

    def test_html_is_valid_without_javascript(self, tmp_path: Path) -> None:
        """HTML tables render as plain HTML (no JS required for content)."""
        analysis = self._make_analysis()
        gen = DashboardGenerator()
        html_path = gen.generate(analysis, tmp_path / "out")
        html = html_path.read_text(encoding="utf-8")

        # All coverage data is in the static HTML, not injected by JS.
        assert "<table" in html
        assert "<tr>" in html
        assert "<td>" in html
        # The actual category names are in the static HTML.
        assert "D1" in html
        assert "D2" in html

    def test_html_with_no_benchmarks(self, tmp_path: Path) -> None:
        """Dashboard renders correctly when no benchmarks are loaded."""
        analysis = BenchmarkAnalysis(
            timestamp=datetime(2026, 3, 24, 10, 0, tzinfo=timezone.utc),
            na0s_categories=2,
            na0s_techniques=5,
            benchmarks={},
            coverage=[
                CategoryCoverage(
                    category_id="D1",
                    category_name="Instruction Override",
                    na0s_technique_count=3,
                    benchmark_matches={},
                    coverage_level="none",
                    gaps=[],
                ),
            ],
            na0s_unique=["D1"],
            benchmark_unique=[],
            overall_overlap_pct=0.0,
        )
        gen = DashboardGenerator()
        html_path = gen.generate(analysis, tmp_path / "out")
        html = html_path.read_text(encoding="utf-8")

        assert "No benchmarks loaded" in html or "0 benchmark(s) loaded" in html
        assert "Na0S Cross-Benchmark Validation" in html
