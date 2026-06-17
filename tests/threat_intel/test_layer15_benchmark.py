"""Tests for Layer 15 benchmark analyzer and dashboard generator.

Covers:
- BenchmarkAnalyzer: taxonomy loading, overlap computation, fuzzy matching
- DashboardGenerator: HTML/JSON output
- Edge cases: no benchmarks, empty taxonomy
"""

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
# Tokenization / Jaccard helpers
# ---------------------------------------------------------------------------


class TestTokenize:

    def test_basic_tokenization(self):
        tokens = _tokenize("Instruction Override Attack")
        assert "instruction" in tokens
        assert "override" in tokens
        assert "attack" in tokens

    def test_removes_stop_words(self):
        tokens = _tokenize("the instructions for the model")
        assert "the" not in tokens
        assert "for" not in tokens
        assert "instructions" in tokens
        assert "model" in tokens

    def test_splits_on_separators(self):
        tokens = _tokenize("data-poisoning/model_extraction")
        assert "data" in tokens
        assert "poisoning" in tokens
        assert "model" in tokens
        assert "extraction" in tokens

    def test_empty_string(self):
        assert _tokenize("") == set()


class TestJaccard:

    def test_identical_sets(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        score = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(score - 0.5) < 0.01  # 2/4

    def test_empty_sets(self):
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer
# ---------------------------------------------------------------------------

MOCK_TAXONOMY_YAML = """\
version: "1.0"
categories:
  D1:
    name: "Instruction Override"
    description: "Attempts to override system prompt instructions."
    type: direct
    severity: critical
    techniques:
      D1.1: { name: "Ignore-previous", severity: critical }
      D1.2: { name: "New-instruction injection", severity: critical }
  D2:
    name: "Persona/Roleplay Hijack"
    description: "Tricks the LLM into adopting an unrestricted persona."
    type: direct
    severity: high
    techniques:
      D2.1: { name: "DAN persona", severity: high }
  E:
    name: "Data Exfiltration"
    description: "Extracts sensitive data from the model."
    type: extraction
    severity: critical
    techniques:
      E.1: { name: "System prompt extraction", severity: critical }
"""


def _make_snapshot(source_name, techniques):
    return SourceSnapshot(
        source_name=source_name,
        fetched_at=datetime(2026, 3, 24, tzinfo=timezone.utc),
        version="v1",
        techniques=techniques,
    )


@pytest.fixture
def analyzer_env(tmp_path):
    """Set up taxonomy YAML and snapshots directory."""
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(MOCK_TAXONOMY_YAML)
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    return taxonomy_path, snapshots_dir


class TestBenchmarkAnalyzer:

    def test_load_taxonomy(self, analyzer_env):
        taxonomy_path, snapshots_dir = analyzer_env
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path, snapshots_dir=snapshots_dir
        )
        cats = analyzer.load_taxonomy()
        assert "D1" in cats
        assert "D2" in cats
        assert "E" in cats
        assert cats["D1"]["name"] == "Instruction Override"

    def test_analyze_with_no_benchmarks(self, analyzer_env):
        taxonomy_path, snapshots_dir = analyzer_env
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path, snapshots_dir=snapshots_dir
        )
        analysis = analyzer.analyze()
        assert analysis.na0s_categories == 3
        assert analysis.na0s_techniques == 4  # D1: 2, D2: 1, E: 1
        assert analysis.benchmarks == {}
        assert analysis.overall_overlap_pct == 0.0
        assert len(analysis.na0s_unique) == 3

    def test_analyze_with_overlapping_benchmark(self, analyzer_env):
        taxonomy_path, snapshots_dir = analyzer_env
        # Create a benchmark snapshot with overlapping techniques
        snapshot = _make_snapshot(
            "jailbreakbench",
            [
                TechniqueEntry(
                    id="jailbreakbench.override",
                    name="Instruction override jailbreak",
                    description="Override system instructions to bypass safety",
                ),
                TechniqueEntry(
                    id="jailbreakbench.persona",
                    name="DAN persona hijack roleplay",
                    description="Adopt unrestricted persona through roleplay",
                ),
                TechniqueEntry(
                    id="jailbreakbench.unrelated",
                    name="Completely unrelated xyz thing",
                    description="Something totally different with no keyword overlap",
                ),
            ],
        )
        snap_path = snapshots_dir / "jailbreakbench_snapshot.json"
        with open(snap_path, "w") as f:
            json.dump(snapshot.to_dict(), f)

        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path, snapshots_dir=snapshots_dir
        )
        analysis = analyzer.analyze()

        assert "jailbreakbench" in analysis.benchmarks
        assert analysis.benchmarks["jailbreakbench"] == 3
        # At least some categories should have benchmark coverage
        covered = [c for c in analysis.coverage if c.coverage_level != "none"]
        assert len(covered) > 0
        assert analysis.overall_overlap_pct > 0

    def test_fuzzy_matching_finds_related(self):
        """Override-related benchmark items match D1 (Instruction Override)."""
        analyzer = BenchmarkAnalyzer()
        # Direct fuzzy test
        cat_tokens = _tokenize("Instruction Override system prompt")
        bench_tokens = _tokenize("instruction override jailbreak prompt")
        score = _jaccard(cat_tokens, bench_tokens)
        assert score >= 0.15  # Should match

    def test_fuzzy_matching_rejects_unrelated(self):
        cat_tokens = _tokenize("Instruction Override")
        bench_tokens = _tokenize("quantum computing algorithms")
        score = _jaccard(cat_tokens, bench_tokens)
        assert score < 0.15  # Should not match

    def test_json_serialization(self, analyzer_env):
        taxonomy_path, snapshots_dir = analyzer_env
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path, snapshots_dir=snapshots_dir
        )
        analysis = analyzer.analyze()
        json_str = analyzer.to_json(analysis)
        data = json.loads(json_str)
        assert data["na0s_categories"] == 3
        assert "coverage" in data
        assert len(data["coverage"]) == 3

    def test_invalid_snapshot_skipped(self, analyzer_env):
        """Malformed snapshot file is skipped gracefully."""
        taxonomy_path, snapshots_dir = analyzer_env
        bad_path = snapshots_dir / "bad_snapshot.json"
        bad_path.write_text("{invalid json")
        analyzer = BenchmarkAnalyzer(
            taxonomy_path=taxonomy_path, snapshots_dir=snapshots_dir
        )
        snapshots = analyzer.load_benchmark_snapshots()
        assert len(snapshots) == 0


# ---------------------------------------------------------------------------
# DashboardGenerator
# ---------------------------------------------------------------------------


class TestDashboardGenerator:

    def _make_analysis(self):
        return BenchmarkAnalysis(
            timestamp=datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc),
            na0s_categories=3,
            na0s_techniques=5,
            benchmarks={"jailbreakbench": 10, "harmbench": 8},
            coverage=[
                CategoryCoverage(
                    category_id="D1",
                    category_name="Instruction Override",
                    na0s_technique_count=3,
                    benchmark_matches={"jailbreakbench": 5, "harmbench": 2},
                    coverage_level="strong",
                    gaps=[],
                ),
                CategoryCoverage(
                    category_id="D2",
                    category_name="Persona Hijack",
                    na0s_technique_count=1,
                    benchmark_matches={"jailbreakbench": 1, "harmbench": 0},
                    coverage_level="partial",
                    gaps=["[harmbench] roleplay attacks"],
                ),
                CategoryCoverage(
                    category_id="E",
                    category_name="Data Exfiltration",
                    na0s_technique_count=1,
                    benchmark_matches={"jailbreakbench": 0, "harmbench": 0},
                    coverage_level="none",
                    gaps=[],
                ),
            ],
            na0s_unique=["E"],
            benchmark_unique=["[jailbreakbench] novel_attack"],
            overall_overlap_pct=66.7,
        )

    def test_generates_html_file(self, tmp_path):
        gen = DashboardGenerator()
        analysis = self._make_analysis()
        result = gen.generate(analysis, tmp_path)
        assert result.exists()
        assert result.suffix == ".html"

    def test_generates_json_file(self, tmp_path):
        gen = DashboardGenerator()
        analysis = self._make_analysis()
        gen.generate(analysis, tmp_path)
        json_path = tmp_path / "dashboard_data.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["na0s_categories"] == 3

    def test_html_contains_key_sections(self, tmp_path):
        gen = DashboardGenerator()
        analysis = self._make_analysis()
        html_path = gen.generate(analysis, tmp_path)
        html = html_path.read_text()
        assert "Na0S Cross-Benchmark" in html
        assert "Instruction Override" in html
        assert "D1" in html
        assert "66.7%" in html
        assert "Coverage" in html or "coverage" in html

    def test_html_is_valid_structure(self, tmp_path):
        gen = DashboardGenerator()
        analysis = self._make_analysis()
        html_path = gen.generate(analysis, tmp_path)
        html = html_path.read_text()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<table" in html

    def test_no_benchmarks_dashboard(self, tmp_path):
        gen = DashboardGenerator()
        analysis = BenchmarkAnalysis(
            timestamp=datetime(2026, 3, 24, tzinfo=timezone.utc),
            na0s_categories=2,
            na0s_techniques=5,
            benchmarks={},
            coverage=[
                CategoryCoverage(
                    category_id="D1",
                    category_name="Test",
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
        html_path = gen.generate(analysis, tmp_path)
        html = html_path.read_text()
        assert "0%" in html or "0.0%" in html
        assert "No benchmarks" in html or "0 benchmark" in html
