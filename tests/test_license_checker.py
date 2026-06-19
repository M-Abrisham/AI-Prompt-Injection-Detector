"""Tests for scripts/license_checker.py.

Validates license classification, conflict detection, report generation,
offline fallback, strict mode, and CLI flags -- all without making real
HuggingFace Hub API calls.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import pytest
import yaml

from scripts.license_checker import HF_HUB_AVAILABLE


def _write_registry(tmpdir, sources):
    """Write a minimal datasets.yaml with the given sources dict."""
    registry = {
        "version": "1.0",
        "output_dir": "data/raw",
        "sources": sources,
    }
    path = os.path.join(tmpdir, "datasets.yaml")
    with open(path, "w") as fh:
        yaml.dump(registry, fh)
    return path


class TestClassifyLicense(unittest.TestCase):
    """Test the classify_license function."""

    def test_permissive_mit(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("mit"), "permissive")

    def test_permissive_apache(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("apache-2.0"), "permissive")

    def test_permissive_cc_by(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("cc-by-4.0"), "permissive")

    def test_permissive_cc0(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("cc0-1.0"), "permissive")

    def test_permissive_case_insensitive(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("MIT"), "permissive")
        self.assertEqual(classify_license("Apache-2.0"), "permissive")

    def test_restrictive_cc_by_nc(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("cc-by-nc-4.0"), "restrictive")

    def test_restrictive_cc_by_sa(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("cc-by-sa-4.0"), "restrictive")

    def test_restrictive_cc_by_nc_sa(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("cc-by-nc-sa-4.0"), "restrictive")

    def test_restrictive_gpl(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("gpl-3.0"), "restrictive")

    def test_unknown_none(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license(None), "unknown")

    def test_unknown_empty(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license(""), "unknown")

    def test_unknown_custom(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("my-custom-license"), "unknown")

    def test_permissive_with_whitespace(self):
        from scripts.license_checker import classify_license
        self.assertEqual(classify_license("  mit  "), "permissive")


class TestConflictDetection(unittest.TestCase):
    """Test the detect_conflicts and individual conflict helpers."""

    def test_nc_conflict(self):
        from scripts.license_checker import has_nc_conflict, detect_conflicts
        self.assertTrue(has_nc_conflict("cc-by-nc-4.0"))
        self.assertIn("non-commercial", detect_conflicts("cc-by-nc-4.0"))

    def test_sa_conflict(self):
        from scripts.license_checker import has_sa_conflict, detect_conflicts
        self.assertTrue(has_sa_conflict("cc-by-sa-4.0"))
        self.assertIn("share-alike", detect_conflicts("cc-by-sa-4.0"))

    def test_nd_conflict(self):
        from scripts.license_checker import has_nd_conflict, detect_conflicts
        self.assertTrue(has_nd_conflict("cc-by-nd-4.0"))
        self.assertIn("no-derivatives", detect_conflicts("cc-by-nd-4.0"))

    def test_nc_sa_combined(self):
        from scripts.license_checker import detect_conflicts
        conflicts = detect_conflicts("cc-by-nc-sa-4.0")
        self.assertIn("non-commercial", conflicts)
        self.assertIn("share-alike", conflicts)

    def test_no_conflicts_permissive(self):
        from scripts.license_checker import detect_conflicts
        self.assertEqual(detect_conflicts("mit"), [])
        self.assertEqual(detect_conflicts("apache-2.0"), [])

    def test_no_conflicts_none(self):
        from scripts.license_checker import detect_conflicts
        self.assertEqual(detect_conflicts(None), [])

    def test_nc_nd_combined(self):
        from scripts.license_checker import detect_conflicts
        conflicts = detect_conflicts("cc-by-nc-nd-4.0")
        self.assertIn("non-commercial", conflicts)
        self.assertIn("no-derivatives", conflicts)


class TestParseDatasets(unittest.TestCase):
    """Test registry parsing to extract HuggingFace datasets."""

    def test_extracts_hf_datasets(self):
        from scripts.license_checker import parse_datasets

        sources = {
            "alpaca": {
                "type": "huggingface",
                "repo": "tatsu-lab/alpaca",
                "split": "train",
                "text_column": "instruction",
                "label": 0,
                "output": "hf_alpaca.csv",
            },
            "github_source": {
                "type": "github_csv",
                "url": "https://example.com/data.csv",
                "text_column": "prompt",
                "label": 1,
                "output": "github.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            datasets = parse_datasets(path)

        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0][0], "alpaca")
        self.assertEqual(datasets[0][1], "tatsu-lab/alpaca")

    def test_skips_github_csv(self):
        from scripts.license_checker import parse_datasets

        sources = {
            "github_only": {
                "type": "github_csv",
                "url": "https://example.com/data.csv",
                "text_column": "prompt",
                "label": 0,
                "output": "gh.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            datasets = parse_datasets(path)

        self.assertEqual(len(datasets), 0)

    def test_multiple_hf_datasets(self):
        from scripts.license_checker import parse_datasets

        sources = {
            "ds1": {
                "type": "huggingface",
                "repo": "org1/dataset1",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "ds1.csv",
            },
            "ds2": {
                "type": "huggingface",
                "repo": "org2/dataset2",
                "split": "train",
                "text_column": "text",
                "label": 1,
                "output": "ds2.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            datasets = parse_datasets(path)

        self.assertEqual(len(datasets), 2)
        repos = {d[1] for d in datasets}
        self.assertIn("org1/dataset1", repos)
        self.assertIn("org2/dataset2", repos)


@pytest.mark.skipif(
    not HF_HUB_AVAILABLE,
    reason=(
        "huggingface_hub not installed: scripts.license_checker.hf_dataset_info "
        "is unbound on the ImportError path, so mock.patch of it would AttributeError"
    ),
)
class TestFetchLicenseFromHub(unittest.TestCase):
    """Test HF Hub API fetching with mocked responses."""

    @mock.patch("scripts.license_checker.HF_HUB_AVAILABLE", True)
    @mock.patch("scripts.license_checker.hf_dataset_info")
    def test_returns_license(self, mock_info):
        from scripts.license_checker import fetch_license_from_hub

        mock_info.return_value = mock.Mock(license="apache-2.0", tags=[])
        result = fetch_license_from_hub("tatsu-lab/alpaca")
        self.assertEqual(result, "apache-2.0")

    @mock.patch("scripts.license_checker.HF_HUB_AVAILABLE", True)
    @mock.patch("scripts.license_checker.hf_dataset_info")
    def test_fallback_to_tags(self, mock_info):
        from scripts.license_checker import fetch_license_from_hub

        mock_info.return_value = mock.Mock(
            license=None, tags=["task:text-classification", "license:mit"]
        )
        result = fetch_license_from_hub("some/repo")
        self.assertEqual(result, "mit")

    @mock.patch("scripts.license_checker.HF_HUB_AVAILABLE", True)
    @mock.patch("scripts.license_checker.hf_dataset_info")
    def test_returns_none_on_error(self, mock_info):
        from scripts.license_checker import fetch_license_from_hub

        mock_info.side_effect = Exception("Network error")
        result = fetch_license_from_hub("some/repo")
        self.assertIsNone(result)

    @mock.patch("scripts.license_checker.HF_HUB_AVAILABLE", False)
    def test_returns_none_when_hub_unavailable(self):
        from scripts.license_checker import fetch_license_from_hub

        result = fetch_license_from_hub("some/repo")
        self.assertIsNone(result)


class TestGenerateReport(unittest.TestCase):
    """Test full report generation with mocked HF API."""

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_report_structure(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = "apache-2.0"

        sources = {
            "alpaca": {
                "type": "huggingface",
                "repo": "tatsu-lab/alpaca",
                "split": "train",
                "text_column": "instruction",
                "label": 0,
                "output": "hf_alpaca.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=False)

        self.assertEqual(len(report), 1)
        entry = report[0]
        self.assertEqual(entry["name"], "alpaca")
        self.assertEqual(entry["repo"], "tatsu-lab/alpaca")
        self.assertEqual(entry["license"], "apache-2.0")
        self.assertEqual(entry["classification"], "permissive")
        self.assertEqual(entry["conflicts"], [])
        self.assertTrue(entry["compliant"])

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_restrictive_flagged(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = "cc-by-nc-4.0"

        sources = {
            "restricted_ds": {
                "type": "huggingface",
                "repo": "org/restricted",
                "split": "train",
                "text_column": "text",
                "label": 1,
                "output": "restricted.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=False)

        entry = report[0]
        self.assertEqual(entry["classification"], "restrictive")
        self.assertIn("non-commercial", entry["conflicts"])
        self.assertFalse(entry["compliant"])

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_offline_fallback_uses_cache(self, mock_fetch):
        from scripts.license_checker import generate_report

        # Simulate API being unreachable
        mock_fetch.return_value = None

        sources = {
            "ds1": {
                "type": "huggingface",
                "repo": "org/dataset1",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "ds1.csv",
            },
        }

        cached = {"org/dataset1": "mit"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, cached_licenses=cached, include_hf_registry=False)

        entry = report[0]
        self.assertEqual(entry["license"], "mit")
        self.assertEqual(entry["license_source"], "cache")
        self.assertEqual(entry["classification"], "permissive")

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_unknown_when_no_api_no_cache(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = None

        sources = {
            "mystery": {
                "type": "huggingface",
                "repo": "org/mystery",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "mystery.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=False)

        entry = report[0]
        self.assertIsNone(entry["license"])
        self.assertEqual(entry["license_source"], "unavailable")
        self.assertEqual(entry["classification"], "unknown")
        self.assertFalse(entry["compliant"])

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_mixed_report(self, mock_fetch):
        from scripts.license_checker import generate_report

        license_map = {
            "org/permissive": "apache-2.0",
            "org/restricted": "cc-by-nc-4.0",
            "org/unknown": None,
        }
        mock_fetch.side_effect = lambda repo: license_map.get(repo)

        sources = {
            "perm": {
                "type": "huggingface",
                "repo": "org/permissive",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "perm.csv",
            },
            "rest": {
                "type": "huggingface",
                "repo": "org/restricted",
                "split": "train",
                "text_column": "text",
                "label": 1,
                "output": "rest.csv",
            },
            "unk": {
                "type": "huggingface",
                "repo": "org/unknown",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "unk.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=False)

        by_name = {e["name"]: e for e in report}
        self.assertTrue(by_name["perm"]["compliant"])
        self.assertFalse(by_name["rest"]["compliant"])
        self.assertFalse(by_name["unk"]["compliant"])


class TestSaveReport(unittest.TestCase):
    """Test report serialisation to JSON."""

    def test_saves_valid_json(self):
        from scripts.license_checker import save_report

        report = [
            {
                "name": "test",
                "repo": "org/test",
                "license": "mit",
                "license_source": "hub",
                "classification": "permissive",
                "conflicts": [],
                "compliant": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            save_report(report, path)

            with open(path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["name"], "test")

    def test_creates_parent_directories(self):
        from scripts.license_checker import save_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "report.json")
            save_report([], path)
            self.assertTrue(os.path.exists(path))


class TestStrictMode(unittest.TestCase):
    """Test --strict CLI flag behaviour."""

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_strict_exits_1_on_restrictive(self, mock_fetch):
        from scripts.license_checker import main

        mock_fetch.return_value = "cc-by-nc-4.0"

        sources = {
            "restricted": {
                "type": "huggingface",
                "repo": "org/restricted",
                "split": "train",
                "text_column": "text",
                "label": 1,
                "output": "restricted.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = _write_registry(tmpdir, sources)
            out_path = os.path.join(tmpdir, "report.json")

            ret = main(["--datasets", reg_path, "--output", out_path, "--strict"])
            self.assertEqual(ret, 1)

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_strict_exits_1_on_unknown(self, mock_fetch):
        from scripts.license_checker import main

        mock_fetch.return_value = None

        sources = {
            "mystery": {
                "type": "huggingface",
                "repo": "org/mystery",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "mystery.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = _write_registry(tmpdir, sources)
            out_path = os.path.join(tmpdir, "report.json")

            ret = main(["--datasets", reg_path, "--output", out_path, "--strict"])
            self.assertEqual(ret, 1)

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_strict_exits_0_when_all_permissive(self, mock_fetch):
        from scripts.license_checker import main

        mock_fetch.return_value = "apache-2.0"

        sources = {
            "good_ds": {
                "type": "huggingface",
                "repo": "org/good",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "good.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = _write_registry(tmpdir, sources)
            out_path = os.path.join(tmpdir, "report.json")

            ret = main(["--datasets", reg_path, "--output", out_path, "--strict"])
            self.assertEqual(ret, 0)

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_non_strict_exits_0_even_with_restrictive(self, mock_fetch):
        from scripts.license_checker import main

        mock_fetch.return_value = "cc-by-nc-4.0"

        sources = {
            "restricted": {
                "type": "huggingface",
                "repo": "org/restricted",
                "split": "train",
                "text_column": "text",
                "label": 1,
                "output": "restricted.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = _write_registry(tmpdir, sources)
            out_path = os.path.join(tmpdir, "report.json")

            ret = main(["--datasets", reg_path, "--output", out_path])
            self.assertEqual(ret, 0)


class TestCLI(unittest.TestCase):
    """Test the CLI argument parser."""

    def test_default_args(self):
        from scripts.license_checker import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.strict)
        self.assertIn("datasets.yaml", args.datasets)
        self.assertIn("license_report.json", args.output)

    def test_custom_args(self):
        from scripts.license_checker import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--datasets", "/tmp/ds.yaml",
            "--output", "/tmp/out.json",
            "--strict",
        ])
        self.assertEqual(args.datasets, "/tmp/ds.yaml")
        self.assertEqual(args.output, "/tmp/out.json")
        self.assertTrue(args.strict)

    @mock.patch("scripts.license_checker.parse_hf_registry", return_value=[])
    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_main_returns_0_for_empty_registry(self, mock_fetch, _mock_hf_reg):
        from scripts.license_checker import main

        sources = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = _write_registry(tmpdir, sources)
            out_path = os.path.join(tmpdir, "report.json")

            ret = main(["--datasets", reg_path, "--output", out_path])
            self.assertEqual(ret, 0)


class TestLoadCachedLicenses(unittest.TestCase):
    """Test extraction of cached license info from datasets.yaml."""

    def test_extracts_license_field(self):
        from scripts.license_checker import load_cached_licenses

        sources = {
            "ds_with_license": {
                "type": "huggingface",
                "repo": "org/ds1",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "ds1.csv",
                "license": "mit",
            },
            "ds_without_license": {
                "type": "huggingface",
                "repo": "org/ds2",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "ds2.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            cache = load_cached_licenses(path)

        self.assertEqual(cache.get("org/ds1"), "mit")
        self.assertNotIn("org/ds2", cache)

    def test_skips_non_hf_sources(self):
        from scripts.license_checker import load_cached_licenses

        sources = {
            "github_src": {
                "type": "github_csv",
                "url": "https://example.com/data.csv",
                "text_column": "prompt",
                "label": 1,
                "output": "gh.csv",
                "license": "mit",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            cache = load_cached_licenses(path)

        self.assertEqual(len(cache), 0)


class TestParseHFRegistry(unittest.TestCase):
    """Test the parse_hf_registry() function."""

    def test_returns_list(self):
        from scripts.license_checker import parse_hf_registry

        results = parse_hf_registry()
        self.assertIsInstance(results, list)

    def test_entries_are_tuples(self):
        from scripts.license_checker import parse_hf_registry

        results = parse_hf_registry()
        self.assertGreater(len(results), 0)
        for entry in results:
            self.assertEqual(len(entry), 3)

    def test_hf_ids_are_strings(self):
        from scripts.license_checker import parse_hf_registry

        results = parse_hf_registry()
        for name, hf_id, spec in results:
            self.assertIsInstance(hf_id, str)
            self.assertTrue(len(hf_id) > 0)


class TestGenerateReportWithHFRegistry(unittest.TestCase):
    """Test that generate_report merges HF registry entries."""

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_includes_hf_registry_entries(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = "apache-2.0"

        sources = {
            "alpaca": {
                "type": "huggingface",
                "repo": "tatsu-lab/alpaca",
                "split": "train",
                "text_column": "instruction",
                "label": 0,
                "output": "hf_alpaca.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=True)

        # Should have the YAML entry plus HF registry entries
        repos = {e["repo"] for e in report}
        self.assertIn("tatsu-lab/alpaca", repos)
        # At least some HF registry entries should be included
        self.assertGreater(len(report), 1)

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_no_duplicates_between_yaml_and_registry(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = "apache-2.0"

        # Use a repo that exists in HF registry
        sources = {
            "squad_yaml": {
                "type": "huggingface",
                "repo": "squad",
                "split": "train",
                "text_column": "question",
                "label": 0,
                "output": "hf_squad.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=True)

        # "squad" should only appear once
        squad_entries = [e for e in report if e["repo"] == "squad"]
        self.assertEqual(len(squad_entries), 1)

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_exclude_hf_registry(self, mock_fetch):
        from scripts.license_checker import generate_report

        mock_fetch.return_value = "mit"

        sources = {
            "ds1": {
                "type": "huggingface",
                "repo": "org/ds1",
                "split": "train",
                "text_column": "text",
                "label": 0,
                "output": "ds1.csv",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=False)

        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["repo"], "org/ds1")

    @mock.patch("scripts.license_checker.fetch_license_from_hub")
    def test_hf_registry_license_used_as_fallback(self, mock_fetch):
        from scripts.license_checker import generate_report

        # Simulate API being unreachable
        mock_fetch.return_value = None

        # Empty YAML, rely on HF registry
        sources = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_registry(tmpdir, sources)
            report = generate_report(path, include_hf_registry=True)

        # HF registry entries that have licenses should use them as cache
        for entry in report:
            if entry["license"] is not None:
                self.assertEqual(entry["license_source"], "cache")


if __name__ == "__main__":
    unittest.main()
