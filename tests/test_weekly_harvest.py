"""Tests for scripts/weekly_harvest.py.

Validates the weekly harvest pipeline: HuggingFace scanning, arXiv parsing,
GitHub scanning, known-datasets registry, run_harvest orchestration, CLI
parsing, and output format -- all without making real network requests.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


# ===================================================================
# Helper: build a fake arXiv Atom XML response
# ===================================================================

_ARXIV_XML_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  {entries}
</feed>
"""

_ARXIV_ENTRY_TEMPLATE = """\
<entry>
  <id>{arxiv_id}</id>
  <title>{title}</title>
  <summary>{abstract}</summary>
  <published>{published}</published>
</entry>
"""


def _make_arxiv_xml(entries):
    """Build a fake arXiv Atom XML string from a list of entry dicts."""
    entry_xml = ""
    for e in entries:
        entry_xml += _ARXIV_ENTRY_TEMPLATE.format(**e)
    return _ARXIV_XML_TEMPLATE.format(entries=entry_xml)


def _make_http_response(json_data=None, content=None, status_code=200, headers=None):
    """Build a mock requests.Response object."""
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
    if content is not None:
        resp.content = content
    resp.raise_for_status.return_value = None
    return resp


# ===================================================================
# 1. Import tests
# ===================================================================

class TestImport(unittest.TestCase):
    """The module should import without side effects."""

    def test_import(self):
        """Module imports without error."""
        import weekly_harvest as mod
        self.assertIsNotNone(mod)

    def test_has_main_functions(self):
        """Key functions exist on the module."""
        import weekly_harvest as mod
        for attr in (
            "scan_huggingface",
            "scan_arxiv",
            "scan_github",
            "load_known_datasets",
            "save_known_datasets",
            "run_harvest",
            "build_parser",
        ):
            self.assertTrue(hasattr(mod, attr), f"Missing attribute: {attr}")


# ===================================================================
# 2. HuggingFace scanner tests
# ===================================================================

class TestScanHuggingFace(unittest.TestCase):
    """HuggingFace dataset scanner with mocked HTTP."""

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_returns_list(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        mock_get.return_value = _make_http_response(json_data=[
            {
                "id": "user/prompt-injection-v1",
                "lastModified": "2026-02-27T12:00:00Z",
                "description": "A prompt injection dataset",
            },
        ])

        results = scan_huggingface(
            queries=["prompt injection"], since_days=30,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_filters_known(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        mock_get.return_value = _make_http_response(json_data=[
            {"id": "user/known-ds", "lastModified": "2026-02-27T12:00:00Z"},
            {"id": "user/new-ds", "lastModified": "2026-02-27T12:00:00Z"},
        ])

        results = scan_huggingface(
            queries=["prompt injection"],
            since_days=30,
            known_ids={"user/known-ds"},
        )
        ids = [r["id"] for r in results]
        self.assertNotIn("user/known-ds", ids)
        self.assertIn("user/new-ds", ids)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_handles_empty_response(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        mock_get.return_value = _make_http_response(json_data=[])

        results = scan_huggingface(
            queries=["prompt injection"], since_days=7,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_handles_network_error(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        mock_get.side_effect = Exception("Connection refused")

        results = scan_huggingface(
            queries=["prompt injection"], since_days=7,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_multiple_queries(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            return _make_http_response(json_data=[
                {
                    "id": f"user/ds-{call_count['n']}",
                    "lastModified": "2026-02-27T12:00:00Z",
                },
            ])

        mock_get.side_effect = side_effect

        queries = ["prompt injection", "jailbreak dataset", "LLM safety"]
        results = scan_huggingface(queries=queries, since_days=30)

        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(len(results), 3)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_hf_result_schema(self, mock_get, mock_sleep):
        from weekly_harvest import scan_huggingface

        mock_get.return_value = _make_http_response(json_data=[
            {
                "id": "org/dataset-name",
                "lastModified": "2026-02-27T12:00:00Z",
                "description": "Some description",
            },
        ])

        results = scan_huggingface(queries=["test"], since_days=30)
        self.assertEqual(len(results), 1)

        result = results[0]
        required_fields = {"id", "source", "url"}
        for field in required_fields:
            self.assertIn(field, result, f"Result missing field: {field}")

        self.assertEqual(result["source"], "huggingface")
        self.assertIn("huggingface.co", result["url"])


# ===================================================================
# 3. arXiv scanner tests
# ===================================================================

class TestScanArxiv(unittest.TestCase):
    """arXiv paper scanner with mocked HTTP."""

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_parses_xml(self, mock_get, mock_sleep):
        from weekly_harvest import scan_arxiv

        xml = _make_arxiv_xml([
            {
                "arxiv_id": "http://arxiv.org/abs/2402.12345",
                "title": "Prompt Injection Attacks",
                "abstract": "We study prompt injection attacks on LLMs.",
                "published": "2026-02-25T00:00:00Z",
            },
        ])
        mock_get.return_value = _make_http_response(
            content=xml.encode("utf-8"),
        )

        results = scan_arxiv(queries=["prompt+injection"], since_days=30)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIn("Prompt Injection Attacks", results[0]["description"])

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_extracts_github_urls(self, mock_get, mock_sleep):
        from weekly_harvest import scan_arxiv

        xml = _make_arxiv_xml([
            {
                "arxiv_id": "http://arxiv.org/abs/2402.99999",
                "title": "Paper with GitHub link",
                "abstract": "Our code is at https://github.com/user/repo and more text.",
                "published": "2026-02-25T00:00:00Z",
            },
        ])
        mock_get.return_value = _make_http_response(
            content=xml.encode("utf-8"),
        )

        results = scan_arxiv(queries=["test"], since_days=30)
        self.assertEqual(len(results), 1)
        self.assertIn("https://github.com/user/repo", results[0]["github_urls"])

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_extracts_hf_urls(self, mock_get, mock_sleep):
        from weekly_harvest import scan_arxiv

        xml = _make_arxiv_xml([
            {
                "arxiv_id": "http://arxiv.org/abs/2402.11111",
                "title": "Paper with HF link",
                "abstract": "Dataset at https://huggingface.co/datasets/org-ds for details.",
                "published": "2026-02-25T00:00:00Z",
            },
        ])
        mock_get.return_value = _make_http_response(
            content=xml.encode("utf-8"),
        )

        results = scan_arxiv(queries=["test"], since_days=30)
        self.assertEqual(len(results), 1)
        self.assertIn(
            "https://huggingface.co/datasets/org-ds",
            results[0]["hf_urls"],
        )

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_handles_empty(self, mock_get, mock_sleep):
        from weekly_harvest import scan_arxiv

        xml = _make_arxiv_xml([])
        mock_get.return_value = _make_http_response(
            content=xml.encode("utf-8"),
        )

        results = scan_arxiv(queries=["prompt+injection"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_handles_network_error(self, mock_get, mock_sleep):
        from weekly_harvest import scan_arxiv

        mock_get.side_effect = Exception("Timeout")

        results = scan_arxiv(queries=["prompt+injection"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)


# ===================================================================
# 4. GitHub scanner tests
# ===================================================================

class TestScanGitHub(unittest.TestCase):
    """GitHub repo scanner with mocked HTTP."""

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_returns_list(self, mock_get, mock_sleep):
        from weekly_harvest import scan_github

        mock_get.return_value = _make_http_response(json_data={
            "items": [
                {
                    "full_name": "user/prompt-injection-data",
                    "html_url": "https://github.com/user/prompt-injection-data",
                    "description": "A collection of prompt injection samples",
                    "stargazers_count": 42,
                    "pushed_at": "2026-02-27T10:00:00Z",
                    "updated_at": "2026-02-27T10:00:00Z",
                    "topics": [],
                },
            ],
        })

        results = scan_github(queries=["prompt+injection+dataset"], since_days=30)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_handles_empty(self, mock_get, mock_sleep):
        from weekly_harvest import scan_github

        mock_get.return_value = _make_http_response(json_data={"items": []})

        results = scan_github(queries=["prompt+injection+dataset"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_handles_rate_limit(self, mock_get, mock_sleep):
        """_http_get raises on rate limit after retries; scanner catches it."""
        from weekly_harvest import scan_github

        mock_get.side_effect = Exception("Rate limited (HTTP 403)")

        results = scan_github(queries=["prompt+injection+dataset"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_handles_429_rate_limit(self, mock_get, mock_sleep):
        """HTTP 429 rate limit also results in graceful empty return."""
        from weekly_harvest import scan_github

        mock_get.side_effect = Exception("Rate limited (HTTP 429)")

        results = scan_github(queries=["prompt+injection+dataset"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_handles_network_error(self, mock_get, mock_sleep):
        from weekly_harvest import scan_github

        mock_get.side_effect = Exception("DNS resolution failed")

        results = scan_github(queries=["prompt+injection+dataset"], since_days=7)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_result_schema(self, mock_get, mock_sleep):
        from weekly_harvest import scan_github

        mock_get.return_value = _make_http_response(json_data={
            "items": [
                {
                    "full_name": "org/repo",
                    "html_url": "https://github.com/org/repo",
                    "description": "desc",
                    "stargazers_count": 10,
                    "pushed_at": "2026-02-27T10:00:00Z",
                    "updated_at": "2026-02-27T10:00:00Z",
                    "topics": ["llm", "security"],
                },
            ],
        })

        results = scan_github(queries=["test"], since_days=30)
        self.assertEqual(len(results), 1)

        result = results[0]
        required_fields = {"id", "source", "url", "description"}
        for field in required_fields:
            self.assertIn(field, result, f"Result missing field: {field}")

        self.assertEqual(result["source"], "github")


# ===================================================================
# 5. Known datasets registry tests
# ===================================================================

class TestKnownDatasets(unittest.TestCase):
    """Known-datasets file I/O."""

    def test_load_known_empty_file(self):
        from weekly_harvest import load_known_datasets

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            path = f.name

        try:
            ids = load_known_datasets(path)
            self.assertIsInstance(ids, set)
            self.assertEqual(len(ids), 0)
        finally:
            os.unlink(path)

    def test_load_known_with_entries(self):
        from weekly_harvest import load_known_datasets

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("user/dataset-a\n")
            f.write("org/dataset-b\n")
            f.write("another/dataset-c\n")
            path = f.name

        try:
            ids = load_known_datasets(path)
            self.assertEqual(ids, {
                "user/dataset-a", "org/dataset-b", "another/dataset-c",
            })
        finally:
            os.unlink(path)

    def test_load_known_skips_comments(self):
        from weekly_harvest import load_known_datasets

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("# This is a comment\n")
            f.write("user/real-dataset\n")
            f.write("# Another comment\n")
            f.write("\n")
            f.write("org/another-dataset\n")
            path = f.name

        try:
            ids = load_known_datasets(path)
            self.assertEqual(ids, {"user/real-dataset", "org/another-dataset"})
        finally:
            os.unlink(path)

    def test_load_known_missing_file(self):
        from weekly_harvest import load_known_datasets

        ids = load_known_datasets("/tmp/nonexistent_harvest_file_12345.txt")
        self.assertIsInstance(ids, set)
        self.assertEqual(len(ids), 0)

    def test_save_known_creates_file(self):
        from weekly_harvest import save_known_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "known.txt")
            save_known_datasets(path, {"user/ds-1", "org/ds-2"})

            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("user/ds-1", content)
            self.assertIn("org/ds-2", content)

    def test_save_known_roundtrip(self):
        from weekly_harvest import load_known_datasets, save_known_datasets

        original = {"user/alpha", "org/beta", "team/gamma"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.txt")
            save_known_datasets(path, original)
            loaded = load_known_datasets(path)
            self.assertEqual(loaded, original)


# ===================================================================
# 6. run_harvest integration tests
# ===================================================================

class TestRunHarvest(unittest.TestCase):
    """Integration tests for the harvest orchestrator."""

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[])
    def test_harvest_creates_output_dir(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "harvest_output")
            self.assertFalse(os.path.exists(out))

            run_harvest(output_dir=out, since_days=7)
            self.assertTrue(os.path.isdir(out))

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds1", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds1",
         "description": "test", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_harvest_writes_latest_scan(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            run_harvest(output_dir=tmpdir, since_days=7)

            latest_path = os.path.join(tmpdir, "latest_scan.json")
            self.assertTrue(os.path.exists(latest_path))

            with open(latest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("scan_time", data)
            self.assertIn("discoveries", data)

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds1", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds1",
         "description": "test1", "last_modified": "2026-02-27T00:00:00Z"},
        {"id": "user/ds2", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds2",
         "description": "test2", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_harvest_writes_new_datasets(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            run_harvest(output_dir=tmpdir, since_days=7)

            new_path = os.path.join(tmpdir, "new_datasets.jsonl")
            self.assertTrue(os.path.exists(new_path))

            with open(new_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds1", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds1",
         "description": "test", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_harvest_dry_run(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "dry_run_output")
            summary = run_harvest(output_dir=out, since_days=7, dry_run=True)

            self.assertIsInstance(summary, dict)
            self.assertEqual(summary["total_discovered"], 1)
            self.assertFalse(
                os.path.exists(os.path.join(out, "latest_scan.json")),
            )
            self.assertFalse(
                os.path.exists(os.path.join(out, "new_datasets.jsonl")),
            )


# ===================================================================
# 7. CLI parser tests
# ===================================================================

class TestBuildParser(unittest.TestCase):
    """CLI argument parser."""

    def test_parser_defaults(self):
        from weekly_harvest import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, "data/harvest")
        self.assertEqual(args.since_days, 7)
        self.assertEqual(args.sources, "hf,arxiv,github")
        self.assertFalse(args.dry_run)

    def test_parser_since_days(self):
        from weekly_harvest import build_parser

        parser = build_parser()
        args = parser.parse_args(["--since-days", "30"])
        self.assertEqual(args.since_days, 30)

    def test_parser_sources_filter(self):
        from weekly_harvest import build_parser

        parser = build_parser()
        args = parser.parse_args(["--sources", "hf,arxiv"])
        self.assertEqual(args.sources, "hf,arxiv")


# ===================================================================
# 8. Output format tests
# ===================================================================

class TestOutputFormat(unittest.TestCase):
    """Verify output file schemas and validity."""

    @mock.patch("weekly_harvest.scan_github", return_value=[
        {"id": "org/repo", "source": "github",
         "url": "https://github.com/org/repo",
         "description": "desc", "stars": 5,
         "updated_at": "2026-02-27T10:00:00Z"},
    ])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[
        {"id": "http://arxiv.org/abs/2402.00001", "source": "arxiv",
         "url": "http://arxiv.org/abs/2402.00001",
         "description": "Paper Title", "summary": "text",
         "published": "2026-02-25T00:00:00Z",
         "github_urls": [], "hf_urls": []},
    ])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds", "source": "huggingface",
         "url": "https://huggingface.co/datasets/user/ds",
         "description": "desc", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_latest_scan_json_schema(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            run_harvest(output_dir=tmpdir, since_days=7)

            latest_path = os.path.join(tmpdir, "latest_scan.json")
            with open(latest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            expected_keys = {"scan_time", "per_source_counts", "discoveries"}
            for key in expected_keys:
                self.assertIn(key, data, f"latest_scan.json missing key: {key}")

            self.assertIsInstance(data["per_source_counts"], dict)
            self.assertIsInstance(data["discoveries"], list)
            self.assertEqual(len(data["discoveries"]), 3)

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds1", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds1",
         "description": "a", "last_modified": "2026-02-27T00:00:00Z"},
        {"id": "user/ds2", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds2",
         "description": "b", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_new_datasets_jsonl_valid(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            run_harvest(output_dir=tmpdir, since_days=7)

            new_path = os.path.join(tmpdir, "new_datasets.jsonl")
            with open(new_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            self.assertGreater(len(lines), 0)
            for line in lines:
                obj = json.loads(line.strip())
                self.assertIsInstance(obj, dict)
                self.assertIn("id", obj)
                self.assertIn("source", obj)

    @mock.patch("weekly_harvest.scan_github", return_value=[])
    @mock.patch("weekly_harvest.scan_arxiv", return_value=[])
    @mock.patch("weekly_harvest.scan_huggingface", return_value=[
        {"id": "user/ds1", "source": "huggingface",
         "url": "https://hf.co/datasets/user/ds1",
         "description": "a", "last_modified": "2026-02-27T00:00:00Z"},
    ])
    def test_scan_history_updated(self, mock_hf, mock_arxiv, mock_gh):
        from weekly_harvest import run_harvest

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run harvest twice
            run_harvest(output_dir=tmpdir, since_days=7)
            run_harvest(output_dir=tmpdir, since_days=7)

            history_path = os.path.join(tmpdir, "scan_history.json")
            self.assertTrue(os.path.exists(history_path))

            with open(history_path, "r", encoding="utf-8") as fh:
                history = json.load(fh)

            self.assertIsInstance(history, list)
            self.assertEqual(len(history), 2)
            for entry in history:
                self.assertIn("scan_time", entry)
                self.assertIn("per_source_counts", entry)
                self.assertIn("total_discovered", entry)


if __name__ == "__main__":
    unittest.main()
