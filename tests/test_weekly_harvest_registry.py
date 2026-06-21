"""Decontam-at-discovery + unified known-corpus registry for weekly_harvest.

Two confirmed gaps, pinned here (all offline; HTTP fully mocked):

  GAP 2 — only HuggingFace IDs were persisted to known_datasets.txt, so every
          weekly run re-surfaced the same arXiv papers and GitHub repos.
          Now ALL discovered IDs (HF + arXiv + GitHub) are persisted.

  GAP 3 — scan_arxiv / scan_github took no `known_ids`, so they never consulted
          the shared registry.  Now both filter against the one shared index a
          crawler-agnostic known-corpus set), so a dataset found by one source
          isn't re-discovered by another / on the next run.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _recent_iso(days_ago=1):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


_RECENT = _recent_iso(1)

_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>"""

_ARXIV_ENTRY = """<entry>
  <id>{arxiv_id}</id>
  <title>{title}</title>
  <summary>{abstract}</summary>
  <published>{published}</published>
</entry>"""


def _arxiv_xml(entries):
    return _ARXIV_XML.format(
        entries="".join(_ARXIV_ENTRY.format(**e) for e in entries)
    )


def _resp(json_data=None, content=None):
    r = mock.MagicMock()
    r.status_code = 200
    r.headers = {}
    if json_data is not None:
        r.json.return_value = json_data
    if content is not None:
        r.content = content
    r.raise_for_status.return_value = None
    return r


# ── GAP 3: scanners consult the shared known_ids registry ───────────────────

class TestDiscoveryDecontam(unittest.TestCase):

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_arxiv_skips_known(self, mock_get, _sleep):
        from weekly_harvest import scan_arxiv

        xml = _arxiv_xml([
            {"arxiv_id": "http://arxiv.org/abs/2402.00001",
             "title": "Known paper", "abstract": "x", "published": _RECENT},
            {"arxiv_id": "http://arxiv.org/abs/2402.00002",
             "title": "New paper", "abstract": "y", "published": _RECENT},
        ])
        mock_get.return_value = _resp(content=xml.encode("utf-8"))

        results = scan_arxiv(
            queries=["test"], since_days=30,
            known_ids={"http://arxiv.org/abs/2402.00001"},
        )
        ids = [r["id"] for r in results]
        self.assertNotIn("http://arxiv.org/abs/2402.00001", ids)
        self.assertIn("http://arxiv.org/abs/2402.00002", ids)

    @mock.patch("weekly_harvest.time.sleep")
    @mock.patch("weekly_harvest._http_get")
    def test_scan_github_skips_known(self, mock_get, _sleep):
        from weekly_harvest import scan_github

        mock_get.return_value = _resp(json_data={"items": [
            {"full_name": "org/known-repo",
             "html_url": "https://github.com/org/known-repo",
             "description": "d", "stargazers_count": 1,
             "updated_at": _RECENT, "topics": []},
            {"full_name": "org/new-repo",
             "html_url": "https://github.com/org/new-repo",
             "description": "d", "stargazers_count": 1,
             "updated_at": _RECENT, "topics": []},
        ]})

        results = scan_github(
            queries=["test"], since_days=30,
            known_ids={"org/known-repo"},
        )
        ids = [r["id"] for r in results]
        self.assertNotIn("org/known-repo", ids)
        self.assertIn("org/new-repo", ids)

    def test_scanners_accept_known_ids_kwarg(self):
        import inspect
        import weekly_harvest as wh
        for fn in (wh.scan_arxiv, wh.scan_github, wh.scan_huggingface):
            self.assertIn(
                "known_ids", inspect.signature(fn).parameters,
                f"{fn.__name__} must accept known_ids",
            )


# ── GAP 2: ALL discovered IDs persist to the unified registry ───────────────

class TestUnifiedRegistryPersistence(unittest.TestCase):

    def _run(self, tmpdir):
        from weekly_harvest import run_harvest, load_known_datasets

        hf = [{"id": "user/hf-ds", "source": "huggingface",
               "url": "https://hf.co/datasets/user/hf-ds",
               "description": "d", "last_modified": _RECENT}]
        arxiv = [{"id": "http://arxiv.org/abs/2402.55555", "source": "arxiv",
                  "url": "http://arxiv.org/abs/2402.55555",
                  "description": "Paper", "summary": "s",
                  "published": _RECENT, "github_urls": [], "hf_urls": []}]
        gh = [{"id": "org/gh-repo", "source": "github",
               "url": "https://github.com/org/gh-repo",
               "description": "d", "stars": 1, "updated_at": _RECENT}]

        with mock.patch("weekly_harvest.scan_huggingface", return_value=hf), \
             mock.patch("weekly_harvest.scan_arxiv", return_value=arxiv), \
             mock.patch("weekly_harvest.scan_github", return_value=gh):
            run_harvest(output_dir=tmpdir, since_days=30)

        return load_known_datasets(os.path.join(tmpdir, "known_datasets.txt"))

    def test_all_sources_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            known = self._run(tmpdir)
        # The HF gap-fix regression: arXiv + GitHub IDs must now be persisted.
        self.assertIn("user/hf-ds", known)
        self.assertIn("http://arxiv.org/abs/2402.55555", known)
        self.assertIn("org/gh-repo", known)

    def test_persisted_ids_not_resurfaced_next_run(self):
        """End-to-end: a second run with the same discoveries yields nothing
        new, because the first run persisted all IDs to the shared registry
        and the scanners now consult it."""
        from weekly_harvest import scan_arxiv, scan_github, scan_huggingface

        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: persist everything.
            self._run(tmpdir)

            # Real scanners (HTTP mocked) returning the SAME items; they must
            # be filtered out by the now-populated registry.
            from weekly_harvest import run_harvest, load_known_datasets
            known_before = load_known_datasets(
                os.path.join(tmpdir, "known_datasets.txt")
            )

            arxiv_xml = _arxiv_xml([
                {"arxiv_id": "http://arxiv.org/abs/2402.55555",
                 "title": "Paper", "abstract": "s", "published": _RECENT},
            ])
            gh_json = {"items": [
                {"full_name": "org/gh-repo",
                 "html_url": "https://github.com/org/gh-repo",
                 "description": "d", "stargazers_count": 1,
                 "updated_at": _RECENT, "topics": []},
            ]}
            hf_json = [
                {"id": "user/hf-ds", "lastModified": _RECENT,
                 "description": "d"},
            ]

            def _http(url, **kw):
                if "arxiv" in url:
                    return _resp(content=arxiv_xml.encode("utf-8"))
                if "github" in url:
                    return _resp(json_data=gh_json)
                return _resp(json_data=hf_json)

            with mock.patch("weekly_harvest.time.sleep"), \
                 mock.patch("weekly_harvest._http_get", side_effect=_http):
                result = run_harvest(output_dir=tmpdir, since_days=30)

            self.assertEqual(
                result["total_discovered"], 0,
                "previously-known IDs were re-surfaced on the second run",
            )
            # Registry unchanged (nothing new to add).
            known_after = load_known_datasets(
                os.path.join(tmpdir, "known_datasets.txt")
            )
            self.assertEqual(known_before, known_after)


if __name__ == "__main__":
    unittest.main()
