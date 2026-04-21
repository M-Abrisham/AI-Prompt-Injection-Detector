"""Scraper/registry drift regression test.

Guards against the bug where scripts/social_scraper.py hardcoded its own
regex patterns that silently diverged from the canonical Layer 1 rules.
If a new rule lands in rules_registry.py at critical/high severity but the
scraper doesn't pick it up, this test fails.
"""

import unittest

from na0s.layer1.rules_registry import RULES

from scripts.social_scraper import INJECTION_PATTERNS


_STRONG_SEVERITIES = ("critical_content", "critical", "high")


class TestScraperCoverage(unittest.TestCase):

    def test_injection_patterns_is_list(self):
        self.assertIsInstance(INJECTION_PATTERNS, list)
        self.assertGreater(len(INJECTION_PATTERNS), 0)

    def test_every_strong_rule_is_scraped(self):
        expected = {
            r._compiled.pattern for r in RULES if r.severity in _STRONG_SEVERITIES
        }
        actual = {p.pattern for p in INJECTION_PATTERNS}
        missing = expected - actual
        self.assertFalse(
            missing,
            "Scraper INJECTION_PATTERNS missing {} Layer 1 strong rules:\n{}".format(
                len(missing), "\n".join(sorted(missing))
            ),
        )

    def test_scraper_not_using_stale_hardcoded_patterns(self):
        expected_count = sum(
            1 for r in RULES if r.severity in _STRONG_SEVERITIES
        )
        self.assertEqual(
            len(INJECTION_PATTERNS),
            expected_count,
            "INJECTION_PATTERNS length differs from strong-rule count — "
            "likely a hardcoded subset was re-introduced",
        )


if __name__ == "__main__":
    unittest.main()
