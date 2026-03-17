"""Layer 12 — Per-probe validation tests.

Parameterized tests that verify every auto-discovered Probe in
ALL_PROBES conforms to the Probe contract: non-empty generate(),
valid category_id, correct tuple structure, and technique_id prefixing.
"""

import os
import unittest

os.environ["SCAN_TIMEOUT_SEC"] = "0"

from scripts.taxonomy import ALL_PROBES
from scripts.taxonomy._base import Probe

# Stale skip guard removed 2026-03-17: ContextOverflowProbe generate()
# was fixed and now returns 208 valid 3-tuples.
_KNOWN_BROKEN_GENERATE = frozenset()


def _safe_generate(test_case, cls):
    """Call probe.generate(), skipping probes with known bugs."""
    if cls.__name__ in _KNOWN_BROKEN_GENERATE:
        test_case.skipTest(
            f"{cls.__name__} has a known bug in generate() — skipped"
        )
    probe = cls()
    return probe, probe.generate()


class TestProbeDiscovery(unittest.TestCase):
    """Verify the auto-discovery mechanism found a reasonable number of probes."""

    def test_all_probes_nonempty(self):
        self.assertGreaterEqual(len(ALL_PROBES), 23, "Expected at least 23 probes")

    def test_all_probes_are_probe_subclasses(self):
        for cls in ALL_PROBES:
            self.assertTrue(
                issubclass(cls, Probe),
                f"{cls.__name__} is not a Probe subclass",
            )

    def test_no_duplicate_category_ids(self):
        ids = [cls.category_id for cls in ALL_PROBES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate category_ids found")


class TestProbeGenerateNotEmpty(unittest.TestCase):
    """Each probe's generate() must return a non-empty list."""

    def test_generate_not_empty(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe, samples = _safe_generate(self, cls)
                self.assertIsInstance(samples, (list, tuple))
                self.assertGreater(
                    len(samples), 0,
                    f"{cls.__name__}.generate() returned empty list",
                )


class TestProbeHasCategoryId(unittest.TestCase):
    """Each probe must have a set and non-empty category_id."""

    def test_category_id_set(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                self.assertIsInstance(cls.category_id, str)
                self.assertTrue(
                    cls.category_id,
                    f"{cls.__name__}.category_id is empty",
                )


class TestProbeSamplesAreTuples(unittest.TestCase):
    """All samples must be (text, technique_id) or (text, technique_id, metadata) tuples."""

    def test_samples_are_tuples(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe, samples = _safe_generate(self, cls)
                for i, item in enumerate(samples):
                    self.assertIsInstance(
                        item, (list, tuple),
                        f"{cls.__name__} sample [{i}] is not a tuple: {type(item)}",
                    )
                    self.assertIn(
                        len(item), (2, 3),
                        f"{cls.__name__} sample [{i}] has {len(item)} elements, "
                        f"expected 2 or 3",
                    )
                    # text must be str
                    self.assertIsInstance(
                        item[0], str,
                        f"{cls.__name__} sample [{i}] text is not str",
                    )
                    # technique_id must be str
                    self.assertIsInstance(
                        item[1], str,
                        f"{cls.__name__} sample [{i}] technique_id is not str",
                    )
                    # metadata (if present) must be dict
                    if len(item) == 3:
                        self.assertIsInstance(
                            item[2], dict,
                            f"{cls.__name__} sample [{i}] metadata is not dict",
                        )


class TestProbeSamplesHaveTechniqueId(unittest.TestCase):
    """All samples must have a non-empty technique_id."""

    def test_technique_ids_nonempty(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe, samples = _safe_generate(self, cls)
                for i, item in enumerate(samples):
                    tech_id = item[1]
                    self.assertTrue(
                        tech_id,
                        f"{cls.__name__} sample [{i}] has empty technique_id",
                    )


class TestProbeTechniqueIdsMatchCategory(unittest.TestCase):
    """technique_ids should start with the probe's category_id prefix."""

    def test_technique_ids_match_category(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe, samples = _safe_generate(self, cls)
                cat_id = cls.category_id
                for i, item in enumerate(samples):
                    tech_id = item[1]
                    self.assertTrue(
                        tech_id.startswith(cat_id),
                        f"{cls.__name__} sample [{i}]: technique_id '{tech_id}' "
                        f"does not start with category_id '{cat_id}'",
                    )


class TestProbeHasBenignSamples(unittest.TestCase):
    """Probes should include benign samples (technique_id ending with '_benign')."""

    def test_has_benign_samples(self):
        probes_with_benign = 0
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                if cls.__name__ in _KNOWN_BROKEN_GENERATE:
                    continue
                probe = cls()
                samples = probe.generate()
                benign = [s for s in samples if s[1].endswith("_benign")]
                if benign:
                    probes_with_benign += 1
        # At least some probes should have benign samples
        self.assertGreater(
            probes_with_benign, 0,
            "No probes have benign samples at all",
        )


class TestProbeValidatedSamples(unittest.TestCase):
    """The _validated_samples() method should succeed for all probes."""

    def test_validated_samples_succeeds(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                if cls.__name__ in _KNOWN_BROKEN_GENERATE:
                    self.skipTest(f"{cls.__name__} has a known bug — skipped")
                probe = cls()
                normalized = probe._validated_samples()
                self.assertIsInstance(normalized, list)
                self.assertGreater(len(normalized), 0)
                # All normalized to 3-tuples
                for item in normalized:
                    self.assertEqual(len(item), 3)
                    self.assertIsInstance(item[0], str)
                    self.assertIsInstance(item[1], str)
                    self.assertIsInstance(item[2], dict)


class TestProbeTextContent(unittest.TestCase):
    """Sample text should be non-empty strings."""

    def test_sample_text_nonempty(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe, samples = _safe_generate(self, cls)
                for i, item in enumerate(samples):
                    self.assertTrue(
                        item[0].strip(),
                        f"{cls.__name__} sample [{i}] has empty/whitespace-only text",
                    )


class TestProbeInstantiation(unittest.TestCase):
    """Every probe class should be instantiable without arguments."""

    def test_instantiation(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe = cls()
                self.assertIsInstance(probe, Probe)
                self.assertTrue(probe.name, f"{cls.__name__} has empty name")


class TestProbeTaxonomyMetadata(unittest.TestCase):
    """Probes should have taxonomy metadata loaded from YAML."""

    def test_has_severity(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe = cls()
                # severity should be set (may be empty for some)
                self.assertIsInstance(probe.severity, str)

    def test_has_tags(self):
        for cls in ALL_PROBES:
            with self.subTest(probe=cls.__name__):
                probe = cls()
                self.assertIsInstance(probe.tags, list)


if __name__ == "__main__":
    unittest.main()
