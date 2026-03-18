"""Tests for scripts/data/hf_dataset_registry.py.

Validates the HFDatasetSpec dataclass, registry contents, lookup helpers,
and data-integrity invariants.
"""

import unittest
from dataclasses import fields


class TestHFDatasetSpec(unittest.TestCase):
    """Test the HFDatasetSpec frozen dataclass."""

    def test_is_frozen(self):
        from scripts.data.hf_dataset_registry import HFDatasetSpec

        spec = HFDatasetSpec(
            hf_id="test/dataset",
            config=None,
            split="train",
            text_field="text",
            label_field=None,
            label_map=None,
            technique_id=None,
            license="MIT",
            license_url="https://example.com",
            attribution="test",
            size_hint="1K",
        )
        with self.assertRaises(AttributeError):
            spec.hf_id = "other/id"

    def test_required_fields_present(self):
        from scripts.data.hf_dataset_registry import HFDatasetSpec

        field_names = {f.name for f in fields(HFDatasetSpec)}
        required = {
            "hf_id", "config", "split", "text_field", "label_field",
            "label_map", "technique_id", "license", "license_url",
            "attribution", "size_hint",
        }
        self.assertTrue(required.issubset(field_names))

    def test_default_language(self):
        from scripts.data.hf_dataset_registry import HFDatasetSpec

        spec = HFDatasetSpec(
            hf_id="test/dataset",
            config=None,
            split="train",
            text_field="text",
            label_field=None,
            label_map=None,
            technique_id=None,
            license="MIT",
            license_url="https://example.com",
            attribution="test",
            size_hint="1K",
        )
        self.assertEqual(spec.language, "en")

    def test_default_requires_auth(self):
        from scripts.data.hf_dataset_registry import HFDatasetSpec

        spec = HFDatasetSpec(
            hf_id="test/dataset",
            config=None,
            split="train",
            text_field="text",
            label_field=None,
            label_map=None,
            technique_id=None,
            license="MIT",
            license_url="https://example.com",
            attribution="test",
            size_hint="1K",
        )
        self.assertFalse(spec.requires_auth)

    def test_default_category(self):
        from scripts.data.hf_dataset_registry import HFDatasetSpec

        spec = HFDatasetSpec(
            hf_id="test/dataset",
            config=None,
            split="train",
            text_field="text",
            label_field=None,
            label_map=None,
            technique_id=None,
            license="MIT",
            license_url="https://example.com",
            attribution="test",
            size_hint="1K",
        )
        self.assertEqual(spec.category, "uncategorized")


class TestRegistryContents(unittest.TestCase):
    """Test the DATASET_REGISTRY list itself."""

    def test_registry_has_30_plus_entries(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        self.assertGreaterEqual(len(DATASET_REGISTRY), 30)

    def test_no_duplicate_hf_ids(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        ids = [spec.hf_id for spec in DATASET_REGISTRY]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate hf_ids found")

    def test_all_have_hf_id(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertTrue(spec.hf_id, f"Empty hf_id in registry: {spec}")

    def test_all_have_split(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertTrue(spec.split, f"Empty split for {spec.hf_id}")

    def test_all_have_text_field(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertTrue(spec.text_field, f"Empty text_field for {spec.hf_id}")

    def test_all_have_attribution(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertTrue(spec.attribution, f"Empty attribution for {spec.hf_id}")

    def test_all_have_license(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertIsNotNone(spec.license, f"None license for {spec.hf_id}")

    def test_all_have_license_url(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            self.assertIsNotNone(spec.license_url, f"None license_url for {spec.hf_id}")
            self.assertTrue(
                spec.license_url.startswith("http"),
                f"Invalid license_url for {spec.hf_id}: {spec.license_url}",
            )

    def test_category_is_valid(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        valid_categories = {
            "jailbreak", "red_team", "mixed", "safe_baseline",
            "multilingual", "prompt_injection", "alignment", "uncategorized",
        }
        for spec in DATASET_REGISTRY:
            self.assertIn(
                spec.category, valid_categories,
                f"Invalid category '{spec.category}' for {spec.hf_id}",
            )

    def test_label_map_values_are_int_when_present(self):
        from scripts.data.hf_dataset_registry import DATASET_REGISTRY

        for spec in DATASET_REGISTRY:
            if spec.label_map:
                for key, val in spec.label_map.items():
                    self.assertIsInstance(
                        val, int,
                        f"label_map value {val} for key {key} in {spec.hf_id} is not int",
                    )


class TestGetRegistry(unittest.TestCase):
    """Test the get_registry() function."""

    def test_returns_list(self):
        from scripts.data.hf_dataset_registry import get_registry

        result = get_registry()
        self.assertIsInstance(result, list)

    def test_returns_copy(self):
        from scripts.data.hf_dataset_registry import get_registry, DATASET_REGISTRY

        result = get_registry()
        self.assertEqual(len(result), len(DATASET_REGISTRY))
        # Mutating the copy should not affect the original
        result.append(None)
        self.assertNotEqual(len(result), len(DATASET_REGISTRY))


class TestGetById(unittest.TestCase):
    """Test the get_by_id() function."""

    def test_find_existing(self):
        from scripts.data.hf_dataset_registry import get_by_id

        spec = get_by_id("squad")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.hf_id, "squad")

    def test_returns_none_for_missing(self):
        from scripts.data.hf_dataset_registry import get_by_id

        self.assertIsNone(get_by_id("nonexistent/dataset-that-does-not-exist"))

    def test_exact_match(self):
        from scripts.data.hf_dataset_registry import get_by_id

        # Should not do partial matching
        self.assertIsNone(get_by_id("squ"))


class TestGetByCategory(unittest.TestCase):
    """Test the get_by_category() function."""

    def test_jailbreak_returns_some(self):
        from scripts.data.hf_dataset_registry import get_by_category

        results = get_by_category("jailbreak")
        self.assertGreater(len(results), 0)
        for spec in results:
            self.assertEqual(spec.category, "jailbreak")

    def test_safe_baseline_returns_some(self):
        from scripts.data.hf_dataset_registry import get_by_category

        results = get_by_category("safe_baseline")
        self.assertGreater(len(results), 0)

    def test_empty_for_unknown_category(self):
        from scripts.data.hf_dataset_registry import get_by_category

        results = get_by_category("nonexistent_category")
        self.assertEqual(results, [])

    def test_multilingual_returns_some(self):
        from scripts.data.hf_dataset_registry import get_by_category

        results = get_by_category("multilingual")
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
