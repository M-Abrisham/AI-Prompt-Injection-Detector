"""CI smoke tests -- fast sanity checks that the build is not broken.

These tests verify that all core modules import without error and that
key public interfaces exist and are callable. They are designed to run
in under 1 second and catch import-time crashes, missing dependencies,
and accidental API deletions.
"""

import unittest


class TestCoreImports(unittest.TestCase):
    """Verify all core na0s modules import without error."""

    def test_import_layer0_package(self):
        import na0s.layer0
        self.assertTrue(hasattr(na0s.layer0, "layer0_sanitize"))
        self.assertTrue(hasattr(na0s.layer0, "Layer0Result"))

    def test_import_scan_result(self):
        from na0s.scan_result import ScanResult
        self.assertIsNotNone(ScanResult)

    def test_import_rules(self):
        from na0s import rules
        self.assertIsNotNone(rules)

    def test_import_obfuscation(self):
        from na0s import obfuscation
        self.assertIsNotNone(obfuscation)

    def test_import_cascade(self):
        from na0s import cascade
        self.assertIsNotNone(cascade)

    def test_import_structural_features(self):
        from na0s import structural as structural_features
        self.assertIsNotNone(structural_features)


class TestLayer0Interface(unittest.TestCase):
    """Verify layer0 public API is intact."""

    def test_layer0_sanitize_is_callable(self):
        from na0s.layer0 import layer0_sanitize
        self.assertTrue(callable(layer0_sanitize))

    def test_layer0_sanitize_accepts_string(self):
        from na0s.layer0 import layer0_sanitize
        result = layer0_sanitize("hello world")
        self.assertIsNotNone(result)

    def test_layer0_result_fields(self):
        from na0s.layer0.result import Layer0Result
        r = Layer0Result()
        expected_fields = [
            "sanitized_text",
            "original_length",
            "chars_stripped",
            "anomaly_flags",
            "token_char_ratio",
            "fingerprint",
            "rejected",
            "rejection_reason",
        ]
        for field_name in expected_fields:
            self.assertTrue(
                hasattr(r, field_name),
                f"Layer0Result missing field: {field_name}",
            )


class TestScanResultInterface(unittest.TestCase):
    """Verify ScanResult dataclass has expected fields."""

    def test_scan_result_fields(self):
        from na0s.scan_result import ScanResult
        r = ScanResult()
        expected_fields = [
            "sanitized_text",
            "is_malicious",
            "risk_score",
            "label",
            "technique_tags",
            "rule_hits",
            "ml_confidence",
            "ml_label",
            "anomaly_flags",
            "rejected",
            "rejection_reason",
        ]
        for field_name in expected_fields:
            self.assertTrue(
                hasattr(r, field_name),
                f"ScanResult missing field: {field_name}",
            )

    def test_scan_result_defaults(self):
        from na0s.scan_result import ScanResult
        r = ScanResult()
        self.assertFalse(r.is_malicious)
        self.assertEqual(r.risk_score, 0.0)
        self.assertEqual(r.label, "safe")
        self.assertFalse(r.rejected)


if __name__ == "__main__":
    unittest.main()
