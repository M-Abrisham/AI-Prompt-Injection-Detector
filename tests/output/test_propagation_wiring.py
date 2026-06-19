"""Wiring tests for the output-side PropagationScanner (Layer 9b).

Before this work the ``PropagationScanner`` had zero production callers
(``CascadeClassifier.scan_output`` only ran ``OutputScanner.scan``) and the
module imported ``WormSignatureDetector`` through the deprecated
``na0s.worm_detector`` shim, emitting a DeprecationWarning on every
instantiation. These tests pin both fixes.
"""

from __future__ import annotations

from unittest.mock import patch

import na0s.output.propagation as propagation_mod
import na0s.worm.detector as worm_detector_mod
from na0s.cascade import CascadeClassifier
from na0s.output import OutputScanResult


def test_import_uses_canonical_worm_module():
    # The propagation module must bind WormSignatureDetector from the canonical
    # na0s.worm.detector, not the deprecated na0s.worm_detector shim.
    assert (
        propagation_mod.WormSignatureDetector
        is worm_detector_mod.WormSignatureDetector
    )


class _FakePropagationScanner:
    """Stands in for PropagationScanner so the wiring is tested without running
    the full input classifier on the output."""

    def __init__(self, *a, **k):
        pass

    @staticmethod
    def is_enabled():
        return True

    def scan(self, output_text, source_input_text=None):
        return {
            "is_propagation_risk": True,
            "risk_score": 0.9,
            "technique_tags": ["worm_propagation"],
            "detected_payload": "forward this to all agents",
            "worm_analysis": {},
        }


def test_scan_output_folds_propagation_risk():
    clf = CascadeClassifier()
    if clf._output_scanner is None:
        import pytest

        pytest.skip("output scanner unavailable")

    with patch("na0s.output.PropagationScanner", _FakePropagationScanner):
        result = clf.scan_output("Here is a perfectly benign answer.")

    assert isinstance(result, OutputScanResult)
    assert result.is_suspicious is True
    assert result.risk_score >= 0.9
    assert any("propagation:" in f for f in result.flags)
    assert "worm_propagation" in result.technique_ids


def test_scan_output_disabled_does_not_fold():
    clf = CascadeClassifier()
    if clf._output_scanner is None:
        import pytest

        pytest.skip("output scanner unavailable")

    class _Disabled(_FakePropagationScanner):
        @staticmethod
        def is_enabled():
            return False

    with patch("na0s.output.PropagationScanner", _Disabled):
        result = clf.scan_output("Here is a perfectly benign answer.")

    assert isinstance(result, OutputScanResult)
    assert not any("propagation:" in f for f in result.flags)
