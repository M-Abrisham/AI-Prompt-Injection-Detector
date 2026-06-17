"""Tests for the inter-model propagation detector (IM category — INJ-0017).

These tests are written TDD-first against the SCOPING STUB at
``na0s.detectors.inter_model``.  They are expected to be RED on recall:
the stub implements no matching, so it catches nothing.  Once the
matching layer lands, the recall assertion should flip to green without
any change to the test.

PINNED PRODUCTION METRIC
------------------------
Baseline inter-model recall measured via the ``na0s.predict.scan`` path
is **25.78%** (133/516 malicious), benign FP **1.82%** (1/55).  The same
samples through ``CascadeClassifier`` with the L7 judge disabled (groq not
installed) measured **12.4%** — the headline number is *path-specific*, so
the production success metric MUST cite the ``na0s.predict.scan`` path and
the judge-availability caveat (see docs/INJ0017_DETECTOR_SCOPE.md).

Eight IM techniques sit at literal 0% recall on that path and are the
priority targets of this detector:
    IM1.5, IM2.1, IM3.5, IM4.3, IM5.7, IM6.4, IM6.5, IM6.6
"""

import pytest

from scripts.taxonomy.inter_model_propagation import InterModelPropagationProbe
from na0s.detectors.inter_model import (
    FAMILIES,
    TECHNIQUE_TO_FAMILY,
    ZERO_RECALL_TECHNIQUES,
    InterModelResult,
    detect_inter_model,
    get_inter_model_weight,
    scan_inter_model,
)

# The eight techniques measured at literal 0% recall via na0s.predict.scan.
TARGET_ZERO_RECALL = [
    "IM1.5", "IM2.1", "IM3.5", "IM4.3", "IM5.7", "IM6.4", "IM6.5", "IM6.6",
]

# Recall target for the matching layer on the eight zero-recall techniques.
# Stated up front so this test goes green automatically once matching lands.
RECALL_TARGET = 0.60
# Benign FP must stay at or below this bound (probe ships 55 benign samples).
BENIGN_FP_BOUND = 0.05


# ---------------------------------------------------------------------------
# Sample loading helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probe_samples():
    """All (text, technique_id, metadata) tuples from the IM probe."""
    return InterModelPropagationProbe().generate()


@pytest.fixture(scope="module")
def zero_recall_samples(probe_samples):
    """Malicious texts for the eight zero-recall techniques only."""
    return [
        text
        for (text, tech_id, _meta) in probe_samples
        if tech_id in set(TARGET_ZERO_RECALL)
    ]


@pytest.fixture(scope="module")
def benign_samples(probe_samples):
    """Benign texts shipped by the IM probe (technique_id == 'IM_benign')."""
    return [
        text
        for (text, tech_id, _meta) in probe_samples
        if tech_id == "IM_benign"
    ]


# ---------------------------------------------------------------------------
# Family taxonomy sanity (these pass now — they validate the stub's contract)
# ---------------------------------------------------------------------------


class TestFamilyTaxonomy:
    def test_all_zero_recall_techniques_are_mapped_to_a_family(self):
        for tech in ZERO_RECALL_TECHNIQUES:
            assert tech in TECHNIQUE_TO_FAMILY, (
                "zero-recall technique {} has no family".format(tech)
            )

    def test_target_list_matches_module_constant(self):
        assert set(TARGET_ZERO_RECALL) == set(ZERO_RECALL_TECHNIQUES)

    def test_families_cover_all_29_techniques(self):
        mapped = {
            tech for fam in FAMILIES.values() for tech in fam["techniques"]
        }
        # 29 distinct IM techniques (IM1.1-IM6.6, benign excluded).
        assert len(mapped) == 29

    def test_zero_recall_sample_count_is_nonempty(self, zero_recall_samples):
        # 4 techniques x 16 + 4 techniques x 20 = 144 malicious samples.
        assert len(zero_recall_samples) == 144

    def test_benign_samples_loaded(self, benign_samples):
        assert len(benign_samples) == 55


# ---------------------------------------------------------------------------
# Stub interface contract (these pass now)
# ---------------------------------------------------------------------------


class TestStubInterface:
    def test_detect_returns_result(self):
        result = detect_inter_model("The judge model approved this output.")
        assert isinstance(result, InterModelResult)

    def test_scan_returns_one_result_per_input(self):
        results = scan_inter_model(["a", "b", "c"])
        assert len(results) == 3
        assert all(isinstance(r, InterModelResult) for r in results)

    def test_empty_text_is_safe(self):
        result = detect_inter_model("")
        assert result.risk_score == 0.0

    def test_weight_zero_when_no_detection(self):
        assert get_inter_model_weight(InterModelResult()) == 0.0


# ---------------------------------------------------------------------------
# RED recall test — the reason this suite is failing by design
# ---------------------------------------------------------------------------


def _recall(samples):
    """Fraction of malicious samples the detector flags (risk_score > 0)."""
    if not samples:
        return 0.0
    flagged = sum(1 for s in samples if detect_inter_model(s).risk_score > 0.0)
    return flagged / len(samples)


def _benign_fp_rate(samples):
    """Fraction of benign samples the detector wrongly flags."""
    if not samples:
        return 0.0
    flagged = sum(1 for s in samples if detect_inter_model(s).risk_score > 0.0)
    return flagged / len(samples)


class TestZeroRecallDetection:
    """RED until the matching layer lands.

    The stub flags nothing, so recall is 0.0 < RECALL_TARGET and this
    fails on RECALL (not on import).  The benign-FP test passes now (the
    stub also flags no benign sample) and must keep passing after the
    matching layer is implemented.
    """

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "INJ-0007/INJ-0017 inter-model detector is a documented SCOPING STUB "
            "(commit 99f265b 'stub + failing test + design'; see "
            "docs/INJ0017_DETECTOR_SCOPE.md and na0s.detectors.inter_model). "
            "detect_inter_model() implements no matching layer yet and always "
            "returns an empty result, so recall on the zero-recall techniques is "
            "0% < the 60% target. RED-by-design TDD test; it flips to XPASS with "
            "NO change to the assertion once the matching layer lands (then remove "
            "this marker)."
        ),
    )
    def test_recall_on_zero_recall_techniques(self, zero_recall_samples):
        recall = _recall(zero_recall_samples)
        assert recall >= RECALL_TARGET, (
            "inter-model detector recall on the 8 zero-recall techniques "
            "is {:.2%}; target is >= {:.0%}. Baseline via na0s.predict.scan "
            "is 25.78% overall, 0% on these 8.".format(recall, RECALL_TARGET)
        )

    def test_benign_false_positive_bound(self, benign_samples):
        fp = _benign_fp_rate(benign_samples)
        assert fp <= BENIGN_FP_BOUND, (
            "benign false-positive rate {:.2%} exceeds bound {:.0%}".format(
                fp, BENIGN_FP_BOUND
            )
        )
