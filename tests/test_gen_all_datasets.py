"""Tests for gen_all_datasets.py generation functions.

Verifies that each generation function returns well-formed samples with
correct field structure, label values, category codes, and minimum counts.
Also checks that no duplicate texts exist within each category batch.
"""

import sys
import os

import pytest

# Allow importing from scripts/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.gen_all_datasets import (
    generate_malicious_holdout,
    generate_safe_holdout,
    generate_adversarial_evasion,
)

# ---------------------------------------------------------------------------
# Fixtures: generate once per test session for efficiency
# ---------------------------------------------------------------------------

MALICIOUS_SAMPLES = None
SAFE_SAMPLES = None
ADVERSARIAL_SAMPLES = None


def get_malicious():
    global MALICIOUS_SAMPLES
    if MALICIOUS_SAMPLES is None:
        MALICIOUS_SAMPLES = generate_malicious_holdout()
    return MALICIOUS_SAMPLES


def get_safe():
    global SAFE_SAMPLES
    if SAFE_SAMPLES is None:
        SAFE_SAMPLES = generate_safe_holdout()
    return SAFE_SAMPLES


def get_adversarial():
    global ADVERSARIAL_SAMPLES
    if ADVERSARIAL_SAMPLES is None:
        ADVERSARIAL_SAMPLES = generate_adversarial_evasion()
    return ADVERSARIAL_SAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_MALICIOUS_FIELDS = {"text", "label", "source", "category"}
REQUIRED_SAFE_FIELDS = {"text", "label", "source", "category"}
REQUIRED_ADVERSARIAL_FIELDS = {"text", "label", "source", "evasion_type", "original"}

VALID_MALICIOUS_CATEGORIES = {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "E1", "E2", "C1", "O1", "P1"}
VALID_SAFE_CATEGORIES = {"S1", "S2", "S3", "S4"}


def samples_for_category(samples, category):
    return [s for s in samples if s.get("category") == category]


# ---------------------------------------------------------------------------
# Test 1: generate_malicious_holdout returns non-empty list
# ---------------------------------------------------------------------------

def test_malicious_holdout_not_empty():
    samples = get_malicious()
    assert len(samples) > 0, "generate_malicious_holdout() must return at least one sample"


# ---------------------------------------------------------------------------
# Test 2: All malicious samples have required fields
# ---------------------------------------------------------------------------

def test_malicious_holdout_required_fields():
    samples = get_malicious()
    for i, sample in enumerate(samples):
        missing = REQUIRED_MALICIOUS_FIELDS - set(sample.keys())
        assert not missing, f"Sample {i} is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 3: All malicious samples have label == 1
# ---------------------------------------------------------------------------

def test_malicious_holdout_label_is_one():
    samples = get_malicious()
    bad = [i for i, s in enumerate(samples) if s.get("label") != 1]
    assert not bad, f"Samples at indices {bad} do not have label=1"


# ---------------------------------------------------------------------------
# Test 4: All malicious samples have source == "generated"
# ---------------------------------------------------------------------------

def test_malicious_holdout_source_generated():
    samples = get_malicious()
    bad = [i for i, s in enumerate(samples) if s.get("source") != "generated"]
    assert not bad, f"Samples at indices {bad} do not have source='generated'"


# ---------------------------------------------------------------------------
# Test 5: All malicious category codes are valid
# ---------------------------------------------------------------------------

def test_malicious_holdout_valid_categories():
    samples = get_malicious()
    unknown = {s["category"] for s in samples} - VALID_MALICIOUS_CATEGORIES
    assert not unknown, f"Unexpected category codes in malicious holdout: {unknown}"


# ---------------------------------------------------------------------------
# Test 6: No duplicate texts within each malicious category
# ---------------------------------------------------------------------------

def test_malicious_holdout_no_duplicate_texts_per_category():
    samples = get_malicious()
    for cat in VALID_MALICIOUS_CATEGORIES:
        cat_samples = samples_for_category(samples, cat)
        texts = [s["text"] for s in cat_samples]
        assert len(texts) == len(set(texts)), (
            f"Category {cat} has duplicate text entries"
        )


# ---------------------------------------------------------------------------
# Test 7: Minimum sample counts per malicious category
# ---------------------------------------------------------------------------

MALICIOUS_MIN_COUNTS = {
    "D1": 40,
    "D2": 30,
    "D3": 30,
    "D4": 25,
    "D5": 25,
    "D6": 30,
    "D7": 25,
    "D8": 25,
    "E1": 25,
    "E2": 20,
    "C1": 25,
    "O1": 20,
    "P1": 20,
}


@pytest.mark.parametrize("category,minimum", MALICIOUS_MIN_COUNTS.items())
def test_malicious_holdout_min_count_per_category(category, minimum):
    samples = get_malicious()
    count = len(samples_for_category(samples, category))
    assert count >= minimum, (
        f"Category {category}: expected >= {minimum} samples, got {count}"
    )


# ---------------------------------------------------------------------------
# Test 8: All malicious text fields are non-empty strings
# ---------------------------------------------------------------------------

def test_malicious_holdout_text_nonempty():
    samples = get_malicious()
    bad = [i for i, s in enumerate(samples) if not isinstance(s.get("text"), str) or not s["text"].strip()]
    assert not bad, f"Samples at indices {bad} have empty or non-string text"


# ---------------------------------------------------------------------------
# Test 9: generate_safe_holdout returns non-empty list
# ---------------------------------------------------------------------------

def test_safe_holdout_not_empty():
    samples = get_safe()
    assert len(samples) > 0, "generate_safe_holdout() must return at least one sample"


# ---------------------------------------------------------------------------
# Test 10: All safe samples have required fields
# ---------------------------------------------------------------------------

def test_safe_holdout_required_fields():
    samples = get_safe()
    for i, sample in enumerate(samples):
        missing = REQUIRED_SAFE_FIELDS - set(sample.keys())
        assert not missing, f"Safe sample {i} is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 11: All safe samples have label == 0
# ---------------------------------------------------------------------------

def test_safe_holdout_label_is_zero():
    samples = get_safe()
    bad = [i for i, s in enumerate(samples) if s.get("label") != 0]
    assert not bad, f"Safe samples at indices {bad} do not have label=0"


# ---------------------------------------------------------------------------
# Test 12: All safe category codes are valid
# ---------------------------------------------------------------------------

def test_safe_holdout_valid_categories():
    samples = get_safe()
    unknown = {s["category"] for s in samples} - VALID_SAFE_CATEGORIES
    assert not unknown, f"Unexpected category codes in safe holdout: {unknown}"


# ---------------------------------------------------------------------------
# Test 13: Safe holdout has at least 100 samples total
# ---------------------------------------------------------------------------

def test_safe_holdout_minimum_total():
    samples = get_safe()
    assert len(samples) >= 100, (
        f"generate_safe_holdout() must return >= 100 samples, got {len(samples)}"
    )


# ---------------------------------------------------------------------------
# Test 14: No duplicate texts within each safe category
# ---------------------------------------------------------------------------

def test_safe_holdout_no_duplicate_texts_per_category():
    samples = get_safe()
    for cat in VALID_SAFE_CATEGORIES:
        cat_samples = samples_for_category(samples, cat)
        texts = [s["text"] for s in cat_samples]
        assert len(texts) == len(set(texts)), (
            f"Safe category {cat} has duplicate text entries"
        )


# ---------------------------------------------------------------------------
# Test 15: generate_adversarial_evasion returns non-empty list with label==1
# ---------------------------------------------------------------------------

def test_adversarial_evasion_not_empty():
    samples = get_adversarial()
    assert len(samples) > 0, "generate_adversarial_evasion() must return at least one sample"


def test_adversarial_evasion_label_is_one():
    samples = get_adversarial()
    bad = [i for i, s in enumerate(samples) if s.get("label") != 1]
    assert not bad, f"Adversarial samples at indices {bad} do not have label=1"


# ---------------------------------------------------------------------------
# Test 16: Adversarial samples have required fields
# ---------------------------------------------------------------------------

def test_adversarial_evasion_required_fields():
    samples = get_adversarial()
    for i, sample in enumerate(samples):
        missing = REQUIRED_ADVERSARIAL_FIELDS - set(sample.keys())
        assert not missing, f"Adversarial sample {i} is missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 17: Adversarial evasion has at least 500 samples
# ---------------------------------------------------------------------------

def test_adversarial_evasion_minimum_total():
    samples = get_adversarial()
    assert len(samples) >= 500, (
        f"generate_adversarial_evasion() must return >= 500 samples, got {len(samples)}"
    )


# ---------------------------------------------------------------------------
# Test 18: Malicious holdout covers all required attack categories
# ---------------------------------------------------------------------------

def test_malicious_holdout_covers_all_categories():
    samples = get_malicious()
    present = {s["category"] for s in samples}
    missing = VALID_MALICIOUS_CATEGORIES - present
    assert not missing, f"Malicious holdout is missing categories: {missing}"


# ---------------------------------------------------------------------------
# Test 19: Safe holdout covers all required safe categories
# ---------------------------------------------------------------------------

def test_safe_holdout_covers_all_categories():
    samples = get_safe()
    present = {s["category"] for s in samples}
    missing = VALID_SAFE_CATEGORIES - present
    assert not missing, f"Safe holdout is missing categories: {missing}"


# ---------------------------------------------------------------------------
# Test 20: Labels are binary (only 0 or 1 used across both holdout sets)
# ---------------------------------------------------------------------------

def test_label_values_are_binary():
    malicious = get_malicious()
    safe = get_safe()
    all_labels = {s["label"] for s in malicious + safe}
    assert all_labels <= {0, 1}, f"Unexpected label values found: {all_labels - {0, 1}}"
