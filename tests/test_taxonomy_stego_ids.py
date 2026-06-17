"""Taxonomy-integrity tests for stego/unicode technique IDs.

Closes the audit gap (2026-06-17): predict.py emitted D4.9/D4.10 which were
undefined in taxonomy.yaml, and tag-char / variation-selector smuggling were
mislabeled as D5.2. These tests pin the new IDs and enforce the general
invariant that every leaf technique ID predict.py emits is defined.
"""
import re
from pathlib import Path

import yaml

import na0s.predict as _predict

_REPO = Path(_predict.__file__).resolve().parents[2]
_TAXONOMY = _REPO / "data" / "taxonomy.yaml"
_PREDICT_SRC = Path(_predict.__file__).read_text(encoding="utf-8")


def _defined_leaf_ids():
    data = yaml.safe_load(_TAXONOMY.read_text(encoding="utf-8"))
    ids = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if re.fullmatch(r"D\d+\.\d+", str(k)):
                    ids.add(str(k))
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(data)
    return ids


def test_new_stego_technique_ids_defined():
    defined = _defined_leaf_ids()
    for tid in ("D4.9", "D4.10", "D5.8", "D5.9"):
        assert tid in defined, f"{tid} missing from taxonomy.yaml"


def test_stego_technique_names():
    data = yaml.safe_load(_TAXONOMY.read_text(encoding="utf-8"))
    flat = {}

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if re.fullmatch(r"D\d+\.\d+", str(k)) and isinstance(v, dict):
                    flat[str(k)] = v.get("name", "")
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(data)
    assert "Whitespace-steganography" in flat["D4.10"]
    assert "Tag-character" in flat["D5.8"]
    assert "Variation-selector" in flat["D5.9"]


def test_no_dangling_technique_ids_in_predict():
    """Every D-leaf technique ID emitted by predict.py must be defined."""
    defined = _defined_leaf_ids()
    referenced = set(re.findall(r'"(D\d+\.\d+)"', _PREDICT_SRC))
    dangling = referenced - defined
    assert not dangling, f"predict.py emits undefined technique IDs: {sorted(dangling)}"


def test_tag_and_vs_no_longer_mislabeled_d52():
    """tag-char and variation-selector smuggling must map to their own IDs,
    not be conflated with D5.2 (zero-width insertion)."""
    assert '"unicode_tag_stego": "D5.8"' in _PREDICT_SRC
    assert '"variation_selector_stego": "D5.9"' in _PREDICT_SRC
    assert '"zero_width_stego": "D5.2"' in _PREDICT_SRC
