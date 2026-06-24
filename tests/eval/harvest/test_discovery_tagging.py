"""Tests for discovery_tagging.tag_discovery / DiscoveryTagger.

These verify the taxonomy-aware tagging of weekly-harvest discovery records:

- ATLAS ids in a record's signals resolve to their mapped Na0S code (anchor).
- Curated keyword phrases map to canonical codes; the LONGEST phrase wins
  ("indirect prompt injection" -> I1 beats bare "prompt injection" -> CT).
- EVERY non-None return is canonical (passes TaxonomyValidator.validate_code).
- No confident match -> None (the caller flags for manual mapping; the record
  is never dropped, never guessed).
- Defensive: malformed records, non-string signals, and bare ambiguous terms
  do not produce a (possibly wrong) tag.

Each test is written to FAIL on a regression (e.g. if an ambiguous term started
tagging, or if a returned code stopped being canonical).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from na0s.eval.harvest import DiscoveryTagger, TaxonomyValidator, tag_discovery
from na0s.eval.harvest.discovery_tagging import _KEYWORD_TO_CODE

# scripts/weekly_harvest.py — loaded by path so the test does not depend on
# scripts/ being importable on sys.path. The harvester's additive tagging hook
# (_tag_discoveries) is the only thing under test here.
_WEEKLY_HARVEST_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "weekly_harvest.py"
)


def _load_weekly_harvest():
    spec = importlib.util.spec_from_file_location(
        "weekly_harvest_under_test", _WEEKLY_HARVEST_PATH
    )
    assert spec and spec.loader, "could not load scripts/weekly_harvest.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── every curated target is canonical (the never-invent contract) ──────────


def test_all_keyword_targets_are_canonical():
    """Each RHS of the keyword table is a real taxonomy code."""
    v = TaxonomyValidator()
    bad = {p: c for p, c in _KEYWORD_TO_CODE.items() if not v.validate_code(c)}
    assert bad == {}, f"non-canonical keyword targets: {bad}"


def test_returned_code_is_always_canonical():
    """Any non-None tag passes validate_code (guaranteed canonical output)."""
    v = TaxonomyValidator()
    records = [
        {"relevance_keywords": ["jailbreak"]},
        {"tags": ["rag poisoning"]},
        {"query": "indirect prompt injection"},
        {"tags": ["AML.T0054"]},
        {"relevance_keywords": ["over-refusal"]},
    ]
    for rec in records:
        code = tag_discovery(rec)
        assert code is not None
        assert v.validate_code(code) is True


# ── keyword path ───────────────────────────────────────────────────────────


def test_jailbreak_keyword_maps_to_d2():
    assert tag_discovery({"relevance_keywords": ["jailbreak"]}) == "D2"


def test_c1_compliance_evasion_keywords_map_to_c1_codes():
    """C1 social-engineering framings tag to canonical C1 / C1.x codes.

    Previously the table only knew "jailbreak" -> D2 (persona), so C1
    compliance-evasion framing (fiction / academia / emotion / authority) was
    never auto-tagged.  Each target is a real data/taxonomy.yaml code.
    """
    cases = {
        "compliance evasion": "C1",
        "social engineering jailbreak": "C1",
        "fictional framing": "C1.2",
        "analogical bypass": "C1.2",
        "academic framing": "C1.3",
        "emotional manipulation": "C1.4",
        "authority impersonation": "C1.5",
        "sycophancy exploitation": "C1.6",
        "crescendo attack": "C1.1",
    }
    v = TaxonomyValidator()
    for phrase, code in cases.items():
        assert tag_discovery({"relevance_keywords": [phrase]}) == code, phrase
        assert v.validate_code(code) is True, code


def test_c1_framing_beats_generic_jailbreak():
    """A C1-framed record tags as C1.2, not the generic D2 (longest wins)."""
    rec = {"relevance_keywords": ["fictional framing", "jailbreak"]}
    assert tag_discovery(rec) == "C1.2"


def test_rag_poisoning_punctuation_insensitive():
    """RAG-poisoning / RAG poisoning / rag_poisoning all hit the same key."""
    for kw in ("RAG poisoning", "RAG-poisoning", "rag_poisoning"):
        assert tag_discovery({"tags": [kw]}) == "IG", kw


def test_longest_phrase_wins_over_generic_substring():
    """'indirect prompt injection' -> I1 must beat the bare 'prompt injection'."""
    assert tag_discovery({"query": "indirect prompt injection"}) == "I1"
    # The generic phrase still resolves to the harvest landing zone.
    assert tag_discovery({"query": "prompt injection"}) == "CT"


def test_keyword_found_inside_longer_description_token():
    """A multi-word relevance keyword embeds the phrase among other words."""
    rec = {"relevance_keywords": ["a study of llm jailbreak techniques"]}
    assert tag_discovery(rec) == "D2"


def test_signal_keys_query_tags_relevance_all_read():
    assert tag_discovery({"query": "jailbreak"}) == "D2"
    assert tag_discovery({"tags": ["jailbreak"]}) == "D2"
    assert tag_discovery({"relevance_keywords": ["jailbreak"]}) == "D2"


# ── ATLAS path (anchor) ────────────────────────────────────────────────────


def test_atlas_id_in_signal_resolves_via_mapping():
    """A real ATLAS id in a tag resolves to its mapped Na0S code."""
    # AML.T0054 (LLM Jailbreak) -> D2 in the committed mapping.
    assert tag_discovery({"tags": ["AML.T0054"]}) == "D2"
    # AML.T0070 (RAG Poisoning) -> IG.
    assert tag_discovery({"relevance_keywords": ["AML.T0070"]}) == "IG"


def test_atlas_hit_takes_priority_over_keyword():
    """ATLAS is the anchor: an ATLAS id wins even alongside a keyword."""
    # AML.T0051.001 -> I1 (indirect); a co-present 'jailbreak' keyword (->D2)
    # must not override the explicit ATLAS id.
    rec = {"relevance_keywords": ["jailbreak"], "tags": ["AML.T0051.001"]}
    assert tag_discovery(rec) == "I1"


def test_unmapped_atlas_id_falls_through_to_none():
    """An ATLAS id with no committed mapping does not tag (no invention)."""
    # AML.T0099 is not a key in the mapping file.
    assert tag_discovery({"tags": ["AML.T9999"]}) is None


# ── None / defensive paths (never guess, never drop) ───────────────────────


def test_no_signal_returns_none():
    assert tag_discovery({}) is None
    assert tag_discovery({"description": "some prompt injection paper"}) is None


def test_ambiguous_terms_do_not_tag():
    """Bare ambiguous words must NOT produce a (likely wrong) tag."""
    for term in ("attack", "adversarial", "security", "benchmark", "evaluation"):
        assert tag_discovery({"relevance_keywords": [term]}) is None, term


def test_non_string_signals_are_skipped():
    rec = {"relevance_keywords": [None, 123, {"x": 1}], "tags": [["nested"]]}
    assert tag_discovery(rec) is None


def test_non_dict_record_returns_none():
    assert tag_discovery(None) is None  # type: ignore[arg-type]
    assert tag_discovery("jailbreak") is None  # type: ignore[arg-type]
    assert tag_discovery(["jailbreak"]) is None  # type: ignore[arg-type]


def test_description_is_not_a_signal_source():
    """Only relevance_keywords/tags/query are signals; description is metadata."""
    # 'jailbreak' only in description must NOT tag (description is not a signal).
    assert tag_discovery({"description": "a jailbreak dataset"}) is None
    # But the same term in a signal field does tag.
    assert tag_discovery({"tags": ["jailbreak"]}) == "D2"


# ── stale-alias guard (table re-validated at construction) ─────────────────


def test_noncanonical_table_entry_dropped_at_construction(tmp_path: Path):
    """A keyword whose target is absent from a custom taxonomy is dropped."""
    custom = tmp_path / "tax.yaml"
    custom.write_text(
        yaml.safe_dump({"categories": {"D2": {"severity": "high"}}}),
        encoding="utf-8",
    )
    # This taxonomy only has D2; every other keyword target is non-canonical.
    tagger = DiscoveryTagger(taxonomy=TaxonomyValidator(taxonomy_path=custom))
    # D2-targeted keyword still works.
    assert tagger.tag({"tags": ["jailbreak"]}) == "D2"
    # An IG-targeted keyword is dropped (IG not in this taxonomy) -> None.
    assert tagger.tag({"tags": ["rag poisoning"]}) is None


def test_word_boundary_prevents_substring_false_positive():
    """A single-token key must not match inside a larger word."""
    # 'dan prompt' is the key; 'abundance prompt' must not match via 'dan'.
    # (Also guards that bare 'dan' inside another word never tags.)
    assert tag_discovery({"relevance_keywords": ["abundance prompts study"]}) is None


# ── weekly_harvest hook: tag in place, flag-not-drop untagged records ───────


def test_harvester_hook_tags_in_place_and_never_drops():
    """``_tag_discoveries`` attaches a canonical code AND keeps every record.

    The additive weekly-harvest hook must:
    - tag records whose signals confidently map (-> canonical attack_category);
    - leave records with no confident match UNTAGGED but PRESENT (flagged for
      manual mapping, never dropped, never guessed);
    - mutate the list in place (same identity, same length).
    """
    wh = _load_weekly_harvest()
    v = TaxonomyValidator()

    discoveries = [
        {"id": "d1", "relevance_keywords": ["jailbreak"]},          # -> D2
        {"id": "d2", "tags": ["AML.T0070"]},                        # ATLAS -> IG
        {"id": "d3", "relevance_keywords": ["adversarial"]},        # ambiguous -> untagged
        {"id": "d4", "description": "a jailbreak paper"},           # no signal field -> untagged
    ]
    before_ids = [id(d) for d in discoveries]
    before_len = len(discoveries)

    wh._tag_discoveries(discoveries)  # noqa: SLF001 - exercising the hook

    # Nothing dropped; same objects mutated in place.
    assert len(discoveries) == before_len
    assert [id(d) for d in discoveries] == before_ids

    by_id = {d["id"]: d for d in discoveries}
    # Confidently-tagged records carry a CANONICAL code.
    assert by_id["d1"]["attack_category"] == "D2"
    assert v.validate_code(by_id["d1"]["attack_category"]) is True
    assert by_id["d2"]["attack_category"] == "IG"
    assert v.validate_code(by_id["d2"]["attack_category"]) is True
    # Untagged records are FLAGGED (no attack_category key) but kept.
    assert "attack_category" not in by_id["d3"]
    assert "attack_category" not in by_id["d4"]


def test_harvester_hook_handles_empty_list():
    """An empty discovery list is a no-op (never raises)."""
    wh = _load_weekly_harvest()
    discoveries: list = []
    wh._tag_discoveries(discoveries)  # noqa: SLF001
    assert discoveries == []
