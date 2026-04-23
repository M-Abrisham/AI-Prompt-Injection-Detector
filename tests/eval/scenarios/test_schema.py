"""Tests for na0s.eval.scenarios.schema — Scenario dataclass + validation."""

from __future__ import annotations

import pytest

from na0s.eval.scenarios import (
    EvaluatorType,
    Scenario,
    ScenarioEvaluator,
    ScenarioTurn,
    ScenarioType,
)


# ── Valid constructions ────────────────────────────────────────────


class TestValidConstruction:
    def test_single_prompt_minimal(self):
        s = Scenario(
            name="D1_test",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="critical",
            attack_category="D1",
            payload="Ignore previous instructions",
        )
        assert s.name == "D1_test"
        assert s.type == ScenarioType.SINGLE_PROMPT
        assert s.payload == "Ignore previous instructions"
        assert s.turns == []

    def test_multi_turn_minimal(self):
        s = Scenario(
            name="escalation_test",
            type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked",
            severity="high",
            attack_category="M1",
            turns=[
                ScenarioTurn(text="hi", expected_label="safe"),
                ScenarioTurn(text="attack", expected_label="malicious"),
            ],
        )
        assert s.type == ScenarioType.MULTI_TURN
        assert len(s.turns) == 2
        assert s.payload is None


# ── Stable ID auto-computation ─────────────────────────────────────


class TestStableId:
    def test_single_prompt_stable_id_autocomputed(self):
        s = Scenario(
            name="x",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="low",
            attack_category="D1",
            payload="Hello world",
        )
        assert s.stable_id is not None
        assert len(s.stable_id) == 64
        assert all(c in "0123456789abcdef" for c in s.stable_id)

    def test_stable_id_deterministic(self):
        s1 = Scenario(
            name="a",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="low",
            attack_category="D1",
            payload="Hello world",
        )
        s2 = Scenario(
            name="b",  # different name
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="allowed",  # different verdict
            severity="high",  # different severity
            attack_category="D2",  # different category
            payload="Hello world",  # same payload
        )
        assert s1.stable_id == s2.stable_id, "same payload → same stable_id"

    def test_stable_id_nfkc_normalized(self):
        s1 = Scenario(
            name="a", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="hello world",
        )
        s2 = Scenario(
            name="b", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="hello\tworld",
        )
        assert s1.stable_id == s2.stable_id, "whitespace collapsed"

    def test_stable_id_respects_override(self):
        s = Scenario(
            name="x",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="low",
            attack_category="D1",
            payload="anything",
            stable_id="explicit-override",
        )
        assert s.stable_id == "explicit-override"

    def test_multi_turn_stable_id_hashes_all_turns(self):
        s1 = Scenario(
            name="a", type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked", severity="low", attack_category="M1",
            turns=[
                ScenarioTurn(text="hi", expected_label="safe"),
                ScenarioTurn(text="attack", expected_label="malicious"),
            ],
        )
        s2 = Scenario(
            name="b", type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked", severity="low", attack_category="M1",
            turns=[
                ScenarioTurn(text="hi", expected_label="safe"),
                ScenarioTurn(text="attack", expected_label="malicious"),
            ],
        )
        assert s1.stable_id == s2.stable_id

    def test_multi_turn_different_turn_order_different_id(self):
        s1 = Scenario(
            name="a", type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked", severity="low", attack_category="M1",
            turns=[
                ScenarioTurn(text="A", expected_label="safe"),
                ScenarioTurn(text="B", expected_label="malicious"),
            ],
        )
        s2 = Scenario(
            name="b", type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked", severity="low", attack_category="M1",
            turns=[
                ScenarioTurn(text="B", expected_label="safe"),
                ScenarioTurn(text="A", expected_label="malicious"),
            ],
        )
        assert s1.stable_id != s2.stable_id


# ── Validation errors ──────────────────────────────────────────────


class TestValidationErrors:
    def test_single_prompt_missing_payload_raises(self):
        with pytest.raises(ValueError, match="SINGLE_PROMPT but payload is empty"):
            Scenario(
                name="x",
                type=ScenarioType.SINGLE_PROMPT,
                expected_verdict="blocked",
                severity="low",
                attack_category="D1",
            )

    def test_single_prompt_with_turns_raises(self):
        with pytest.raises(ValueError, match="turns must be empty"):
            Scenario(
                name="x",
                type=ScenarioType.SINGLE_PROMPT,
                expected_verdict="blocked",
                severity="low",
                attack_category="D1",
                payload="x",
                turns=[ScenarioTurn(text="y", expected_label="safe")],
            )

    def test_multi_turn_empty_turns_raises(self):
        with pytest.raises(ValueError, match="MULTI_TURN but turns list is empty"):
            Scenario(
                name="x",
                type=ScenarioType.MULTI_TURN,
                expected_verdict="blocked",
                severity="low",
                attack_category="M1",
            )

    def test_multi_turn_with_payload_raises(self):
        with pytest.raises(ValueError, match="payload must be empty"):
            Scenario(
                name="x",
                type=ScenarioType.MULTI_TURN,
                expected_verdict="blocked",
                severity="low",
                attack_category="M1",
                payload="x",
                turns=[ScenarioTurn(text="y", expected_label="safe")],
            )

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValueError, match="expected_verdict must be"):
            Scenario(
                name="x",
                type=ScenarioType.SINGLE_PROMPT,
                expected_verdict="maybe",
                severity="low",
                attack_category="D1",
                payload="x",
            )

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity must be"):
            Scenario(
                name="x",
                type=ScenarioType.SINGLE_PROMPT,
                expected_verdict="blocked",
                severity="catastrophic",
                attack_category="D1",
                payload="x",
            )


# ── Serialization ──────────────────────────────────────────────────


class TestToDict:
    def test_single_prompt_dict_has_payload_not_turns(self):
        s = Scenario(
            name="x",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="critical",
            attack_category="D1",
            payload="attack",
        )
        d = s.to_dict()
        assert d["payload"] == "attack"
        assert "turns" not in d
        assert d["type"] == "single_prompt"
        assert d["severity"] == "critical"
        assert d["stable_id"] == s.stable_id

    def test_multi_turn_dict_has_turns_not_payload(self):
        s = Scenario(
            name="x",
            type=ScenarioType.MULTI_TURN,
            expected_verdict="blocked",
            severity="high",
            attack_category="M1",
            turns=[
                ScenarioTurn(text="hi", expected_label="safe", risk_score=0.1),
                ScenarioTurn(text="attack", expected_label="malicious", risk_score=0.9),
            ],
        )
        d = s.to_dict()
        assert "payload" not in d
        assert len(d["turns"]) == 2
        assert d["turns"][0]["text"] == "hi"
        assert d["turns"][0]["risk_score"] == 0.1

    def test_compliance_tags_roundtrip(self):
        s = Scenario(
            name="x",
            type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked",
            severity="critical",
            attack_category="D1",
            payload="attack",
            compliance_tags=["owasp:llm:01", "mitre:atlas:exfiltration"],
        )
        d = s.to_dict()
        assert d["compliance_tags"] == ["owasp:llm:01", "mitre:atlas:exfiltration"]


# ── Mutable-default guards ─────────────────────────────────────────


class TestMutableDefaults:
    def test_tags_default_independent(self):
        s1 = Scenario(
            name="a", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="x",
        )
        s2 = Scenario(
            name="b", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="y",
        )
        s1.tags.append("roleplay")
        assert s2.tags == [], "field(default_factory=list) required"

    def test_compliance_tags_default_independent(self):
        s1 = Scenario(
            name="a", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="x",
        )
        s2 = Scenario(
            name="b", type=ScenarioType.SINGLE_PROMPT,
            expected_verdict="blocked", severity="low", attack_category="D1",
            payload="y",
        )
        s1.compliance_tags.append("owasp:llm:01")
        assert s2.compliance_tags == []
