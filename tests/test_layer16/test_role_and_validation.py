"""Tests for T1.1 (role field) and T1.6 (input validation) in layer16."""

from __future__ import annotations

import pytest

from na0s.layer16.models import ConversationState, ConversationTurn
from na0s.layer16.state import add_turn, from_dict, to_dict, update_cumulative_risk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state(session_id: str = "test-session") -> ConversationState:
    return ConversationState(session_id=session_id)


# ===========================================================================
# T1.1 — Role field
# ===========================================================================


class TestRoleField:
    """ConversationTurn.role and its propagation through state functions."""

    def test_default_role_is_user(self):
        turn = ConversationTurn(turn_id="t1", text="hello")
        assert turn.role == "user"

    def test_explicit_role_assistant(self):
        turn = ConversationTurn(turn_id="t1", text="hello", role="assistant")
        assert turn.role == "assistant"

    def test_add_turn_with_role(self):
        state = _fresh_state()
        turn = add_turn(state, "hi", role="assistant")
        assert turn.role == "assistant"
        assert state.turns[0].role == "assistant"

    def test_add_turn_default_role(self):
        state = _fresh_state()
        turn = add_turn(state, "hi")
        assert turn.role == "user"

    def test_role_serialization_roundtrip(self):
        state = _fresh_state()
        add_turn(state, "user msg", role="user")
        add_turn(state, "assistant msg", role="assistant")
        add_turn(state, "system msg", role="system")

        d = to_dict(state)
        restored = from_dict(d)

        assert restored.turns[0].role == "user"
        assert restored.turns[1].role == "assistant"
        assert restored.turns[2].role == "system"

    def test_from_dict_backward_compat_missing_role(self):
        """Old serialized data without 'role' should default to 'user'."""
        state = _fresh_state()
        add_turn(state, "old message")

        d = to_dict(state)
        # Simulate old data that lacks the role key
        for turn_dict in d["turns"]:
            turn_dict.pop("role", None)

        restored = from_dict(d)
        assert restored.turns[0].role == "user"


# ===========================================================================
# T1.6 — Input validation on add_turn()
# ===========================================================================


class TestAddTurnValidation:
    """Validation guards on add_turn()."""

    def test_risk_score_above_one_raises(self):
        with pytest.raises(ValueError):
            add_turn(_fresh_state(), "text", risk_score=1.1)

    def test_risk_score_below_zero_raises(self):
        with pytest.raises(ValueError):
            add_turn(_fresh_state(), "text", risk_score=-0.01)

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            add_turn(_fresh_state(), "")

    def test_none_text_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            add_turn(_fresh_state(), None)  # type: ignore[arg-type]

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            add_turn(_fresh_state(), "text", role="bot")

    def test_invalid_label_type_raises(self):
        with pytest.raises(TypeError):
            add_turn(_fresh_state(), "text", label=123)  # type: ignore[arg-type]

    def test_valid_call_no_false_rejection(self):
        state = _fresh_state()
        turn = add_turn(state, "hello world", risk_score=0.5, label="safe", role="system")
        assert turn.text == "hello world"
        assert turn.risk_score == 0.5
        assert turn.role == "system"

    def test_boundary_risk_scores_accepted(self):
        state = _fresh_state()
        t0 = add_turn(state, "low", risk_score=0.0)
        t1 = add_turn(state, "high", risk_score=1.0)
        assert t0.risk_score == 0.0
        assert t1.risk_score == 1.0


# ===========================================================================
# T1.6 — Input validation on update_cumulative_risk()
# ===========================================================================


class TestUpdateCumulativeRiskValidation:
    """Validation guards on update_cumulative_risk()."""

    def test_turn_risk_above_one_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=1.5)

    def test_turn_risk_below_zero_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=-0.1)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=0.5, decay=1.5)

    def test_negative_decay_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=0.5, decay=-0.1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=0.5, alpha=0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            update_cumulative_risk(_fresh_state(), turn_risk=0.5, alpha=-1)

    def test_valid_call_works(self):
        state = _fresh_state()
        result = update_cumulative_risk(state, turn_risk=0.5, decay=0.8, alpha=0.2)
        assert 0.0 <= result <= 1.0
