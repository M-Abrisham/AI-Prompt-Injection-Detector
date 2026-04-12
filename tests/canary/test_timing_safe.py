"""Tests for timing-safe canary token comparison."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from na0s.canary.verifier import CanaryTokenVerifier


class TestTimingSafeVerify:
    """CanaryTokenVerifier.verify uses hmac.compare_digest."""

    def test_intact_prompt_returns_true(self):
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("You are helpful.")
        result = v.verify(prompt, canary)
        assert result["intact"] is True
        assert result["reason"] == ""

    def test_tampered_prompt_returns_false(self):
        v = CanaryTokenVerifier()
        _prompt, canary = v.embed("You are helpful.")
        result = v.verify("Totally different prompt.", canary)
        assert result["intact"] is False
        assert "tampered" in result["reason"]

    def test_stripped_canary_returns_false(self):
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("You are helpful.")
        # Remove the integrity check line
        stripped = prompt.split("\n")[0]
        result = v.verify(stripped, canary)
        assert result["intact"] is False

    def test_modified_canary_value_returns_false(self):
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("You are helpful.")
        # Replace the canary value with a different one
        wrong_canary = "__NA0S_VERIFY_0000000000000000__"
        result = v.verify(prompt, wrong_canary)
        assert result["intact"] is False

    def test_hmac_compare_digest_is_called(self):
        """Verify that hmac.compare_digest is actually used for the comparison."""
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("You are helpful.")
        with patch("na0s.canary.verifier.hmac.compare_digest", return_value=True) as mock_cd:
            result = v.verify(prompt, canary)
            mock_cd.assert_called_once()
            # Verify it was called with the right arguments
            args = mock_cd.call_args[0]
            assert args[0] == canary  # found in prompt
            assert args[1] == canary  # expected
        assert result["intact"] is True

    def test_hmac_compare_digest_called_with_wrong_token(self):
        """When canary in prompt differs from expected, compare_digest sees both."""
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("You are helpful.")
        wrong = "__NA0S_VERIFY_aaaaaaaaaaaaaaaa__"
        with patch("na0s.canary.verifier.hmac.compare_digest", return_value=False) as mock_cd:
            result = v.verify(prompt, wrong)
            mock_cd.assert_called_once_with(canary, wrong)
        assert result["intact"] is False

    def test_true_negative_no_canary_in_output(self):
        """Normal text without any canary should not match."""
        v = CanaryTokenVerifier()
        _prompt, canary = v.embed("System prompt.")
        result = v.verify("Normal LLM response with no canary.", canary)
        assert result["intact"] is False

    def test_true_positive_canary_present(self):
        """Embedded canary round-trips correctly."""
        v = CanaryTokenVerifier()
        prompt, canary = v.embed("Be concise.")
        assert canary in prompt
        result = v.verify(prompt, canary)
        assert result["intact"] is True
