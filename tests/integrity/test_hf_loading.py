"""Tests for na0s.integrity.hf_loading — hardened from_pretrained kwargs.

Covers the centralized supply-chain hardening for HuggingFace ``Auto*`` loads:
``use_safetensors=True`` (pickle-``.bin`` RCE) + ``trust_remote_code=False``
(custom-code path; does NOT cover CVE-2026-4372, which bypasses it) + pinned
``revision`` (the runtime control that does cover CVE-2026-4372).  No network /
real HF: the transformers ``from_pretrained`` symbols are mocked.
"""

import importlib
import sys
from unittest import mock

import pytest

from na0s.integrity import hf_loading as H


# ---------------------------------------------------------------------------
# C3 / C4 — helper contract + pin integrity
# ---------------------------------------------------------------------------

class TestHelperContract:
    def test_model_kwargs_always_hardened(self):
        """Model kwargs always force safetensors + deny remote code."""
        k = H.hf_from_pretrained_kwargs("meta-llama/Prompt-Guard-2-22M")
        assert k["use_safetensors"] is True
        assert k["trust_remote_code"] is False

    def test_tokenizer_kwargs_no_safetensors_but_deny_remote_code(self):
        """Tokenizer loads no weights -> no use_safetensors, but trust_remote_code=False."""
        tk = H.hf_tokenizer_kwargs("meta-llama/Prompt-Guard-2-22M")
        assert "use_safetensors" not in tk
        assert tk["trust_remote_code"] is False

    def test_unknown_model_still_hardened_revision_omitted(self):
        """An unmapped model still gets the safetensors + no-remote-code hardening, no revision."""
        k = H.hf_from_pretrained_kwargs("some/unknown-model")
        assert k["use_safetensors"] is True
        assert k["trust_remote_code"] is False
        assert "revision" not in k

    def test_unknown_model_logs_exactly_one_warning(self, caplog):
        with caplog.at_level("WARNING", logger="na0s.integrity.hf_loading"):
            H.hf_from_pretrained_kwargs("some/unknown-model")
        warnings = [r for r in caplog.records if "No pinned HF revision" in r.message]
        assert len(warnings) == 1

    def test_tokenizer_does_not_double_warn(self, caplog):
        """Tokenizer variant defaults warn=False so the paired model owns the warning."""
        with caplog.at_level("WARNING", logger="na0s.integrity.hf_loading"):
            H.hf_tokenizer_kwargs("some/unknown-model")
        warnings = [r for r in caplog.records if "No pinned HF revision" in r.message]
        assert warnings == []

    def test_env_override_injects_revision(self, monkeypatch):
        sha = "0" * 40
        monkeypatch.setenv(
            "NA0S_HF_REVISION_META_LLAMA_PROMPT_GUARD_2_22M", sha,
        )
        k = H.hf_from_pretrained_kwargs("meta-llama/Prompt-Guard-2-22M")
        assert k["revision"] == sha

    def test_env_override_name_mapping(self):
        assert (
            H._env_override_name("meta-llama/Prompt-Guard-2-22M")
            == "NA0S_HF_REVISION_META_LLAMA_PROMPT_GUARD_2_22M"
        )

    def test_is_pinned_revision_distinguishes_sha_from_branch(self):
        assert H.is_pinned_revision("a" * 40) is True
        assert H.is_pinned_revision("b" * 64) is True
        assert H.is_pinned_revision("main") is False
        assert H.is_pinned_revision("v1.0") is False
        assert H.is_pinned_revision(None) is False
        assert H.is_pinned_revision("") is False

    def test_non_sha_revision_warns(self, caplog, monkeypatch):
        """A branch/tag revision is forwarded but flagged as not content-addressed."""
        monkeypatch.setenv(
            "NA0S_HF_REVISION_META_LLAMA_PROMPT_GUARD_2_22M", "main",
        )
        with caplog.at_level("WARNING", logger="na0s.integrity.hf_loading"):
            k = H.hf_from_pretrained_kwargs("meta-llama/Prompt-Guard-2-22M")
        assert k["revision"] == "main"
        assert any("not an immutable commit" in r.message for r in caplog.records)

    def test_pinned_revisions_no_moving_ref_placeholder(self):
        """C4 — no PINNED_REVISIONS value is a moving ref (branch/tag).

        Values are either an immutable SHA or ``None`` (explicit unpinned).  A
        future regression that sets a value to ``"main"`` or a tag must fail here.
        """
        for model, rev in H.PINNED_REVISIONS.items():
            assert rev is None or H.is_pinned_revision(rev), (
                f"PINNED_REVISIONS[{model!r}] = {rev!r} is neither None nor an "
                f"immutable commit SHA — a moving ref would defeat the pin."
            )


# ---------------------------------------------------------------------------
# C1 — the kwargs are ACTUALLY passed at the from_pretrained call sites
# ---------------------------------------------------------------------------

def _reload_promptguard_with_mocks():
    """Reload na0s.ml.promptguard with fake torch + transformers.

    Returns (module, fake_transformers). The fake AutoTokenizer /
    AutoModelForSequenceClassification record the kwargs they were called with.
    """
    fake_torch = mock.MagicMock()
    fake_torch.no_grad.return_value.__enter__ = mock.MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = mock.MagicMock(return_value=False)

    fake_transformers = mock.MagicMock()

    saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_transformers
    # AutoTokenizer / AutoModel are attributes of the fake transformers module;
    # the module does `from transformers import AutoTokenizer, ...`
    sys.modules["transformers"].AutoTokenizer = mock.MagicMock()
    sys.modules["transformers"].AutoModelForSequenceClassification = mock.MagicMock()

    mod_name = "na0s.ml.promptguard"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)

    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    return mod


class TestFromPretrainedWiring:
    def test_promptguard_passes_hardened_kwargs(self):
        mod = _reload_promptguard_with_mocks()
        if not mod._HAS_TRANSFORMERS:
            pytest.skip("transformers fake did not register _HAS_TRANSFORMERS")

        clf = mod.PromptGuardClassifier(model_name="meta-llama/Prompt-Guard-2-22M")

        # Model mock must support .to(...).eval() chaining
        model_instance = mock.MagicMock()
        model_instance.to.return_value = model_instance
        mod.AutoModelForSequenceClassification.from_pretrained.return_value = model_instance

        ok = clf._ensure_loaded()
        assert ok is True

        # Tokenizer: trust_remote_code=False, NO use_safetensors
        tok_kwargs = mod.AutoTokenizer.from_pretrained.call_args.kwargs
        assert tok_kwargs.get("trust_remote_code") is False
        assert "use_safetensors" not in tok_kwargs

        # Model: use_safetensors=True AND trust_remote_code=False
        model_kwargs = mod.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs
        assert model_kwargs.get("use_safetensors") is True
        assert model_kwargs.get("trust_remote_code") is False

    def test_promptguard_load_failure_degrades_gracefully(self):
        """U2 — a hardened load that raises disables the layer, never crashes."""
        mod = _reload_promptguard_with_mocks()
        if not mod._HAS_TRANSFORMERS:
            pytest.skip("transformers fake did not register _HAS_TRANSFORMERS")

        clf = mod.PromptGuardClassifier(model_name="meta-llama/Prompt-Guard-2-22M")
        # Simulate the error transformers raises when use_safetensors=True but no
        # safetensors file exists.
        mod.AutoModelForSequenceClassification.from_pretrained.side_effect = OSError(
            "no safetensors found"
        )

        ok = clf._ensure_loaded()
        assert ok is False
        assert clf._init_failed is True
        # A second call short-circuits on _init_failed without re-raising.
        assert clf._ensure_loaded() is False
