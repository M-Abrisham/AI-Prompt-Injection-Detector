"""Tests for the shared pinned sentence-transformers loader (``_st_loader``).

This is the single place every Na0S site constructs a ``SentenceTransformer``.
These tests assert it threads the revision pin + cache-folder + offline mode
into the constructor, mirroring the revision assertions in
``test_embedding_classifier.py``.  They NEVER touch the real HuggingFace Hub:
the ``st_class`` is always a mock.
"""

import os
from unittest import mock

from na0s.ml import _st_loader


class TestPinnedLoaderRevision:
    def test_revision_constant_is_a_sha(self):
        rev = _st_loader.DEFAULT_MODEL_REVISION
        assert isinstance(rev, str)
        # 40-char lowercase hex (a git commit SHA).
        assert len(rev) == 40
        assert all(c in "0123456789abcdef" for c in rev)

    def test_default_revision_passed_to_constructor(self):
        fake_st = mock.MagicMock(return_value="MODEL")
        out = _st_loader.load_pinned_sentence_transformer(
            fake_st, "all-MiniLM-L6-v2",
        )
        assert out == "MODEL"
        fake_st.assert_called_once()
        args, kwargs = fake_st.call_args
        assert args == ("all-MiniLM-L6-v2",)
        assert kwargs.get("revision") == _st_loader.DEFAULT_MODEL_REVISION

    def test_explicit_revision_is_honored(self):
        fake_st = mock.MagicMock()
        _st_loader.load_pinned_sentence_transformer(
            fake_st, "some/other-model", revision="deadbeef",
        )
        _, kwargs = fake_st.call_args
        assert kwargs.get("revision") == "deadbeef"

    def test_extra_kwargs_forwarded(self):
        fake_st = mock.MagicMock()
        _st_loader.load_pinned_sentence_transformer(
            fake_st, "all-MiniLM-L6-v2", device="cpu",
        )
        _, kwargs = fake_st.call_args
        assert kwargs.get("device") == "cpu"
        assert kwargs.get("revision") == _st_loader.DEFAULT_MODEL_REVISION


class TestPinnedLoaderCacheAndOffline:
    def test_local_files_only_when_offline(self):
        fake_st = mock.MagicMock()
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            _st_loader.load_pinned_sentence_transformer(
                fake_st, "all-MiniLM-L6-v2",
            )
        _, kwargs = fake_st.call_args
        assert kwargs.get("local_files_only") is True

    def test_local_files_only_absent_when_online(self):
        fake_st = mock.MagicMock()
        env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
        with mock.patch.dict(os.environ, env, clear=True):
            _st_loader.load_pinned_sentence_transformer(
                fake_st, "all-MiniLM-L6-v2",
            )
        _, kwargs = fake_st.call_args
        assert "local_files_only" not in kwargs

    def test_cache_folder_from_hf_home(self):
        fake_st = mock.MagicMock()
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME")}
        env["HF_HOME"] = "/tmp/na0s_hf_cache"
        with mock.patch.dict(os.environ, env, clear=True):
            _st_loader.load_pinned_sentence_transformer(
                fake_st, "all-MiniLM-L6-v2",
            )
        _, kwargs = fake_st.call_args
        assert kwargs.get("cache_folder") == "/tmp/na0s_hf_cache"


class TestPinnedLoaderFallback:
    def test_typeerror_falls_back_without_pinning_kwargs(self):
        """Older sentence-transformers that reject pinning kwargs still load."""
        calls = []

        def fake_st(name, **kwargs):
            calls.append((name, kwargs))
            if "revision" in kwargs:
                raise TypeError("unexpected keyword argument 'revision'")
            return "MODEL"

        out = _st_loader.load_pinned_sentence_transformer(
            fake_st, "all-MiniLM-L6-v2",
        )
        assert out == "MODEL"
        # First (pinned) attempt raised; second attempt has no pinning kwargs.
        assert len(calls) == 2
        assert "revision" in calls[0][1]
        assert "revision" not in calls[1][1]

    def test_typeerror_fallback_preserves_extra_kwargs(self):
        """The retry keeps caller's own extra kwargs (e.g. device)."""
        calls = []

        def fake_st(name, **kwargs):
            calls.append((name, kwargs))
            if "revision" in kwargs:
                raise TypeError("revision unsupported")
            return "MODEL"

        _st_loader.load_pinned_sentence_transformer(
            fake_st, "all-MiniLM-L6-v2", device="cpu",
        )
        assert calls[1][1] == {"device": "cpu"}
