"""Shared pinned sentence-transformers loader.

Every site in Na0S that constructs a ``SentenceTransformer`` must route through
this module so that model-revision pinning, cache-folder selection, and
HuggingFace-Hub offline mode are applied in exactly one place.  An unpinned
``SentenceTransformer("all-MiniLM-L6-v2")`` resolves whatever snapshot the Hub
currently calls "latest", which can silently change between CI runs — a source
of runtime flakiness.  Pinning ``revision`` removes that nondeterminism.

The canonical loader is :func:`load_pinned_sentence_transformer`.  It takes the
``SentenceTransformer`` *class* as an explicit argument rather than importing it
here, for two reasons:

  1. Each caller imports ``SentenceTransformer`` at its own module scope behind
     its own ``try/except ImportError`` guard.  Passing the class in keeps the
     existing graceful-degradation behavior unchanged and lets each site's
     tests keep patching ``<that_module>.SentenceTransformer``.
  2. No duplicate top-level import of the optional dependency.

``DEFAULT_MODEL_REVISION`` is the pinned revision for the project-default
``all-MiniLM-L6-v2`` model.  Sites using a *different* model should pass their
own ``revision``; sites using the MiniLM default get this pin for free.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned revision for sentence-transformers/all-MiniLM-L6-v2
#
# This is the canonical ``main`` commit of
# ``sentence-transformers/all-MiniLM-L6-v2``.  Pinning it stops HuggingFace Hub
# from silently resolving a different "latest" snapshot across CI runs.
#
# NOTE: verify this SHA against
#   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commits/main
# before relying on it in production; updating it is a one-line change here.
# TODO: confirm the pinned revision SHA against the HF Hub.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def hub_offline() -> bool:
    """Return True when HuggingFace Hub network access should be disabled.

    Honors the standard ``HF_HUB_OFFLINE`` env var (any truthy value) so that
    callers/CI can force fully-local, deterministic loads.
    """
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def cache_folder() -> Optional[str]:
    """Return an explicit sentence-transformers cache folder if configured.

    Prefers ``HF_HOME``, then ``SENTENCE_TRANSFORMERS_HOME``.  Returns None when
    neither is set, letting sentence-transformers use its own default.
    """
    for env_var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME"):
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def load_pinned_sentence_transformer(
    st_class,
    model_name: str,
    revision: str = DEFAULT_MODEL_REVISION,
    **extra_kwargs,
):
    """Construct *st_class* with deterministic, pinned settings.

    Parameters
    ----------
    st_class :
        The ``SentenceTransformer`` class (passed in by the caller from its own
        module scope so per-site test patches keep intercepting construction).
    model_name : str
        Sentence-transformer model name or path.
    revision : str
        Model revision to pin.  Defaults to :data:`DEFAULT_MODEL_REVISION`
        (the all-MiniLM-L6-v2 pin); pass a model-specific revision for other
        models.
    **extra_kwargs :
        Any additional kwargs to forward verbatim to ``st_class`` (e.g.
        ``device=...``).

    Returns
    -------
    An ``st_class`` instance.  Raises whatever ``st_class(...)`` raises (the
    caller handles fallback).

    Older sentence-transformers releases may not accept ``revision`` /
    ``cache_folder`` / ``local_files_only``; we fall back to a construction
    without the pinning kwargs on ``TypeError`` so the pin is best-effort,
    never fatal.
    """
    kwargs = dict(extra_kwargs)
    kwargs["revision"] = revision
    folder = cache_folder()
    if folder:
        kwargs["cache_folder"] = folder
    if hub_offline():
        kwargs["local_files_only"] = True
    try:
        return st_class(model_name, **kwargs)
    except TypeError:
        # Installed sentence-transformers predates one of the pinning kwargs.
        logger.warning(
            "SentenceTransformer does not accept pinning kwargs %s; "
            "loading '%s' without them",
            sorted(k for k in kwargs if k not in extra_kwargs), model_name,
        )
        return st_class(model_name, **extra_kwargs)
