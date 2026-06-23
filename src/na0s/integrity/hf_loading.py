"""Centralized, hardened keyword arguments for HuggingFace ``from_pretrained``.

Every site in Na0S that calls ``AutoTokenizer.from_pretrained`` /
``AutoModel*.from_pretrained`` must route its kwargs through this module so the
three supply-chain hardenings below are applied in exactly one place — there is
no copy-pasted security policy to drift across the call sites:

1. ``use_safetensors=True`` — refuses the pickle-backed ``pytorch_model.bin``
   deserialization path (arbitrary code execution at load time) and forces the
   safetensors format.  ``transformers`` will *prefer* safetensors when present
   but silently falls back to the pickle on its absence; forcing ``True`` turns
   that fallback into a hard error instead of a silent RCE surface.  This is the
   classic ``.bin`` deserialization RCE — a SEPARATE issue from CVE-2026-4372,
   which it does NOT mitigate.

2. ``trust_remote_code=False`` — refuses to execute model-repo-supplied Python
   (``modeling_*.py`` / custom code) at load time.  This closes the
   custom-code-execution path (the one ``trust_remote_code`` was designed to
   gate); we keep it as defense-in-depth.

   It does **NOT** mitigate **CVE-2026-4372** (CVSS 7.8, HIGH).  That advisory
   is explicit: the RCE *bypasses* ``trust_remote_code=False``.  A malicious
   ``config.json`` ``_attn_implementation_internal`` field whose value matches
   ``owner/repo`` is treated as a kernel repo id and flows through the
   ``transformers`` ``kernels``-package dispatch (``get_kernel_hub()``), which
   downloads+``importlib``-imports the attacker package under the standard
   ``from_pretrained`` call.  That dispatch path is never gated by
   ``trust_remote_code``, so the flag provides no protection against this CVE.

   The mitigations that actually cover CVE-2026-4372 are, in the order this
   runtime can apply them today:
     (a) PIN ``revision=<trusted commit SHA>`` (item 3 below) so a poisoned
         upstream ``config.json`` cannot be pulled at load time — the main
         residual the runtime can act on while the version floor is deferred;
     (b) the ``transformers>=5.3.0`` floor (the advisory's fix, which guards the
         offending fields during deserialization) — DEFERRED, see
         ``pyproject.toml``: ``sentence-transformers>=2.2,<4`` caps
         ``transformers<5``, so the floor is left as a TODO and is NOT added;
     (c) not installing the optional ``kernels`` package — the exploit chain
         requires it, so a deploy without ``kernels`` is not exposed.

3. ``revision=<pinned commit SHA>`` — content-addresses the artifact so a
   re-pointed or compromised Hub repo serving a different model (or a poisoned
   ``config.json``) at the same name is rejected.  This is the runtime control
   that does cover CVE-2026-4372 (see item 2(a)).  Pins live in
   :data:`PINNED_REVISIONS` below and are overridable via env var for
   air-gapped / local-mirror deploys (see :func:`_env_revision`).

This module is pure-python (no transformers/torch import) so importing it never
forces the optional heavy deps onto the core path.  Callers import it lazily,
behind their own ``_HAS_TRANSFORMERS`` guard.

The sentence-transformers loaders are hardened separately in
``na0s.ml._st_loader`` (``SentenceTransformer`` does not accept
``use_safetensors`` and pins ``revision`` there); this module is only the
``from_pretrained`` (transformers ``Auto*``) side.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pinned model revisions for transformers ``from_pretrained`` load sites.
#
# Maps a HuggingFace model id -> an immutable commit SHA (40-hex) or a signed,
# immutable tag.  Pinning ``revision`` stops the Hub from silently resolving a
# different snapshot at the same name across CI runs / deploys, and makes a
# re-pointed or compromised repo tamper-evident.
#
# IMPORTANT — placeholder policy:
#   The exact production SHAs are not known in this working tree.  Each value is
#   left as ``None`` (== "use the loader's own default, with a one-time warning")
#   rather than a fabricated SHA, because shipping a wrong/placeholder SHA would
#   either brick the load or give a false sense of pinning (review-checklist §7:
#   no unjustified magic constants).  An operator pins a model by either:
#     (a) filling in the real SHA here (verify against
#         https://huggingface.co/<model_id>/commits/main), or
#     (b) setting the per-model env override (see ``_env_revision``).
#
# The MiniLM sentence-transformers pin lives in ``na0s.ml._st_loader``
# (DEFAULT_MODEL_REVISION); the entries here are the transformers ``Auto*``
# models reachable via ``from_pretrained``.
# ---------------------------------------------------------------------------
PINNED_REVISIONS: Dict[str, Optional[str]] = {
    # Meta Prompt-Guard-2-22M — used by promptguard.py / promptguard_classifier.py
    "meta-llama/Prompt-Guard-2-22M": None,
    # all-MiniLM-L6-v2 transformer backbone — used by late_chunking.py via
    # AutoModel/AutoTokenizer (NOT the SentenceTransformer wrapper, which is
    # pinned in _st_loader.py).
    "sentence-transformers/all-MiniLM-L6-v2": None,
}

# Matches a HuggingFace immutable commit revision: a 40- or 64-char lowercase
# hex string.  Anything else (a branch like "main", a tag) is *not* treated as
# pinned by :func:`is_pinned_revision`.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


def _env_override_name(model_name: str) -> str:
    """Return the env-var name that overrides the pinned revision for *model_name*.

    e.g. ``meta-llama/Prompt-Guard-2-22M`` ->
         ``NA0S_HF_REVISION_META_LLAMA_PROMPT_GUARD_2_22M``.

    The override lets air-gapped / local-mirror deploys pin to a SHA they
    control without editing source, mirroring the ``NA0S_PROMPTGUARD_MODEL``
    env pattern used by promptguard_classifier.
    """
    slug = re.sub(r"[^0-9A-Za-z]+", "_", model_name).strip("_").upper()
    return "NA0S_HF_REVISION_" + slug


def _env_revision(model_name: str) -> Optional[str]:
    """Return an env-var-overridden revision for *model_name*, or None."""
    val = os.environ.get(_env_override_name(model_name))
    if val:
        return val.strip() or None
    return None


def resolve_revision(model_name: str) -> Optional[str]:
    """Return the pinned revision for *model_name*, honoring the env override.

    Resolution order: per-model env var -> :data:`PINNED_REVISIONS` -> None.
    Returns None when the model is unknown / unpinned (the caller then loads at
    the Hub default, which we surface as a one-time warning in
    :func:`hf_from_pretrained_kwargs`).
    """
    env_rev = _env_revision(model_name)
    if env_rev is not None:
        return env_rev
    return PINNED_REVISIONS.get(model_name)


def is_pinned_revision(revision: Optional[str]) -> bool:
    """True iff *revision* is an immutable commit SHA (not a branch/tag/None)."""
    return bool(revision) and bool(_SHA_RE.match(revision))  # type: ignore[arg-type]


def _revision_kwarg(model_name: str, warn: bool) -> Dict[str, object]:
    """Return ``{"revision": <pinned>}`` (or ``{}`` if unpinned).

    *warn* gates the unpinned / non-SHA warning so it fires exactly once per
    load (we call the model variant with ``warn=True`` and the sibling tokenizer
    variant with ``warn=False`` to avoid double-logging the same model).
    """
    revision = resolve_revision(model_name)
    if not revision:
        if warn:
            logger.warning(
                "No pinned HF revision for '%s'; loading at the Hub default. Set "
                "PINNED_REVISIONS['%s'] or the %s env var to pin it.",
                model_name, model_name, _env_override_name(model_name),
            )
        return {}
    if warn and not is_pinned_revision(revision):
        logger.warning(
            "HF revision for '%s' is '%s', which is not an immutable commit "
            "SHA; the artifact is not content-addressed and could be "
            "re-pointed at the Hub.",
            model_name, revision,
        )
    return {"revision": revision}


def hf_from_pretrained_kwargs(model_name: str, warn: bool = True) -> Dict[str, object]:
    """Return the hardened kwargs for an ``AutoModel*.from_pretrained(...)`` call.

    Always includes ``use_safetensors=True`` (blocks the pickle-``.bin`` RCE)
    and ``trust_remote_code=False`` (blocks the custom-code path; defense-in-
    depth — note it does NOT cover CVE-2026-4372, see module docstring).
    Includes ``revision=<pinned>`` only when a pin is known for *model_name* —
    that pin is the runtime control that covers CVE-2026-4372; if the model is
    unknown / unpinned, ``revision`` is omitted and a single warning is logged
    (unless *warn* is False) so an env-overridden custom model still loads
    (un-pinned but flagged).

    The returned dict is meant to be splatted into the call:
    ``AutoModel.from_pretrained(name, **hf_from_pretrained_kwargs(name))``.
    """
    kwargs: Dict[str, object] = {
        "use_safetensors": True,
        "trust_remote_code": False,
    }
    kwargs.update(_revision_kwarg(model_name, warn=warn))
    return kwargs


def hf_tokenizer_kwargs(model_name: str, warn: bool = False) -> Dict[str, object]:
    """Return the hardened kwargs for an ``AutoTokenizer.from_pretrained(...)`` call.

    A tokenizer loads no model weights, so ``use_safetensors`` is *not* passed
    (it is a weight-loading arg and is needlessly broad on the tokenizer path
    across the supported transformers version range).  We still pass
    ``trust_remote_code=False`` (a tokenizer repo can ship custom tokenizer code)
    and the pinned ``revision``.

    *warn* defaults to False so the paired model load (which is called with
    ``warn=True``) owns the single unpinned-revision warning for the model.
    """
    kwargs: Dict[str, object] = {"trust_remote_code": False}
    kwargs.update(_revision_kwarg(model_name, warn=warn))
    return kwargs
