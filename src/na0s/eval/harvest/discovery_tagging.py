"""Tag weekly-harvest discovery records with a CANONICAL Na0S attack_category.

The weekly harvester (``scripts/weekly_harvest.py``) discovers datasets, papers,
and repos and emits *discovery records* — loose metadata dicts (``id``, ``source``,
``description``, ``tags``, ``relevance_keywords``, …). Those records eventually
seed the F14 eval library, where every scenario MUST carry a real
``attack_category`` from ``data/taxonomy.yaml`` (an invented code silently
corrupts per-category TPR/FPR scoring — it is an injection vector into the eval
library, exactly the threat this whole pipeline defends against).

This module bridges the gap *deterministically and offline*. It reads the
record's own discovery signals (``relevance_keywords`` / ``tags`` / ``query``)
and maps them to a canonical Na0S code via two paths:

1. **Direct MITRE ATLAS hit.** If a signal token is a real ATLAS technique id
   (``AML.Txxxx``), it is resolved through the ATLAS-aware
   :class:`~na0s.eval.harvest.taxonomy.TaxonomyValidator` (Build-1's
   ``resolve_to_na0s`` + the human-reviewed
   ``data/threat_intel_snapshots/atlas_to_na0s_mapping.yaml``). ATLAS is the
   anchor, so this path wins when present.
2. **Curated keyword -> code table.** A small, conservative table of unambiguous
   attack-class phrases (``"jailbreak" -> D2``, ``"rag poisoning" -> IG``, …).
   Longest phrase wins, so ``"indirect prompt injection"`` beats the bare
   ``"prompt injection"``.

Discipline (mirrors PR #437's ``_validated_technique`` — these are the security
contract, not nice-to-haves):

- **Never invent a code.** Every table value AND every ATLAS resolution is
  re-validated against the live taxonomy at tag time via
  :meth:`TaxonomyValidator.validate_code`. A target that is not canonical is
  dropped, never emitted.
- **No confident match -> ``None``.** The caller leaves the record *untagged and
  flagged for manual mapping*; it is NEVER dropped and NEVER guessed.
- **Pure / local / keyless.** No network, no external LLM (an LLM in this path
  would itself be an injection surface). Deterministic given the record + the
  committed taxonomy + ATLAS mapping.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

from na0s.eval.harvest.taxonomy import TaxonomyValidator

logger = logging.getLogger(__name__)

# Record keys whose values are the harvester's own relevance signals. ``query``
# is accepted for forward-compat even though the current weekly_harvest emits
# ``relevance_keywords`` rather than a bare ``query`` (the spec names all three).
_SIGNAL_KEYS = ("relevance_keywords", "tags", "query")

# MITRE ATLAS technique id, recognized anywhere inside a free-text signal token
# (e.g. a tag like "AML.T0054" or a relevance keyword embedding the id). The
# captured id is then routed through TaxonomyValidator.resolve_to_na0s — this
# regex only *finds* candidate ids, it never validates them on its own.
_ATLAS_ID_IN_TEXT_RE = re.compile(r"AML\.T\d{4}(?:\.\d{3})?")

# ---------------------------------------------------------------------------
# Curated keyword -> canonical Na0S code table.
#
# Each LHS is an unambiguous attack-class phrase that a defensive
# prompt-injection SDK observes at the input/output boundary; each RHS is a
# CANONICAL data/taxonomy.yaml code (re-validated at construction time — see
# DiscoveryTagger.__init__, which drops any RHS that is not canonical so a
# taxonomy edit can never leave a stale alias pointing at a dead code).
#
# Deliberately CONSERVATIVE: only phrases with a single defensible home are
# listed. Genuinely ambiguous terms (e.g. bare "attack", "adversarial",
# "security", "evaluation", "benchmark") are intentionally OMITTED so the
# tagger returns None and routes the record to manual mapping rather than
# guessing — false-positive tags corrupt scoring exactly like invented codes.
#
# Matching is done on whole, normalized phrases (see _normalize), with the
# LONGEST matching phrase winning, so the more specific "indirect prompt
# injection" (-> I1) beats the generic "prompt injection" (-> CT).
# ---------------------------------------------------------------------------
_KEYWORD_TO_CODE: dict[str, str] = {
    # --- prompt injection family ---
    # Bare/direct prompt injection -> Combo Techniques is the harvest landing
    # zone the ATLAS parent (AML.T0051) also maps to; a human reviewer narrows
    # it to D1 (direct) vs I1 (indirect) at promotion.
    "prompt injection": "CT",
    "direct prompt injection": "D1",
    "instruction override": "D1",
    "ignore previous instructions": "D1",
    "indirect prompt injection": "I1",
    "data poisoning": "I1",
    "training data poisoning": "I1",
    "context poisoning": "I1",
    # --- jailbreak / persona ---
    "jailbreak": "D2",
    "roleplay attack": "D2",
    "persona hijack": "D2",
    "guardrail bypass": "D2",
    "safety bypass": "D2",
    "dan prompt": "D2",
    # --- obfuscation / encoding ---
    "prompt obfuscation": "D4",
    "encoding attack": "D4",
    "base64 injection": "D4",
    "unicode evasion": "D5",
    "homoglyph attack": "D5",
    "multilingual injection": "D6",
    # --- context window ---
    "context window manipulation": "D8",
    # --- HTML / markup ---
    "html injection": "I2",
    "markdown injection": "I2",
    # --- exfiltration ---
    "system prompt extraction": "E",
    "system prompt leak": "E",
    "prompt leaking": "E",
    "data exfiltration": "E",
    # --- privacy ---
    "membership inference": "P2",
    "training data extraction": "P2",
    "training data leakage": "P2",
    "data extraction attack": "P2",
    "pii elicitation": "P2",
    "pii extraction": "P2",
    "personal data extraction": "P2",
    "model inversion": "A",
    "privacy attack": "P",
    "credential leakage": "P",
    "malicious code generation": "P3",
    # --- adversarial ML ---
    "adversarial example": "A",
    "adversarial perturbation": "A",
    "model evasion": "M",
    # --- output manipulation ---
    "output manipulation": "O",
    "response rendering attack": "O",
    # --- agent / tool abuse ---
    "agent hijack": "T",
    "tool abuse": "T",
    "tool invocation abuse": "T",
    "sandbox escape": "T",
    "command injection": "T",
    # --- resource / availability ---
    "denial of service": "R",
    "resource exhaustion": "R",
    # --- supply chain ---
    "supply chain attack": "S",
    "model backdoor": "S",
    # --- ingestion / RAG ---
    "rag poisoning": "IG",
    "retrieval poisoning": "IG",
    "ingestion manipulation": "IG",
    # --- inter-model / self-replication ---
    "prompt self-replication": "IM",
    "llm worm": "IM",
    # --- multimodal ---
    "multimodal injection": "M",
    "image injection": "M",
    # --- benign control (NOT an attack; explicit so harvested over-refusal
    #     corpora can be filed as the benign sentinel rather than dropped) ---
    "benign control": "BEN",
    "over-refusal": "BEN",
}


def _normalize(text: str) -> str:
    """Lowercase + collapse non-alphanumerics to single spaces (kept '.' for ATLAS).

    Phrase matching is whitespace/punctuation-insensitive so that, e.g.,
    "RAG-poisoning", "RAG poisoning", and "rag_poisoning" all match the table
    key "rag poisoning". ATLAS ids are extracted from the *raw* token before
    normalization, so the dot in ``AML.T0054`` is preserved on that path.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class DiscoveryTagger:
    """Map a harvest discovery record to a canonical Na0S ``attack_category``.

    Parameters
    ----------
    taxonomy : TaxonomyValidator | None
        The ATLAS-aware validator. Defaults to a fresh
        :class:`~na0s.eval.harvest.taxonomy.TaxonomyValidator` over
        ``data/taxonomy.yaml`` (+ the optional ATLAS mapping).

    Notes
    -----
    Construction re-validates every keyword-table RHS against the live taxonomy
    and drops non-canonical entries (logging a warning), so the table can never
    out-live a code it points at. ATLAS hits are validated on resolution.
    """

    def __init__(self, taxonomy: Optional[TaxonomyValidator] = None) -> None:
        self.taxonomy = taxonomy or TaxonomyValidator()
        # Keep only entries whose target is canonical RIGHT NOW. This is the
        # _validated_technique discipline applied to the static table: a stale
        # alias to a removed/renamed code is dropped rather than silently
        # tagging records with a dead category.
        validated: dict[str, str] = {}
        for phrase, code in _KEYWORD_TO_CODE.items():
            if self.taxonomy.validate_code(code):
                validated[_normalize(phrase)] = code
            else:  # pragma: no cover - guards against a future taxonomy edit
                logger.warning(
                    "discovery_tagging: dropping keyword %r -> %r "
                    "(target not in taxonomy)",
                    phrase,
                    code,
                )
        # Longest phrase first so specific phrases win over generic substrings.
        self._phrase_codes: list[tuple[str, str]] = sorted(
            validated.items(), key=lambda kv: len(kv[0]), reverse=True
        )

    # ------------------------------------------------------------------ #
    def _iter_signal_tokens(self, record: dict[str, Any]) -> Iterable[str]:
        """Yield raw string signals from the record's relevance fields.

        Accepts both list-valued (``relevance_keywords``, ``tags``) and
        scalar (``query``) fields; non-string entries are skipped. Empty
        strings are skipped. Order is ``_SIGNAL_KEYS`` order — this only
        affects the ATLAS first-hit path's determinism, never correctness.
        """
        for key in _SIGNAL_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                if value.strip():
                    yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        yield item

    def _atlas_hit(self, tokens: list[str]) -> Optional[str]:
        """Return the Na0S code for the first resolvable ATLAS id, else None."""
        for token in tokens:
            for match in _ATLAS_ID_IN_TEXT_RE.findall(token):
                resolved = self.taxonomy.resolve_to_na0s(match)
                # resolve_to_na0s already restricts to canonical targets, but
                # re-assert via validate_code to mirror the never-invent rule.
                if resolved is not None and self.taxonomy.validate_code(resolved):
                    return resolved
        return None

    def _keyword_hit(self, tokens: list[str]) -> Optional[str]:
        """Return the code for the longest matching curated phrase, else None.

        A phrase matches when it appears as a normalized substring of the
        joined, normalized signal text (word-boundary padded so "dan" inside
        "abundance" cannot match the "dan prompt" key — the key itself is
        multi-word, but padding makes single-token keys safe too).
        """
        haystack = " " + _normalize(" ".join(tokens)) + " "
        for phrase, code in self._phrase_codes:
            if f" {phrase} " in haystack:
                return code
        return None

    def tag(self, record: dict[str, Any]) -> Optional[str]:
        """Return a canonical Na0S ``attack_category`` for ``record``, or None.

        Resolution order (ATLAS is the anchor):

        1. Direct ATLAS id in any signal -> its mapped Na0S code.
        2. Curated keyword phrase (longest wins) -> its canonical code.
        3. No confident match -> ``None`` (caller flags for manual mapping;
           the record is never dropped, never guessed).

        The return value, when not None, is guaranteed canonical (it passed
        :meth:`TaxonomyValidator.validate_code` at tag time).
        """
        if not isinstance(record, dict):
            return None
        tokens = list(self._iter_signal_tokens(record))
        if not tokens:
            return None
        return self._atlas_hit(tokens) or self._keyword_hit(tokens)


# Module-level convenience: a single shared tagger so a hot loop (the weekly
# harvester tagging hundreds of records) does not reload taxonomy.yaml per call.
# Built lazily so importing this module never touches the filesystem.
_DEFAULT_TAGGER: Optional[DiscoveryTagger] = None


def _default_tagger() -> DiscoveryTagger:
    global _DEFAULT_TAGGER
    if _DEFAULT_TAGGER is None:
        _DEFAULT_TAGGER = DiscoveryTagger()
    return _DEFAULT_TAGGER


def tag_discovery(record: dict[str, Any]) -> Optional[str]:
    """Tag a harvest discovery ``record`` with a canonical Na0S attack_category.

    Thin functional wrapper over a process-wide :class:`DiscoveryTagger`. See
    :meth:`DiscoveryTagger.tag` for the resolution contract. Returns ``None``
    when no canonical category can be confidently assigned — the caller must
    leave the record untagged and flagged for manual mapping, NOT drop it.

    Pure / offline / keyless. Safe to call in a tight loop.
    """
    return _default_tagger().tag(record)
