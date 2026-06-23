"""Multimodal hidden-channel scoring (M1/M2/M3) — corroborating boost.

Attackers smuggle instructions through a non-text channel (text rendered
into image pixels, EXIF/XMP metadata, white-on-white text in a PDF/DOCX,
hidden audio commands) so a text-only filter never sees the payload.

Layer-0 content-type sniffing (:mod:`na0s.input.content_type`) already
emits ``embedded_image`` / ``base64_hidden_image`` / ``embedded_pdf`` /
``embedded_mp3`` / ... flags when binary is embedded inside a text string,
and :mod:`na0s.detectors.visual_injection` does OCR + metadata analysis.
This module supplies two things both ``predict.scan`` and the cascade need,
as a single source of truth:

1. :data:`M_FLAG_MAP` — the **taxonomy-correct** Layer-0-flag -> M-code
   mapping (image=M1, audio=M2, document=M3 per ``data/taxonomy.yaml``).
   Attribution only; mapping a flag to a tag carries no risk by itself.

2. :func:`get_multimodal_boost` — a small, bounded, **corroborating**
   risk boost.  CRITICAL FP-safety contract: the mere *presence* of an
   embedded image / data-URI / attachment is **not** malicious — a clean
   avatar PNG, an inline screenshot, or an attached PDF must stay BELOW
   threshold.  The boost lifts the score ONLY when a hidden-channel flag
   is present AND an *independent* injection indicator also fired (a real
   rule/detector hit, or the composite was already elevated by the text
   itself).  It never decides on modality presence alone.

The boost mirrors the obfuscation / rag_poison pattern (a per-signal
weight under a hard cap, folded with ``min(risk + w, 1.0)``).  All
constants are named, not inline magic numbers.

Technique IDs: M1 (image), M2 (audio), M3 (document); see taxonomy.yaml.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Taxonomy-correct Layer-0-flag -> M-code mapping (single source of truth)
# ---------------------------------------------------------------------------
# Per data/taxonomy.yaml M-block:
#   M1 = Image     (M1.1 hidden text, M1.2 adversarial, M1.3 steganographic)
#   M2 = Audio     (M2.1 hidden voice, M2.2 adversarial, M2.3 ultrasonic)
#   M3 = Document  (M3.1 PDF/DOCX hidden, M3.2 metadata, M3.3 font-stego,
#                   M3.4 embedded macro)
#   M4 = Code
# There is NO M1.4 — the old map sent documents/audio/executables to the
# non-existent M1.4 and audio to M1.3 (a *steganographic image* code).
M_FLAG_MAP: dict[str, str] = {
    # --- Image (M1) ---
    "embedded_image": "M1.1",
    "embedded_png": "M1.1",
    "embedded_jpeg": "M1.1",
    "embedded_gif": "M1.1",
    "embedded_bmp": "M1.1",
    "embedded_tiff": "M1.1",
    "embedded_psd": "M1.1",
    "embedded_ico": "M1.1",
    "embedded_webp": "M1.1",
    "base64_hidden_image": "M1.1",
    "image_metadata_text": "M1.1",
    # --- Audio (M2) — was mis-mapped to M1.3 (a stego-IMAGE code) ---
    "embedded_audio": "M2.1",
    "embedded_mp3": "M2.1",
    "embedded_flac": "M2.1",
    "embedded_ogg": "M2.1",
    "embedded_aac": "M2.1",
    "embedded_midi": "M2.1",
    "embedded_wav": "M2.1",
    "embedded_aiff": "M2.1",
    "base64_hidden_audio": "M2.1",
    # --- Document (M3) — was mis-mapped to the non-existent M1.4 ---
    "embedded_document": "M3.1",
    "embedded_pdf": "M3.1",
    "embedded_ole2": "M3.1",
    "embedded_docx": "M3.1",
    "embedded_xlsx": "M3.1",
    "embedded_pptx": "M3.1",
    "embedded_ooxml": "M3.1",
    "embedded_odf": "M3.1",
    "base64_hidden_pdf": "M3.1",
    "base64_hidden_document": "M3.1",
    # PDF active-content surfaces are document-injection (macro-like) -> M3.4
    "pdf_javascript": "M3.4",
    "pdf_auto_action": "M3.4",
}

# Flags that denote a *hidden binary channel* eligible to corroborate an
# injection.  (Image / audio / document magic-byte + base64-hidden flags.)
_HIDDEN_CHANNEL_FLAGS: frozenset[str] = frozenset(M_FLAG_MAP.keys())

# Rule/flag hits that, on a base64/binary blob, are NOT independent evidence
# of injection — they fire on the *shape* of the blob (high entropy, mixed
# casing) or on the base64 wrapper itself, not on smuggled instructions.
# When a hidden-channel flag is present and ONLY these fired, we must not
# treat the input as corroborated.
_NON_CORROBORATING_HITS: frozenset[str] = frozenset({
    "high_entropy",
    "weird_casing",
    "punctuation_flood",
    "base64",
    "base64_blob_detected",
    "data_uri_detected",
    "entire_input_base64",
    "invisible_chars",
    # the visual detector's self-rescan echo of the SAME text (see
    # predict.py visual-routing); never corroborates by itself.
    "visual:ocr_injection",
    # PII patterns frequently false-match on random base64 (e.g. a run of
    # base64 chars looking like an API key); not independent injection text.
    "pii_api_key",
    # The centroid/embedding-similarity signal is documented (predict.py) to
    # produce low-confidence matches on benign text; on a base64 blob it is
    # matching the byte-noise, not smuggled instructions.  Not independent
    # corroboration when a hidden channel is present.  scan() emits the bare
    # name; the cascade emits it per-technique-id as ``embedding:<tid>`` — both
    # families are caught (see _NON_CORROBORATING_PREFIXES for the cascade form).
    "embedding_similarity",
    # Weak text-SHAPE structural signal (config weight 0.05): fires whenever
    # the prose's first word is an imperative verb ("tell", "show", "list").
    # It describes the SURROUNDING benign prose, not the hidden channel, and
    # false-fires on innocuous requests — so it is NOT independent evidence
    # that an embedded image/data-URI/attachment carries an injection.  The
    # STRONG structural injection signals (role_assignment, instruction_
    # boundary, negation_command) are deliberately left OUT of this set: those
    # are genuine override/persona/deny patterns and DO corroborate.  Without
    # this, a clean data-URI next to "tell me about ..." amplified a benign
    # 0.246 prompt into a BLOCK (modality-presence FP).
    "structural:imperative_start",
})

# Hit *prefixes* that are non-corroborating regardless of suffix.  The cascade
# names the centroid/embedding signal per-technique-id (``embedding:D7``,
# ``embedding:E1`` ...) rather than the bare ``embedding_similarity`` that
# scan() uses.  That signal is a capped (<=0.20) low-confidence semantic match;
# on a base64/binary blob it matches the byte-noise, not smuggled instructions,
# so the whole ``embedding:*`` family is non-corroborating — exactly mirroring
# ``embedding_similarity`` above.  Prefix-matched so predict.scan() and the
# cascade reach the SAME corroboration verdict (single source of truth).
_NON_CORROBORATING_PREFIXES: tuple[str, ...] = ("embedding:",)


def _is_corroborating_hit(hit: str) -> bool:
    """A hit is independent injection evidence unless it is a blob-shape /
    self-rescan / low-confidence-semantic artefact (exact name or known
    non-corroborating prefix)."""
    if hit in _NON_CORROBORATING_HITS:
        return False
    if hit.startswith(_NON_CORROBORATING_PREFIXES):
        return False
    return True

# ---------------------------------------------------------------------------
# Boost configuration (named constants, FP-calibrated against the 9-case
# benign-multimodal probe + the pooled benign holdout)
# ---------------------------------------------------------------------------
#: Per-corroborated-flag boost increment (mirrors obfuscation 0.15/flag).
MULTIMODAL_BOOST_PER_FLAG: float = 0.15
#: Hard cap on the total multimodal boost (mirrors obfuscation/rag_poison 0.30).
MULTIMODAL_BOOST_CAP: float = 0.30
#: Risk ceiling a clean (un-corroborated) embedded image/attachment is
#: dampened to.  Sits just under DECISION_THRESHOLD (0.55) so a base64 blob
#: that the ML/entropy/casing signals falsely score high on cannot block on
#: modality presence alone.  Calibrated against the 9-case benign-multimodal
#: probe (all clean data-URI PNGs must stay SAFE).
MULTIMODAL_CLEAN_RISK_CEILING: float = 0.50


def map_m_flags(flags: Iterable[str]) -> list[str]:
    """Map Layer-0 anomaly flags to taxonomy-correct M-codes.

    Attribution only — returns the deduped list of M-codes for the given
    flags, preserving first-seen order.  Carries no risk weight.
    """
    out: list[str] = []
    for f in flags:
        code = M_FLAG_MAP.get(f)
        if code and code not in out:
            out.append(code)
    return out


def has_hidden_channel(flags: Iterable[str]) -> bool:
    """True if any image/audio/document hidden-channel flag is present."""
    return bool(_HIDDEN_CHANNEL_FLAGS & set(flags))


def get_multimodal_boost(
    flags: Iterable[str],
    hits: Iterable[str],
    corroborated: bool = False,
) -> float:
    """Bounded, corroborating multimodal-injection boost.

    Parameters
    ----------
    flags:
        Layer-0 anomaly flags (e.g. ``embedded_image``,
        ``base64_hidden_pdf``).
    hits:
        Rule / detector hit names that have fired on the input.
    corroborated:
        Caller-supplied independent-evidence signal — e.g. the composite
        was already elevated by genuine injection text, or a visual
        *instruction-pattern* (not the self-rescan echo) fired.  When True,
        the boost lifts even if ``hits`` only contains blob-shape signals.

    Returns
    -------
    float
        ``0.0`` when there is no hidden channel, OR when a hidden channel is
        present but no independent injection indicator corroborates it
        (the clean-attachment case — presence is not malicious).  Otherwise
        a bounded boost in ``(0, MULTIMODAL_BOOST_CAP]``.

    FP-safety: a clean embedded image / data-URI / attachment with no
    injection text returns ``0.0`` and therefore cannot cross threshold on
    modality presence alone.
    """
    flag_set = set(flags)
    channel_flags = _HIDDEN_CHANNEL_FLAGS & flag_set
    if not channel_flags:
        return 0.0

    # Independent evidence?  Either the caller says so (composite already
    # elevated by real injection text), or a hit fired that is NOT just a
    # blob-shape / self-rescan / low-confidence-semantic artefact.
    independent_hit = any(_is_corroborating_hit(h) for h in hits)
    if not (corroborated or independent_hit):
        # Hidden channel present but nothing corroborates -> clean
        # attachment.  Presence is not malicious.
        return 0.0

    # One bounded boost per distinct M-category (image / audio / document),
    # not per raw flag, so a single PNG that trips both embedded_image and
    # embedded_png does not double-count.
    categories = {M_FLAG_MAP[f].split(".")[0] for f in channel_flags}
    boost = MULTIMODAL_BOOST_PER_FLAG * len(categories)
    return min(boost, MULTIMODAL_BOOST_CAP)


def is_uncorroborated_channel(
    flags: Iterable[str],
    hits: Iterable[str],
    corroborated: bool = False,
) -> bool:
    """True if a hidden channel is present but nothing corroborates it.

    This is the clean-attachment case: an embedded image / data-URI / doc
    whose only signals are blob-shape artefacts (high entropy, mixed casing,
    base64 wrapper, the OCR self-rescan echo) — i.e. the ML / entropy / PII
    rules false-fired on the *bytes* of the blob, not on smuggled
    instructions.  Callers use this to dampen the score back below
    threshold so modality presence alone never blocks.
    """
    flag_set = set(flags)
    if not (_HIDDEN_CHANNEL_FLAGS & flag_set):
        return False
    if corroborated:
        return False
    independent_hit = any(_is_corroborating_hit(h) for h in hits)
    return not independent_hit
