# PHASE A — Safe extraction of shared weighted composite math.
# Both callers (predict.py._weighted_decision and
# cascade.py.WeightedClassifier.classify) use this shared helper
# for the ML+rule+obf composite calculation only.
#
# PHASE B (deferred) — cascade.py alignment items:
#   - Add structural feature support (predict.py has 10+ features, cascade has 0)
#   - Add multi-layer agreement boost (predict.py only)
#   - Add critical-content floor: max(composite, 0.60) (predict.py only)
#   - Align override protection: set comparison vs string comparison
#   - Align score clamping position (end vs immediate)
# Phase B requires dedicated testing wave — do not implement without approval.


"""Canonical weighted voting logic for Na0S prompt injection detection.

This module is the **single source of truth** for combining ML confidence,
rule severity, obfuscation signals, structural features, and embedding
similarity into a composite score that determines the SAFE/MALICIOUS verdict.

Both ``predict.py`` and ``cascade.py`` delegate to this module.  No other
file should implement weighted voting logic.

History:
    2026-03-12 — Extracted from predict.py to eliminate duplication with
    cascade.py (Issue #2).  predict.py had the complete implementation;
    cascade.py was a frozen subset missing structural features, decoded
    view classification, multi-layer boost, technique-family boost, and
    extended override protection.
"""

from __future__ import annotations

import json
import logging
import math
import os

from ..rules import RULES, SEVERITY_WEIGHTS
from ..detectors.multilingual_intent import HEURISTIC_HITS as _HEURISTIC_HITS
from .signal_boost import calculate_boost_from_names

_logger = logging.getLogger(__name__)

# ── Constants (exported for tests and downstream consumers) ──────────────

#: Hardcoded fallback when optimal_threshold.json is absent.
_FALLBACK_THRESHOLD = 0.55

#: Path to the threshold JSON produced by scripts/optimize_threshold.py.
_THRESHOLD_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir,
    "data", "processed", "optimal_threshold.json",
)

#: Cached value — computed once by :func:`get_decision_threshold`.
_cached_threshold = None


def _valid_threshold(value):
    """Return ``value`` as a finite float in (0, 1], or ``None`` if invalid.

    A NaN/inf/out-of-range threshold is a fail-OPEN hazard: ``score >= nan`` is
    always False (everything becomes SAFE) and a non-positive threshold marks
    everything MALICIOUS. We reject such values so the resolver falls through
    to the calibrated JSON or the safe hardcoded fallback (fail CLOSED).
    """
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    # math.isfinite rejects NaN and ±inf; bound to the meaningful (0, 1] range.
    if not math.isfinite(f) or not (0.0 < f <= 1.0):
        return None
    return f


def get_decision_threshold():
    """Return the active decision threshold (cached after first call).

    Resolution order:
        1. ``DECISION_THRESHOLD`` environment variable  (float)
        2. ``recall95_threshold`` from ``data/processed/optimal_threshold.json``
        3. Hardcoded fallback ``0.55``
    """
    global _cached_threshold
    if _cached_threshold is not None:
        return _cached_threshold

    # 1. Env-var override
    env_val = os.environ.get("DECISION_THRESHOLD")
    if env_val is not None:
        valid = _valid_threshold(env_val)
        if valid is not None:
            _cached_threshold = valid
            _logger.info(
                "Decision threshold set from DECISION_THRESHOLD env var: %.4f",
                _cached_threshold,
            )
            return _cached_threshold
        _logger.warning(
            "Invalid DECISION_THRESHOLD env var %r (must be finite, in (0,1]); "
            "ignoring.", env_val,
        )

    # 2. Load from optimal_threshold.json
    json_path = os.path.normpath(_THRESHOLD_JSON_PATH)
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            valid = _valid_threshold(data["recall95_threshold"])
            if valid is None:
                raise ValueError(
                    f"recall95_threshold not finite in (0,1]: "
                    f"{data.get('recall95_threshold')!r}"
                )
            _cached_threshold = valid
            _logger.info(
                "Decision threshold loaded from %s: %.4f",
                json_path, _cached_threshold,
            )
            return _cached_threshold
        except Exception as exc:
            _logger.warning(
                "Failed to load threshold from %s: %s; using fallback.",
                json_path, exc,
            )

    # 3. Fallback
    _cached_threshold = _FALLBACK_THRESHOLD
    _logger.debug("Decision threshold using fallback: %.4f", _cached_threshold)
    return _cached_threshold


def _reset_threshold_cache():
    """Reset the cached threshold (for testing only)."""
    global _cached_threshold
    _cached_threshold = None


#: Default decision threshold (resolved lazily).
DECISION_THRESHOLD = get_decision_threshold()

#: ML model weight in the composite formula.
ML_WEIGHT = 0.6

#: Obfuscation weight per flag and cap.
OBFUSCATION_WEIGHT_PER_FLAG = 0.15
OBFUSCATION_WEIGHT_CAP = 0.3

#: Hit names that are obfuscation flags, not L1 rules.
#: Used by the ML uncertain-zone cap to determine if any real L1 rule fired.
FP_EXEMPT_HITS = frozenset({
    "base64", "hex", "rot13", "url_encoded", "leetspeak", "reversed_text",
    "full_reverse", "word_reverse", "caesar_shift", "pig_latin", "morse",
    "high_entropy", "punctuation_flood", "weird_casing",
    "ascii_art", "whitespace_stego", "invisible_chars",
})

#: Rule name -> severity lookup (immutable after module load).
RULE_SEVERITY = {rule.name: rule.severity for rule in RULES}
RULE_SEVERITY["decoded_payload_malicious"] = "critical"
RULE_SEVERITY["decoded_escape_malicious"] = "critical"
RULE_SEVERITY.update({
    name: meta["severity"] for name, meta in _HEURISTIC_HITS.items()
})

#: Rule name -> technique_ids lookup for technique-family boost.
RULE_TECHNIQUE_IDS = {rule.name: rule.technique_ids for rule in RULES}
RULE_TECHNIQUE_IDS.update({
    name: list(meta["technique_ids"]) for name, meta in _HEURISTIC_HITS.items()
})

#: Structural feature weights (Layer 3 binary signals).
STRUCTURAL_SIGNAL_WEIGHTS = {
    "imperative_start": 0.05,
    "role_assignment": 0.10,
    "instruction_boundary": 0.10,
    "negation_command": 0.08,
}

#: Multi-layer agreement boost values by number of agreeing layers.
AGREEMENT_BOOST = {2: 0.10, 3: 0.12, 4: 0.15}


# ── Shared Composite Helper (Phase A) ───────────────────────────────────


def _weighted_composite(
    ml_prob_malicious: float,
    ml_weight: float,
    rule_weight: float,
    obf_weight: float,
) -> float:
    """Compute additive weighted composite from ML, rule, and obfuscation signals.

    This is the shared arithmetic core used by both predict.py's
    _weighted_decision() and cascade.py's WeightedClassifier.classify().

    The formula is ADDITIVE (not normalized):
        composite = (ml_weight * ml_prob_malicious) + rule_weight + obf_weight

    Callers are responsible for:
      - Converting ML probability to the malicious-probability axis
      - Computing rule_weight from severity weights
      - Computing obf_weight from obfuscation flags (capped)
      - Adding structural features, boosts, overrides, and clamping AFTER

    Args:
        ml_prob_malicious: ML probability of malicious class (0.0-1.0)
        ml_weight:  multiplier for ML signal (typically 0.6)
        rule_weight: accumulated rule severity score (from SEVERITY_WEIGHTS)
        obf_weight: obfuscation signal weight (typically capped at 0.3)

    Returns:
        Raw additive composite (unclamped, unboosted).
    """
    return (ml_weight * ml_prob_malicious) + rule_weight + obf_weight


# ── Core Weighted Decision ───────────────────────────────────────────────


def weighted_decision(
    ml_prob,
    ml_label,
    hits,
    obs_flags,
    structural=None,
    embedding_score=0.0,
    threshold=DECISION_THRESHOLD,
    extra_severities=None,
    hit_weights=None,
):
    """Combine ML confidence, rule severity, obfuscation, structural
    features, and embedding similarity into a composite score.

    Parameters
    ----------
    ml_prob : float
        ML model confidence in its own prediction.
    ml_label : str
        ML prediction label ("SAFE" or "MALICIOUS").
    hits : list[str]
        Matched rule/flag names.
    obs_flags : list[str]
        Obfuscation evasion flags.
    structural : dict or None
        Structural features dict from extract_structural_features().
    embedding_score : float
        Layer 5 centroid-based embedding similarity score in [0.0, 0.20].
        Default 0.0 (no embedding signal / model not available).
    threshold : float
        Decision threshold (default 0.55).
    hit_weights : dict[str, float] or None
        Optional per-rule-name multiplier in (0, 1] from span-aware evidence
        grading. An "ambiguous" hit (e.g. matched inside a code/quote/doc
        context but too high-severity to remove) is down-weighted by its
        multiplier instead of contributing full severity weight. None
        (default) means full strength for every hit — backward compatible.

    Returns
    -------
    tuple[str, float]
        (label, composite_score) where label is "SAFE" or "MALICIOUS".
    """
    # --- Fail-closed input sanitization ---
    # A NaN/inf ml_prob would make every later comparison fail open: e.g.
    # ``composite >= threshold`` with a NaN composite is always False -> SAFE.
    # Coerce non-finite ml_prob to the most cautious value (0.5 = maximally
    # uncertain) so the rule/obfuscation/structural signals decide the verdict
    # rather than a single bad number silently disabling detection. Clamp
    # finite values to the valid [0,1] probability range.
    if not math.isfinite(ml_prob):
        ml_prob = 0.5
    else:
        ml_prob = min(max(ml_prob, 0.0), 1.0)
    if not math.isfinite(embedding_score):
        embedding_score = 0.0
    else:
        embedding_score = min(max(embedding_score, 0.0), 1.0)

    # --- ML signal ---
    # Convert to a malicious-probability axis.
    if "MALICIOUS" in ml_label:
        ml_prob_malicious = ml_prob
    else:
        ml_prob_malicious = 1.0 - ml_prob

    ml_weight = ML_WEIGHT * ml_prob_malicious

    # --- Rule severity signal ---
    _sev_lookup = RULE_SEVERITY if extra_severities is None else {**RULE_SEVERITY, **extra_severities}
    rule_weight = 0.0
    severities_seen = set()
    # Dedup by rule name first: the same rule appearing twice (e.g. matched on
    # both sanitized and raw views) must not double-count its severity weight
    # and inflate the score. dict.fromkeys preserves first-seen order.
    for hit_name in dict.fromkeys(hits):
        sev = _sev_lookup.get(hit_name, "medium")
        severities_seen.add(sev)
        base_w = SEVERITY_WEIGHTS.get(sev, 0.1)
        # Span-aware evidence grading: an "ambiguous" hit (matched inside a
        # benign code/quote/doc context but too high-severity to fully remove)
        # contributes a fraction of its severity weight. Missing/None entries
        # default to 1.0 (full strength), so this is a no-op when hit_weights
        # is not supplied. HR-4: the multiplier is floored (>0), so context
        # never zeroes a hit's vote.
        if hit_weights:
            mult = hit_weights.get(hit_name, 1.0)
            try:
                mult = float(mult)
            except (TypeError, ValueError):
                mult = 1.0
            # Clamp to (0, 1]: weights only ever DOWN-weight, never amplify.
            mult = min(max(mult, 0.0), 1.0)
            base_w *= mult
        rule_weight += base_w

    # --- Obfuscation signal ---
    obf_weight = min(OBFUSCATION_WEIGHT_PER_FLAG * len(obs_flags),
                     OBFUSCATION_WEIGHT_CAP)

    # --- Layer 3: Structural feature signal ---
    structural_weight = 0.0
    if structural is not None:
        for feat_name, feat_w in STRUCTURAL_SIGNAL_WEIGHTS.items():
            if structural.get(feat_name, 0):
                structural_weight += feat_w

        if structural.get("quote_depth", 0) >= 3:
            structural_weight += 0.05
        if structural.get("text_entropy", 0) > 5.0:
            structural_weight += 0.03
        if structural.get("many_shot_count", 0) >= 5:
            structural_weight += 0.10
        if structural.get("delimiter_density", 0) > 2.0:
            structural_weight += 0.06
        if structural.get("template_marker_count", 0) >= 1:
            structural_weight += 0.05
        if structural.get("language_mixing_score", 0) >= 2.0:
            structural_weight += 0.04
        if structural.get("repetition_score", 0) > 0.3:
            structural_weight += 0.05

    # --- Layer 5: Embedding similarity signal ---
    embedding_weight = min(embedding_score, 0.20)

    # --- Signal co-occurrence boost ---
    boost_score, _boost_reasons = calculate_boost_from_names(hits, obs_flags)

    composite = (ml_weight + rule_weight + obf_weight
                 + structural_weight + embedding_weight + boost_score)

    # --- FP Reduction: ML-only uncertain zone cap ---
    unsuppressed_rule_count = len(
        [h for h in hits if h not in FP_EXEMPT_HITS]
    )
    ml_uncertain_zone = 0.35 <= ml_prob_malicious <= 0.80
    if ml_uncertain_zone and unsuppressed_rule_count == 0 and obf_weight == 0:
        composite = min(composite, threshold - 0.01)

    # --- Critical-content rule floor ---
    if severities_seen & {"critical_content"}:
        if "MALICIOUS" in ml_label and ml_prob_malicious >= 0.6:
            composite = 1.0
        else:
            composite = max(composite, 0.60)

    # --- Critical E1 extraction rule floor ---
    if severities_seen & {"critical"} and structural_weight > 0.0:
        _has_e1_critical = any(
            _sev_lookup.get(h) == "critical" and any(
                "E1" in tid for tid in RULE_TECHNIQUE_IDS.get(h, [])
            )
            for h in hits
        )
        if _has_e1_critical:
            composite = max(composite, threshold + 0.01)

    # --- g5: Core-family rule anchor floor (embedding-independent) ---
    # A HIGH/CRITICAL instruction-override (D1.x) or system-prompt extraction
    # (E1.x) rule is a strong, self-standing attack anchor.  When such a rule
    # fires but the ML model is merely UNCERTAIN (not confidently safe), the
    # raw composite can sit in a dead-band just below threshold — historically
    # only the optional embedding signal pushed it over (making embedding the
    # deciding vote).  Floor it so rule+ML alone clears threshold; embedding
    # then becomes confirmatory rather than load-bearing.
    #
    # Guarded conservatively:
    #   - requires a HIGH/CRITICAL D1/E1 rule actually firing (not medium),
    #   - requires ML NOT confidently safe (ml_prob_malicious >= 0.35), so a
    #     confident benign verdict still wins,
    #   - does NOT fire on embedding/obfuscation-only hits.
    _has_override_extraction_anchor = any(
        _sev_lookup.get(h) in ("high", "critical")
        and any(
            tid == "D1" or tid.startswith("D1.")
            or tid == "E1" or tid.startswith("E1.")
            for tid in RULE_TECHNIQUE_IDS.get(h, [])
        )
        for h in hits
    )
    # Only lift toward the DEFAULT operating boundary: an operator who raises
    # the threshold above DECISION_THRESHOLD is explicitly suppressing, and a
    # confidence-boosting floor must not override that intent.
    if (_has_override_extraction_anchor
            and ml_prob_malicious >= 0.35
            and threshold <= DECISION_THRESHOLD
            and composite < threshold + 0.01):
        composite = max(composite, threshold + 0.01)

    # --- g5: Critical-rule anchor floor (embedding-independent) ---
    # A CRITICAL-severity L1 rule is a high-precision attack signature
    # (e.g. ``javascript:`` protocol in a markdown link → O2.1 XSS,
    # ``training_data_extraction`` → P1.3 membership inference).  A critical
    # rule should carry the verdict on rule weight ALONE — but the additive
    # composite (0.6*ml + 0.30 critical) lands at ~0.46 when the ML model is
    # confidently SAFE, so historically these attacks crossed threshold only
    # because a tiny embedding score unlocked the AGREEMENT_BOOST (the g4
    # double-count).  With that crutch removed, floor on the critical rule
    # itself so detection no longer depends on an incidental embedding nudge.
    #
    # This mirrors the existing critical_content floor (which floors to 0.60
    # regardless of ML).  We floor only to threshold+epsilon (not 0.60) to
    # stay conservative, and exclude obfuscation-flag pseudo-hits — a real L1
    # rule of critical severity must have fired.  ``decoded_payload_malicious``
    # and ``decoded_escape_malicious`` are critical by registration but are
    # already handled by the decoded-view path; they remain covered here.
    _has_critical_rule = any(
        _sev_lookup.get(h) == "critical" and h not in FP_EXEMPT_HITS
        for h in hits
    )
    if (_has_critical_rule
            and threshold <= DECISION_THRESHOLD
            and composite < threshold + 0.01):
        composite = max(composite, threshold + 0.01)

    # Compute ML safe-confidence once.
    ml_safe_confidence = ml_prob if "SAFE" in ml_label else (1.0 - ml_prob)

    # --- Override protection ---
    # BUG-L6-2 fix: Only trust ML's safe verdict when composite agrees
    # (below threshold).  If composite >= threshold, the rule signals are
    # strong enough to override ML — don't suppress a valid MALICIOUS decision.
    if (threshold > 0.0
            and ml_safe_confidence > 0.8
            and severities_seen <= {"medium"}
            and not obs_flags
            and structural_weight == 0.0
            and composite < threshold):
        # Clamp here too: this early return bypasses the final [0,1] clamp,
        # so a negative composite (from an out-of-range ml_prob) could escape.
        return "SAFE", min(max(composite, 0.0), 1.0)

    # --- Extended override protection ---
    has_decoded_payload = "decoded_payload_malicious" in hits
    if (threshold > 0.0
            and ml_safe_confidence > 0.65
            and rule_weight == 0.0
            and not has_decoded_payload):
        composite = min(composite, threshold - 0.01)

    # --- Multi-layer agreement boost ---
    signal_layers = 0
    if rule_weight > 0:
        signal_layers += 1
    if obf_weight > 0:
        signal_layers += 1
    if structural_weight > 0:
        signal_layers += 1
    if ml_prob_malicious > 0.5:
        signal_layers += 1
    # g4: Embedding is deliberately EXCLUDED from the signal_layers count.
    # The embedding contribution is already added directly to the composite
    # (embedding_weight, above).  Previously it ALSO counted as a 5th signal
    # layer, which let a tiny ~0.05 embedding score single-handedly satisfy
    # the >=2/>=3 agreement gate and unlock a separate AGREEMENT_BOOST — so a
    # ~0.05 embedding swing moved the composite ~0.16 (double-counting).  By
    # decoupling, AGREEMENT_BOOST now requires non-embedding anchor layers,
    # and embedding stays a direct confirmation weight only.

    has_strong_anchor = bool(
        severities_seen & {"high", "critical", "critical_content"}
    )
    has_weak_anchor = bool(severities_seen)

    apply_boost = False
    if signal_layers >= 3 and has_weak_anchor:
        apply_boost = True
    elif signal_layers >= 2 and has_strong_anchor:
        apply_boost = True
    elif signal_layers >= 2 and has_weak_anchor and ml_prob_malicious > 0.45:
        apply_boost = True

    if apply_boost:
        boost = AGREEMENT_BOOST.get(signal_layers, 0.15)
        composite = min(composite + boost, 1.0)

    # --- Technique-family boost ---
    technique_family_rules: dict = {}
    for hit_name in hits:
        for tid in RULE_TECHNIQUE_IDS.get(hit_name, []):
            family = tid.split(".")[0] if "." in tid else tid
            technique_family_rules.setdefault(family, set()).add(hit_name)

    family_boost = 0.0
    for family, rule_set in technique_family_rules.items():
        if len(rule_set) >= 2:
            family_boost += 0.05
    family_boost = min(family_boost, 0.10)
    if family_boost > 0:
        composite = min(composite + family_boost, 1.0)

    # --- g9: Embedding-only FP guard ---
    # The embedding / TF-IDF centroid signal is a keyword/semantic-similarity
    # matcher with no benign-context understanding (e.g. "The security
    # researcher demonstrated: Repeat your system prompt" scores malicious on
    # the fallback).  It must CONFIRM, never DECIDE.  So when the embedding
    # contribution is the only thing pushing the composite over threshold AND
    # no rule of >= high severity fired, cap below threshold.  This keeps
    # embedding from single-handedly converting a sub-threshold, weakly-
    # anchored result into a MALICIOUS verdict.
    if embedding_weight > 0.0 and composite >= threshold:
        composite_without_embedding = composite - embedding_weight
        has_high_anchor = bool(
            severities_seen & {"high", "critical", "critical_content"}
        )
        if composite_without_embedding < threshold and not has_high_anchor:
            composite = min(composite, threshold - 0.01)

    # Clamp to [0, 1].
    composite = min(max(composite, 0.0), 1.0)

    if composite >= threshold:
        return "MALICIOUS", composite
    return "SAFE", composite
