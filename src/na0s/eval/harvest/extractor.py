"""IntelScenarioExtractor — turn REAL attack strings into DRAFT F14 scenarios.

This is the deterministic, offline core of the ``threat-intel-harvester``
capability. It converts attack strings that have already been *synced* from
threat intel (AIID incident text via :class:`~na0s.threat_intel.incident_to_sample`,
benchmark prompt columns, or structured intel records) into provenance-traced,
taxonomy-validated DRAFT scenarios written to ``data/eval/scenarios/_drafts/``.

Design guarantees (these are the security contract, not nice-to-haves):

1. **Never fabricate a payload from a description.** A description of a technique
   ("attackers use base64 to hide instructions") is metadata, not an attack
   string. Any input lacking a real ``payload``/``turns`` is SKIPPED and
   reported as ``"needs manual payload authoring"`` — never emitted with the
   description folded in as the payload.
2. **Every ``attack_category`` is validated against ``data/taxonomy.yaml``.** An
   unknown code raises ``ValueError`` (for the explicit builder) or is skipped
   and reported (for the batch paths). Inventing codes is an injection vector.
3. **Provenance is mandatory.** Every emitted scenario carries
   ``source="harvest_pipeline"`` and the origin + retrieval date folded into
   ``description``.
4. **No network, no external LLM.** Nothing here calls out; the LLM path in
   :class:`~na0s.threat_intel.incident_to_sample.IncidentToSamplePipeline` is the
   caller's concern — this module consumes the *output* (real text) only.
5. **Nothing is auto-promoted.** Output is DRAFT YAML for human review.

Skips are always collected and returned, never silently dropped — a silently
truncated batch reads as "we covered everything" when we did not.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from na0s.eval.scenarios import Scenario, ScenarioTurn, ScenarioType
from na0s.eval.harvest.taxonomy import TaxonomyValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Untrusted-field hardening.
#
# Every field in a harvested record (payload, turn text, name, description,
# tags, expected_verdict) is ATTACKER-CONTROLLED external threat-intel text.
# The extractor must treat all of it as hostile data, never as instructions
# and never as trusted control values. These limits/sanitizers are the
# defensive boundary between the untrusted intel feed and the DRAFT library.
# ---------------------------------------------------------------------------

# Max characters for a single payload or turn text. An oversized blob is a
# denial/poisoning vector (bloats drafts, can wedge downstream tooling); we
# TRUNCATE (recording a flag) rather than silently dropping the record.
MAX_PAYLOAD_CHARS = 8000

# Max number of turns kept from a single multi-turn record. Extra turns are
# dropped and the truncation flagged.
MAX_TURNS = 64

# Verdicts a harvested ATTACK record is allowed to assert. An intel feed that
# labels an attack record "allowed"/"benign" is a draft-poisoning attempt:
# it would teach the eval set to PASS an attack. We force the conservative
# verdict for harvested attacks (benign siblings go through a separate,
# explicit pass-through path with its own hardcoded "allowed").
_CONSERVATIVE_VERDICT = "blocked"
_BENIGN_VERDICTS = {"allowed", "allow", "benign", "pass", "safe", "ok"}

# Truncation marker appended to a truncated string so a human reviewer sees it.
_TRUNCATION_MARKER = "…[harvest-truncated]"


def _strip_control_chars(text: str) -> str:
    """Remove C0/C1 control chars (keeping \\n and \\t) from untrusted text.

    Control characters in harvested payloads are an obfuscation / terminal-
    injection / parser-confusion vector. We keep newline and tab (meaningful
    in real prompts) and strip everything else in the C0 (U+0000-U+001F) and
    C1 (U+007F-U+009F) ranges, plus the Unicode "Cc"/"Cf" format/control
    categories (zero-width joiners, bidi overrides, etc.).
    """
    out_chars: list[str] = []
    for ch in text:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue
        codepoint = ord(ch)
        if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf"):
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _is_text_blob(value: Any) -> bool:
    """True iff ``value`` is a str that, after stripping, has printable text.

    Rejects non-str types and strings that are entirely control/whitespace
    once sanitized (a non-text blob masquerading as a payload).
    """
    if not isinstance(value, str):
        return False
    return bool(_strip_control_chars(value).strip())


def _sanitize_text(value: str) -> tuple[str, bool]:
    """Sanitize one untrusted text field.

    Returns ``(clean_text, truncated)``:

    - strips C0/C1 control chars (keeping \\n / \\t),
    - caps length at :data:`MAX_PAYLOAD_CHARS`, appending a visible marker and
      setting ``truncated=True`` when the cap is hit.

    Length is measured AFTER control-char stripping so an attacker can't pad
    past the cap with junk that we'd discard anyway.
    """
    cleaned = _strip_control_chars(value)
    truncated = False
    if len(cleaned) > MAX_PAYLOAD_CHARS:
        cleaned = cleaned[:MAX_PAYLOAD_CHARS] + _TRUNCATION_MARKER
        truncated = True
    return cleaned, truncated


_TAG_RE = re.compile(r"[^0-9A-Za-z_.:\-/ ]+")


def _sanitize_tags(raw_tags: Any) -> list[str]:
    """Coerce an untrusted ``tags`` value into a clean list of short str tags.

    Drops non-str entries, strips control chars, removes characters outside a
    conservative tag charset, and caps tag count + per-tag length so a hostile
    record can't smuggle a payload through the tags field.
    """
    if not isinstance(raw_tags, (list, tuple)):
        return []
    clean: list[str] = []
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        t = _TAG_RE.sub("", _strip_control_chars(tag)).strip()
        if not t:
            continue
        clean.append(t[:64])
        if len(clean) >= 32:
            break
    return clean

# Default landing zone for drafts, per data/eval/scenarios/_drafts/README.md.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"

_HARVEST_SOURCE = "harvest_pipeline"

# Valid severity vocabulary for the Scenario schema (mirrors schema.py).
_VALID_SEVERITIES = ("critical", "high", "medium", "low")
_DEFAULT_SEVERITY = "high"

# Benign-sibling sentinel. Mirrors the existing v0.1 ``*_benign.yaml`` convention
# (``attack_category: BEN``). BEN is a verdict marker (this prompt should be
# ALLOWED), not an attack technique — so the benign path does NOT go through
# taxonomy validation. NOTE: BEN is not yet in data/taxonomy.yaml; formalizing
# it is a tracked TODO (see taxonomy.py).
_BENIGN_CATEGORY = "BEN"


@dataclass(frozen=True)
class IntelProvenance:
    """Where a batch of harvested scenarios came from.

    Attributes
    ----------
    source_slug : str
        Short kebab-case id for the intel source, used as the draft filename
        stem (e.g. ``"aiid-2025-snapshot"``, ``"tensortrust-prompts"``).
    origin : str
        The origin URL or dataset identifier (e.g. an AIID incident URL, a
        benchmark repo URL). Folded verbatim into each scenario's description.
    retrieved : str
        Retrieval date (ISO ``YYYY-MM-DD``) — when this intel was synced.
    """

    source_slug: str
    origin: str
    retrieved: str

    def description_suffix(self) -> str:
        """Render the provenance line appended to every scenario description."""
        return (
            f"[provenance: source={self.source_slug} "
            f"origin={self.origin} retrieved={self.retrieved}]"
        )


@dataclass
class SkippedInput:
    """A harvest input that was NOT turned into a scenario, with the reason.

    Surfaced in :class:`HarvestReport` so nothing is silently dropped.
    """

    identifier: str
    reason: str


@dataclass
class HarvestReport:
    """Outcome of a batch extraction: emitted scenarios + skipped inputs.

    ``notes`` carries non-fatal diagnostics for emitted scenarios — e.g. a
    payload that was truncated to the length cap, a record whose hostile
    ``expected_verdict`` was forced back to the conservative value, or a
    duplicate that was deduplicated. These are NOT skips (the scenario was
    still emitted, except for dedup); surfacing them keeps the batch honest
    rather than silently mutating attacker-supplied data.
    """

    scenarios: list[Scenario] = field(default_factory=list)
    skipped: list[SkippedInput] = field(default_factory=list)
    notes: list[SkippedInput] = field(default_factory=list)

    @property
    def emitted_count(self) -> int:
        return len(self.scenarios)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


class IntelScenarioExtractor:
    """Build DRAFT F14 scenarios from real harvested attack strings.

    Parameters
    ----------
    taxonomy : TaxonomyValidator | None
        Validator for ``attack_category`` codes. Defaults to a fresh
        :class:`TaxonomyValidator` over ``data/taxonomy.yaml``.
    """

    def __init__(self, taxonomy: Optional[TaxonomyValidator] = None) -> None:
        self.taxonomy = taxonomy or TaxonomyValidator()

    # ------------------------------------------------------------------ #
    # Single-scenario builder (raises on bad input — used by callers that
    # have already vetted their data, and by the batch paths below).
    # ------------------------------------------------------------------ #
    def build_scenario(
        self,
        *,
        name: str,
        attack_category: str,
        severity: str,
        provenance: IntelProvenance,
        payload: Optional[str] = None,
        turns: Optional[list[ScenarioTurn]] = None,
        expected_verdict: str = "blocked",
        description: str = "",
        tags: Optional[list[str]] = None,
    ) -> Scenario:
        """Construct a single provenance-traced, validated DRAFT scenario.

        Enforces:

        - ``attack_category`` exists in the taxonomy (else ``ValueError``).
        - exactly one of ``payload`` / ``turns`` is provided (payload XOR turns).
        - ``source`` is forced to ``"harvest_pipeline"``.
        - provenance is folded into ``description``.

        Raises
        ------
        ValueError
            On unknown ``attack_category``, payload/turns ambiguity, or a
            payload that is only whitespace.
        """
        if not self.taxonomy.validate_code(attack_category):
            raise ValueError(
                f"Scenario {name!r}: attack_category {attack_category!r} is not "
                f"in the taxonomy (data/taxonomy.yaml). Refusing to emit — "
                f"inventing codes corrupts the eval library."
            )

        has_payload = bool(payload and payload.strip())
        has_turns = bool(turns)
        if has_payload == has_turns:
            raise ValueError(
                f"Scenario {name!r}: exactly one of payload / turns is required "
                f"(payload XOR turns); got payload={has_payload}, turns={has_turns}."
            )

        if severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"Scenario {name!r}: severity must be one of {_VALID_SEVERITIES}, "
                f"got {severity!r}"
            )

        scenario_type = (
            ScenarioType.SINGLE_PROMPT if has_payload else ScenarioType.MULTI_TURN
        )

        full_description = description.strip()
        suffix = provenance.description_suffix()
        full_description = (
            f"{full_description} {suffix}".strip() if full_description else suffix
        )

        return Scenario(
            name=name,
            type=scenario_type,
            expected_verdict=expected_verdict,
            severity=severity,
            attack_category=attack_category,
            description=full_description,
            payload=payload if has_payload else None,
            turns=list(turns) if has_turns else [],
            source=_HARVEST_SOURCE,
            tags=list(tags or []),
        )

    # ------------------------------------------------------------------ #
    # Benign sibling (over-refusal control) — pass-through only, never synth.
    # ------------------------------------------------------------------ #
    def build_benign_sibling(
        self,
        *,
        name: str,
        provenance: IntelProvenance,
        benign_payload: Optional[str] = None,
        benign_turns: Optional[list[ScenarioTurn]] = None,
        description: str = "",
    ) -> Scenario:
        """Build a benign sibling from EXPLICIT benign text — pass-through only.

        A benign sibling is the over-refusal control for an attack scenario: a
        near-identical-but-harmless prompt the defender must ALLOW. This method
        NEVER synthesizes benign text; it only wraps benign text the intel
        record already provides. If no real benign text is given it raises — a
        fabricated "benign" prompt corrupts over-refusal scoring exactly as a
        fabricated attack corrupts recall.

        The sibling uses ``attack_category="BEN"`` (the sentinel the existing
        ``v0.1/*_benign.yaml`` set uses), ``expected_verdict="allowed"`` and
        ``severity="low"``. BEN is a verdict marker, not an attack technique, so
        this path deliberately does NOT go through taxonomy validation (and BEN
        is itself a tracked taxonomy-formalization TODO — see taxonomy.py).

        Raises
        ------
        ValueError
            If neither / both of ``benign_payload`` / ``benign_turns`` is given,
            or the benign text is only whitespace.
        """
        has_payload = bool(benign_payload and benign_payload.strip())
        has_turns = bool(benign_turns)
        if has_payload == has_turns:
            raise ValueError(
                f"Benign sibling {name!r}: exactly one of benign_payload / "
                f"benign_turns is required (no synthesis); got "
                f"payload={has_payload}, turns={has_turns}."
            )

        scenario_type = (
            ScenarioType.SINGLE_PROMPT if has_payload else ScenarioType.MULTI_TURN
        )
        suffix = provenance.description_suffix()
        full_description = description.strip()
        full_description = (
            f"{full_description} {suffix}".strip() if full_description else suffix
        )

        return Scenario(
            name=name,
            type=scenario_type,
            expected_verdict="allowed",
            severity="low",
            attack_category=_BENIGN_CATEGORY,
            description=full_description,
            payload=benign_payload if has_payload else None,
            turns=list(benign_turns) if has_turns else [],
            source=_HARVEST_SOURCE,
            tags=["benign_sibling", "over_refusal_test"],
        )

    # ------------------------------------------------------------------ #
    # Batch path 1: reuse GeneratedSample (real incident-derived text).
    # ------------------------------------------------------------------ #
    def scenarios_from_generated_samples(
        self,
        samples: list[Any],
        provenance: IntelProvenance,
    ) -> HarvestReport:
        """Convert :class:`GeneratedSample` objects into DRAFT scenarios.

        Each sample's ``text`` is a real attack string; ``category_hint`` is a
        best-guess taxonomy code. Samples are SKIPPED (and reported) when:

        - ``text`` is empty/whitespace (no real payload), or
        - ``category_hint`` is missing or not a valid taxonomy code.

        Severity is looked up from the taxonomy for the mapped category; if the
        taxonomy records no severity, it falls back to ``"high"`` (conservative —
        harvested attacks default to needing a hard block, and a human reviewer
        recalibrates before promotion).

        Returns
        -------
        HarvestReport
            Emitted scenarios + skipped inputs with reasons.
        """
        report = HarvestReport()
        for idx, sample in enumerate(samples):
            text = (getattr(sample, "text", "") or "").strip()
            category = (getattr(sample, "category_hint", "") or "").strip()
            incident_id = getattr(sample, "source_incident_id", "") or f"#{idx}"
            ident = f"sample[{idx}] (incident={incident_id})"

            if not text:
                report.skipped.append(
                    SkippedInput(
                        identifier=ident,
                        reason="no real payload (empty text) — needs manual "
                        "payload authoring",
                    )
                )
                continue
            if not category:
                report.skipped.append(
                    SkippedInput(
                        identifier=ident,
                        reason="no category_hint — needs manual taxonomy mapping",
                    )
                )
                continue
            if not self.taxonomy.validate_code(category):
                report.skipped.append(
                    SkippedInput(
                        identifier=ident,
                        reason=f"category_hint {category!r} not in taxonomy — "
                        "needs manual taxonomy mapping",
                    )
                )
                continue

            severity = self.taxonomy.get_severity(category) or _DEFAULT_SEVERITY
            if severity not in _VALID_SEVERITIES:
                severity = _DEFAULT_SEVERITY

            technique_hint = (
                getattr(sample, "technique_hint", "") or ""
            ).strip()
            tags = [t for t in ("harvested", technique_hint) if t]
            name = f"{provenance.source_slug}__{incident_id}__{idx}".replace(
                " ", "_"
            )
            report.scenarios.append(
                self.build_scenario(
                    name=name,
                    attack_category=category,
                    severity=severity,
                    provenance=provenance,
                    payload=text,
                    description=(
                        f"Harvested from incident {incident_id}."
                    ),
                    tags=tags,
                )
            )
        return report

    # ------------------------------------------------------------------ #
    # Batch path 2: generic structured intel / benchmark rows.
    # ------------------------------------------------------------------ #
    def scenarios_from_records(
        self,
        records: list[dict[str, Any]],
        provenance: IntelProvenance,
        existing_stable_ids: Optional[set[str]] = None,
    ) -> HarvestReport:
        """Convert structured intel/benchmark rows into DRAFT scenarios.

        Each record dict must carry a real ``payload`` (str) OR ``turns``
        (list of ``{"text", "expected_label", "risk_score"?}`` dicts), plus an
        ``attack_category``. Optional keys: ``name``, ``severity``,
        ``expected_verdict``, ``description``, ``tags``, and ``benign_payload``
        / ``benign_turns`` (+ ``benign_description``) to emit a linked benign
        over-refusal sibling (pass-through only — never synthesized).

        Rows are SKIPPED (and reported) when they lack a real payload/turns or
        carry an invalid/absent ``attack_category``. A record whose only
        attack-relevant field is a textual *description* is treated as having no
        payload and skipped — descriptions are NEVER promoted to payloads.

        Untrusted-field hardening applied here (see module docstring):

        - every record is hostile data: payload / turn text / description are
          control-char-stripped and length-capped (truncation flagged in
          ``report.notes``), tags are charset/length-filtered;
        - a non-text or empty-after-sanitization payload is rejected;
        - a harvested ATTACK record may NOT assert a benign verdict — an
          incoming ``expected_verdict`` of allowed/benign is forced back to the
          conservative ``"blocked"`` (the note records the override). Real
          benign over-refusal controls flow through the explicit, pass-through
          ``benign_*`` keys, never through an attack record's verdict;
        - emitted scenarios are deduplicated by ``stable_id`` within the batch
          and against ``existing_stable_ids`` (if provided).

        Parameters
        ----------
        existing_stable_ids : set[str] | None
            Stable IDs of drafts already on disk; a record whose scenario
            hashes to one of these is skipped as a duplicate.

        Returns
        -------
        HarvestReport
        """
        report = HarvestReport()
        seen_ids: set[str] = set(existing_stable_ids or set())
        for idx, record in enumerate(records):
            ident = record.get("name") or f"record[{idx}]"

            category = (record.get("attack_category") or "").strip()
            if not category:
                report.skipped.append(
                    SkippedInput(ident, "missing attack_category")
                )
                continue
            if not self.taxonomy.validate_code(category):
                report.skipped.append(
                    SkippedInput(
                        ident,
                        f"attack_category {category!r} not in taxonomy",
                    )
                )
                continue

            raw_payload = record.get("payload")
            raw_turns = record.get("turns")
            # A payload is only real if it survives control-char stripping with
            # printable text left — a non-text blob is not a payload.
            has_payload = _is_text_blob(raw_payload)
            has_turns = bool(isinstance(raw_turns, list) and raw_turns)

            if not has_payload and not has_turns:
                report.skipped.append(
                    SkippedInput(
                        ident,
                        "no real payload or turns (description-only / non-text "
                        "inputs are NOT payloads) — needs manual payload "
                        "authoring",
                    )
                )
                continue
            if has_payload and has_turns:
                report.skipped.append(
                    SkippedInput(
                        ident,
                        "both payload and turns present — payload XOR turns "
                        "violated; needs manual disambiguation",
                    )
                )
                continue

            truncated = False
            payload: Optional[str] = None
            if has_payload:
                payload, payload_trunc = _sanitize_text(raw_payload)
                truncated = truncated or payload_trunc

            turns: Optional[list[ScenarioTurn]] = None
            if has_turns:
                turns, turns_trunc = self._build_turns_hardened(raw_turns)
                truncated = truncated or turns_trunc
                if not turns:
                    report.skipped.append(
                        SkippedInput(
                            ident,
                            "turns present but none had real text — needs "
                            "manual payload authoring",
                        )
                    )
                    continue

            severity = (
                record.get("severity")
                or self.taxonomy.get_severity(category)
                or _DEFAULT_SEVERITY
            )
            if severity not in _VALID_SEVERITIES:
                severity = _DEFAULT_SEVERITY

            # Verdict forcing: a harvested attack record is never trusted to
            # declare itself benign/allowed. Force the conservative verdict and
            # record the override so a human reviewer sees the tampering.
            raw_verdict = str(record.get("expected_verdict", "") or "").strip()
            verdict = _CONSERVATIVE_VERDICT
            if raw_verdict and raw_verdict.lower() in _BENIGN_VERDICTS:
                report.notes.append(
                    SkippedInput(
                        ident,
                        f"hostile expected_verdict {raw_verdict!r} on a "
                        f"harvested attack record forced to "
                        f"{_CONSERVATIVE_VERDICT!r} (benign controls must use "
                        f"the explicit benign_* keys)",
                    )
                )
            elif raw_verdict and raw_verdict != _CONSERVATIVE_VERDICT:
                # Any other non-standard verdict (typo, unknown) is also forced
                # to the conservative default; never trusted verbatim.
                verdict = _CONSERVATIVE_VERDICT

            description, desc_trunc = _sanitize_text(
                str(record.get("description", "") or "")
            )
            truncated = truncated or desc_trunc

            name = str(ident).replace(" ", "_")[:200]
            try:
                attack = self.build_scenario(
                    name=name,
                    attack_category=category,
                    severity=severity,
                    provenance=provenance,
                    payload=payload,
                    turns=turns,
                    expected_verdict=verdict,
                    description=description,
                    tags=_sanitize_tags(record.get("tags")),
                )
            except ValueError as exc:
                # build_scenario re-validates; surface rather than crash the batch.
                report.skipped.append(SkippedInput(ident, str(exc)))
                continue

            if truncated:
                report.notes.append(
                    SkippedInput(
                        ident,
                        "payload/turn/description text exceeded "
                        f"MAX_PAYLOAD_CHARS={MAX_PAYLOAD_CHARS} or "
                        f"MAX_TURNS={MAX_TURNS} and was truncated "
                        "(not dropped) — review before promotion",
                    )
                )

            # Dedup by stable_id (content hash) within the batch + vs existing.
            if attack.stable_id in seen_ids:
                report.notes.append(
                    SkippedInput(
                        ident,
                        f"duplicate of an existing/seen draft "
                        f"(stable_id={attack.stable_id}); deduplicated",
                    )
                )
                continue

            # Optional benign sibling (over-refusal control). Pass-through only:
            # emitted ONLY when the record explicitly carries benign text, never
            # synthesized. A malformed benign half is skipped; the attack stays.
            benign = self._maybe_build_benign(record, name, provenance, report)
            seen_ids.add(attack.stable_id)
            report.scenarios.append(attack)
            if benign is not None and benign.stable_id not in seen_ids:
                attack.paired_benign_id = benign.stable_id
                seen_ids.add(benign.stable_id)
                report.scenarios.append(benign)
        return report

    def _maybe_build_benign(
        self,
        record: dict[str, Any],
        attack_name: str,
        provenance: IntelProvenance,
        report: HarvestReport,
    ) -> Optional[Scenario]:
        """Build a benign sibling iff the record explicitly provides benign text.

        Returns None when no benign text is present (the common case) or when the
        benign half is malformed — in which case the reason is recorded in
        ``report.skipped`` and the caller still keeps the attack scenario.
        """
        benign_payload = record.get("benign_payload")
        raw_benign_turns = record.get("benign_turns")
        if not benign_payload and not raw_benign_turns:
            return None
        # Benign text is also untrusted intel data: sanitize + cap it, same as
        # the attack half. (The benign VERDICT is hardcoded "allowed" in
        # build_benign_sibling — only the TEXT comes from the record.)
        clean_payload: Optional[str] = None
        if _is_text_blob(benign_payload):
            clean_payload, _ = _sanitize_text(benign_payload)
        benign_turns = (
            self._build_turns_hardened(raw_benign_turns)[0]
            if isinstance(raw_benign_turns, list)
            else None
        )
        clean_desc, _ = _sanitize_text(
            str(record.get("benign_description", "") or "")
        )
        try:
            return self.build_benign_sibling(
                name=f"{attack_name}__benign",
                provenance=provenance,
                benign_payload=clean_payload,
                benign_turns=benign_turns,
                description=clean_desc,
            )
        except ValueError as exc:
            report.skipped.append(
                SkippedInput(f"{attack_name}__benign", str(exc))
            )
            return None

    @staticmethod
    def _build_turns(raw_turns: list[Any]) -> list[ScenarioTurn]:
        """Build ScenarioTurns from record dicts, dropping turns without text."""
        turns: list[ScenarioTurn] = []
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            turns.append(
                ScenarioTurn(
                    text=turn["text"],
                    expected_label=turn.get("expected_label", "malicious"),
                    risk_score=turn.get("risk_score"),
                )
            )
        return turns

    @staticmethod
    def _build_turns_hardened(
        raw_turns: list[Any],
    ) -> tuple[list[ScenarioTurn], bool]:
        """Build ScenarioTurns from untrusted record dicts, sanitized + capped.

        Returns ``(turns, truncated)``. Each turn's ``text`` is control-char
        stripped and length-capped; turns left with no printable text are
        dropped. The number of turns is capped at :data:`MAX_TURNS`; extra
        turns are dropped and ``truncated`` is set. ``expected_label`` is only
        accepted from a small allowlist (defaulting to ``"malicious"``) so a
        hostile record can't relabel an attack turn as benign.
        """
        turns: list[ScenarioTurn] = []
        truncated = False
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            raw_text = turn.get("text")
            if not _is_text_blob(raw_text):
                continue
            if len(turns) >= MAX_TURNS:
                truncated = True
                break
            text, text_trunc = _sanitize_text(raw_text)
            truncated = truncated or text_trunc
            label = turn.get("expected_label", "malicious")
            if label not in ("malicious", "benign", "suspicious"):
                label = "malicious"
            risk = turn.get("risk_score")
            if not isinstance(risk, (int, float)):
                risk = None
            turns.append(
                ScenarioTurn(
                    text=text,
                    expected_label=label,
                    risk_score=risk,
                )
            )
        # If we hit the cap mid-stream, the break above already flagged it; also
        # flag when the input simply had more than MAX_TURNS entries with text.
        return turns, truncated

    # ------------------------------------------------------------------ #
    # Writer.
    # ------------------------------------------------------------------ #
    def write_drafts(
        self,
        scenarios: list[Scenario],
        output_dir: Optional[Path] = None,
        source_slug: str = "harvest",
    ) -> Path:
        """Write scenarios to ``<output_dir>/<source_slug>.yaml`` and return path.

        Serializes via :meth:`Scenario.to_dict` so the file round-trips cleanly
        through :class:`~na0s.eval.scenarios.ScenarioLoader`. Creates
        ``output_dir`` if needed. Defaults to ``data/eval/scenarios/_drafts``.

        Raises
        ------
        ValueError
            If ``scenarios`` is empty (nothing to write).
        """
        if not scenarios:
            raise ValueError("write_drafts: no scenarios to write")

        out_dir = Path(output_dir or DEFAULT_DRAFTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source_slug}.yaml"

        # Dedup by stable_id: against any drafts already on disk at this path
        # (so re-running a harvest is idempotent, not duplicating) and within
        # the batch itself. First occurrence wins; later duplicates are dropped.
        existing = self._load_existing_stable_ids(out_path)
        seen: set[str] = set(existing)
        deduped: list[Scenario] = []
        for scenario in scenarios:
            sid = scenario.stable_id
            if sid in seen:
                logger.info(
                    "write_drafts: skipping duplicate scenario %r "
                    "(stable_id=%s)",
                    scenario.name,
                    sid,
                )
                continue
            seen.add(sid)
            deduped.append(scenario)

        if not deduped:
            raise ValueError(
                "write_drafts: all scenarios were duplicates of existing "
                f"drafts at {out_path}; nothing new to write"
            )

        # Preserve any pre-existing drafts: append the new (deduped) ones.
        prior = self._load_existing_scenarios_dicts(out_path)
        payload = prior + [s.to_dict() for s in deduped]
        out_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info(
            "Wrote %d new draft scenario(s) (+%d pre-existing) to %s",
            len(deduped),
            len(prior),
            out_path,
        )
        return out_path

    @staticmethod
    def _load_existing_scenarios_dicts(path: Path) -> list[dict[str, Any]]:
        """Load raw scenario dicts already at ``path`` (best-effort, never raises).

        A malformed/unreadable existing file is treated as empty so a poisoned
        or partial prior draft can't crash the writer; the new drafts are still
        written (overwriting the unreadable content).
        """
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return []
        if not isinstance(data, list):
            return []
        return [d for d in data if isinstance(d, dict)]

    @classmethod
    def _load_existing_stable_ids(cls, path: Path) -> set[str]:
        """Stable IDs already serialized at ``path`` (best-effort)."""
        ids: set[str] = set()
        for d in cls._load_existing_scenarios_dicts(path):
            sid = d.get("stable_id")
            if isinstance(sid, str) and sid:
                ids.add(sid)
        return ids
