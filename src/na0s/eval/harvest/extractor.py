"""IntelScenarioExtractor — turn REAL attack strings into DRAFT F14 scenarios.

This is the deterministic, offline core of the ``threat-intel-harvester``
capability. It converts attack strings that have already been *synced* from
threat intel (AIID incident text via :class:`~na0s.layer15.incident_to_sample`,
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
   :class:`~na0s.layer15.incident_to_sample.IncidentToSamplePipeline` is the
   caller's concern — this module consumes the *output* (real text) only.
5. **Nothing is auto-promoted.** Output is DRAFT YAML for human review.

Skips are always collected and returned, never silently dropped — a silently
truncated batch reads as "we covered everything" when we did not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from na0s.eval.scenarios import Scenario, ScenarioTurn, ScenarioType
from na0s.eval.harvest.taxonomy import TaxonomyValidator

logger = logging.getLogger(__name__)

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
    """Outcome of a batch extraction: emitted scenarios + skipped inputs."""

    scenarios: list[Scenario] = field(default_factory=list)
    skipped: list[SkippedInput] = field(default_factory=list)

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

        Returns
        -------
        HarvestReport
        """
        report = HarvestReport()
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

            payload = record.get("payload")
            raw_turns = record.get("turns")
            has_payload = bool(isinstance(payload, str) and payload.strip())
            has_turns = bool(isinstance(raw_turns, list) and raw_turns)

            if not has_payload and not has_turns:
                report.skipped.append(
                    SkippedInput(
                        ident,
                        "no real payload or turns (description-only inputs are "
                        "NOT payloads) — needs manual payload authoring",
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

            turns = self._build_turns(raw_turns) if has_turns else None
            if has_turns and not turns:
                report.skipped.append(
                    SkippedInput(
                        ident,
                        "turns present but none had text — needs manual "
                        "payload authoring",
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

            name = str(ident).replace(" ", "_")
            try:
                attack = self.build_scenario(
                    name=name,
                    attack_category=category,
                    severity=severity,
                    provenance=provenance,
                    payload=payload if has_payload else None,
                    turns=turns,
                    expected_verdict=record.get("expected_verdict", "blocked"),
                    description=record.get("description", ""),
                    tags=list(record.get("tags", []) or []),
                )
            except ValueError as exc:
                # build_scenario re-validates; surface rather than crash the batch.
                report.skipped.append(SkippedInput(ident, str(exc)))
                continue

            # Optional benign sibling (over-refusal control). Pass-through only:
            # emitted ONLY when the record explicitly carries benign text, never
            # synthesized. A malformed benign half is skipped; the attack stays.
            benign = self._maybe_build_benign(record, name, provenance, report)
            report.scenarios.append(attack)
            if benign is not None:
                attack.paired_benign_id = benign.stable_id
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
        benign_turns = (
            self._build_turns(raw_benign_turns)
            if isinstance(raw_benign_turns, list)
            else None
        )
        try:
            return self.build_benign_sibling(
                name=f"{attack_name}__benign",
                provenance=provenance,
                benign_payload=(
                    benign_payload if isinstance(benign_payload, str) else None
                ),
                benign_turns=benign_turns,
                description=record.get("benign_description", ""),
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

        payload = [s.to_dict() for s in scenarios]
        out_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info(
            "Wrote %d draft scenario(s) to %s", len(scenarios), out_path
        )
        return out_path
