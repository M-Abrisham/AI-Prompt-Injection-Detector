"""F14 scenario-admission gate — deterministic, REPORT-ONLY validation of DRAFT scenarios.

This gate inspects candidate scenarios sitting in ``data/eval/scenarios/_drafts/``
and produces a per-scenario ADMIT/REJECT report. It is the automated pre-flight
check a human runs *before* hand-promoting a draft into the live ``v0.1/`` set via
PR (see ``data/eval/scenarios/_drafts/README.md``).

DESIGN INVARIANTS (do not weaken)
---------------------------------
* **REPORT-ONLY.** The gate NEVER writes, moves, or copies scenario files into
  ``v0.1/``. It only reads. Promotion stays a human PR step. A REJECT means
  "do not promote this draft"; it does not delete or quarantine anything.
* **Deterministic + offline.** No network, no external LLM, no embedding model
  loaded by default. Every check is a pure function of the inputs on disk.
* **Fail-loud on real leaks, soft on provenance.** Schema / taxonomy / exact
  collision / near-dup / benign-twin failures BLOCK (REJECT). The trust signal
  is advisory only (WARNING) because drafts are human-reviewed downstream.

CHECK ORDER (per draft scenario)
--------------------------------
1. ``schema``        — loads via :class:`ScenarioLoader` (BLOCK on failure).
2. ``taxonomy``      — ``attack_category`` valid via :class:`TaxonomyValidator`
                       (BLOCK). Benign scenarios (``expected_verdict == "allowed"``)
                       are EXEMPT — benign siblings use ``BEN``, which is not an
                       attack-taxonomy code.
3. ``exact_decontam``— draft ``stable_id`` must not collide with the training
                       corpus OR the live ``v0.1/`` set (BLOCK — a collision is a
                       train/eval leak).
4. ``near_dup_decontam`` — MinHash/Jaccard similarity vs training + live texts;
                       BLOCK if ``>= near_dup_threshold`` (default 0.85).
                       **This is a Jaccard/MinHash PROXY, not true embedding
                       cosine.** See the ``embedding_fn`` hook + TODO below.
5. ``benign_twin``   — if ``paired_benign_id`` is set, BLOCK unless a scenario
                       with that ``stable_id`` exists (draft or live) AND its
                       ``expected_verdict == "allowed"``.
6. ``trust``         — SOFT (WARNING only). A provenance/source trust signal is
                       derived and reported but never blocks.

DECONTAM PROXY NOTE
-------------------
Step 4 is a *lexical* near-duplicate proxy built on character-3-gram MinHash
(``na0s.dataset.near_duplicate``). It catches copy-paste / light-edit overlap
but NOT paraphrase. True semantic decontamination would compute embedding
cosine similarity; that is intentionally deferred to avoid pulling
``sentence-transformers`` into a fast deterministic gate. An ``embedding_fn``
hook is exposed so a caller can supply a real embedder later; when supplied,
the gate additionally checks cosine and reports both methods. The report always
states which method(s) actually ran — it never silently skips decontam.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from na0s.dataset.near_duplicate import (
    MINHASH_NUM_HASHES,
    jaccard_from_minhash,
    minhash_signature,
)
from na0s.eval.scenarios import Scenario, ScenarioLoader, load_scenarios_dir

# Optional dependencies — imported defensively so the gate degrades gracefully
# rather than crashing if the surrounding repo layout shifts.
try:  # pragma: no cover - exercised indirectly
    from na0s.eval.harvest.taxonomy import TaxonomyValidator
except Exception:  # pragma: no cover
    TaxonomyValidator = None  # type: ignore[assignment]

try:  # pragma: no cover
    from na0s.dataset.schema import Na0SSample
except Exception:  # pragma: no cover
    Na0SSample = None  # type: ignore[assignment]


DEFAULT_NEAR_DUP_THRESHOLD = 0.85

# Default repo-relative locations (resolved from this file:
# src/na0s/eval/scenarios/admission_gate.py -> repo root is 4 parents up).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DRAFTS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "_drafts"
_DEFAULT_LIVE_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "v0.1"
_DEFAULT_TRAINING_DIRS = (
    _REPO_ROOT / "data" / "raw",
    _REPO_ROOT / "data" / "aggregated",
)


# Embedding hook: maps text -> dense vector. None means "no semantic check".
EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass
class AdmissionResult:
    """Outcome of running every admission check against one draft scenario."""

    name: str
    status: str  # "ADMIT" | "REJECT"
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return self.status == "ADMIT"


@dataclass
class AdmissionReport:
    """Aggregate report over all drafts in a directory."""

    results: list[AdmissionResult] = field(default_factory=list)
    # Provenance of what the gate actually ran (for honest reporting).
    training_sample_count: int = 0
    live_scenario_count: int = 0
    decontam_methods: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> list[AdmissionResult]:
        return [r for r in self.results if r.status == "ADMIT"]

    @property
    def rejected(self) -> list[AdmissionResult]:
        return [r for r in self.results if r.status == "REJECT"]

    @property
    def any_rejected(self) -> bool:
        return any(r.status == "REJECT" for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)


class ScenarioAdmissionGate:
    """Deterministic, report-only admission gate for DRAFT F14 scenarios.

    Parameters
    ----------
    taxonomy : TaxonomyValidator | None
        Validator for ``attack_category``. If None, one is constructed from the
        canonical ``data/taxonomy.yaml``. If that fails, taxonomy checks become a
        WARNING (reported, never block) so the gate still runs offline.
    training_dirs : sequence of paths | None
        Directories scanned for legacy training CSVs to harvest stable_ids +
        texts for decontamination. Missing/empty dirs are fine (reported as
        "0 training samples"). Defaults to ``data/raw`` + ``data/aggregated``.
    live_dir : path | None
        The live ``v0.1/`` scenario directory (read-only). Used for exact +
        near-dup decontamination and benign-twin resolution.
    near_dup_threshold : float
        Jaccard-proxy similarity at/above which a draft is REJECTed (default 0.85).
    embedding_fn : callable | None
        Optional ``text -> vector`` embedder enabling an ADDITIONAL semantic
        cosine decontam check. None (default) keeps the gate fully offline and
        deterministic; only the MinHash proxy runs.
    """

    # Cap how many corpus texts we keep MinHash signatures for, to bound the
    # O(drafts * corpus) near-dup comparison. This is a performance guard, not a
    # correctness knob; exact decontam (step 3) is unaffected and remains total.
    _MAX_NEAR_DUP_CORPUS = 50_000

    def __init__(
        self,
        taxonomy: Optional["TaxonomyValidator"] = None,
        training_dirs: Optional[Sequence[Path]] = None,
        live_dir: Optional[Path] = None,
        near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        embedding_fn: Optional[EmbeddingFn] = None,
    ) -> None:
        self.near_dup_threshold = float(near_dup_threshold)
        self.embedding_fn = embedding_fn
        self._taxonomy_note: Optional[str] = None

        # Taxonomy validator (best-effort; degrades to WARNING if unavailable).
        if taxonomy is not None:
            self.taxonomy = taxonomy
        elif TaxonomyValidator is not None:
            try:
                self.taxonomy = TaxonomyValidator()
            except Exception as exc:  # pragma: no cover - env-dependent
                self.taxonomy = None
                self._taxonomy_note = f"taxonomy unavailable ({exc})"
        else:  # pragma: no cover
            self.taxonomy = None
            self._taxonomy_note = "taxonomy module not importable"

        self.training_dirs = (
            [Path(p) for p in training_dirs]
            if training_dirs is not None
            else list(_DEFAULT_TRAINING_DIRS)
        )
        self.live_dir = Path(live_dir) if live_dir is not None else _DEFAULT_LIVE_DIR

        # Lazily populated corpora.
        self._training_ids: Optional[set[str]] = None
        self._training_texts: Optional[list[str]] = None
        self._live_scenarios: Optional[list[Scenario]] = None

    # ── Corpus loaders ──────────────────────────────────────────────

    def _load_training_stable_ids(self) -> tuple[set[str], list[str]]:
        """Harvest (stable_ids, texts) from legacy training CSVs.

        Scans every ``*.csv`` under each training dir for a ``text`` column and
        builds :class:`Na0SSample` rows (reusing the canonical NFKC stable_id).
        Files without a usable ``text`` column, or rows that fail to parse, are
        skipped silently — a heterogeneous data lake must not crash the gate.

        Returns ``(set(), [])`` when no training corpus is present.
        """
        if self._training_ids is not None and self._training_texts is not None:
            return self._training_ids, self._training_texts

        ids: set[str] = set()
        texts: list[str] = []

        if Na0SSample is None:  # pragma: no cover - dataset schema always present
            self._training_ids, self._training_texts = ids, texts
            return ids, texts

        for tdir in self.training_dirs:
            if not tdir.is_dir():
                continue
            for csv_path in sorted(tdir.rglob("*.csv")):
                try:
                    with csv_path.open("r", encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        if reader.fieldnames is None:
                            continue
                        # Only mine files that actually carry sample text.
                        if "text" not in {
                            (f or "").strip().lower() for f in reader.fieldnames
                        }:
                            continue
                        # Normalize header casing once so from_legacy_csv_row finds "text".
                        lower_map = {
                            (f or ""): (f or "").strip().lower()
                            for f in reader.fieldnames
                        }
                        for row in reader:
                            norm_row = {
                                lower_map.get(k, k): v for k, v in row.items()
                            }
                            text = (norm_row.get("text") or "").strip()
                            if not text:
                                continue
                            try:
                                sample = Na0SSample.from_legacy_csv_row(norm_row)
                            except Exception:
                                # Label missing/unknown -> hash the text directly so
                                # decontam still has coverage.
                                sample = Na0SSample(
                                    text=text, label=_safe_default_label()
                                )
                            if sample.stable_id:
                                ids.add(sample.stable_id)
                                if len(texts) < self._MAX_NEAR_DUP_CORPUS:
                                    texts.append(sample.text)
                except (OSError, csv.Error, UnicodeDecodeError):
                    # Unreadable/binary file masquerading as CSV — skip it.
                    continue

        self._training_ids, self._training_texts = ids, texts
        return ids, texts

    def _load_live_scenarios(self) -> list[Scenario]:
        """Load the live ``v0.1/`` scenarios (read-only). Empty if dir absent."""
        if self._live_scenarios is not None:
            return self._live_scenarios
        if not self.live_dir.is_dir():
            self._live_scenarios = []
            return self._live_scenarios
        try:
            self._live_scenarios = load_scenarios_dir(self.live_dir)
        except Exception:
            # A malformed live set must not crash admission of drafts; treat as
            # empty and surface a note at report level.
            self._live_scenarios = []
        return self._live_scenarios

    # ── Per-check methods ───────────────────────────────────────────

    def _check_taxonomy(self, scenario: Scenario, result: AdmissionResult) -> None:
        """BLOCK if attack_category is unknown. Benign scenarios are exempt."""
        # Benign siblings use BEN (not in the attack taxonomy) — exempt them.
        if scenario.expected_verdict == "allowed":
            result.checks_passed.append("taxonomy")
            return
        if self.taxonomy is None:
            # Could not load taxonomy — do not silently pass; warn instead.
            result.warnings.append(
                "taxonomy: validator unavailable; attack_category "
                f"{scenario.attack_category!r} NOT verified "
                f"({self._taxonomy_note or 'unknown reason'})"
            )
            result.checks_passed.append("taxonomy")
            return
        if self.taxonomy.validate_code(scenario.attack_category):
            result.checks_passed.append("taxonomy")
        else:
            result.checks_failed.append("taxonomy")
            result.reasons.append(
                f"taxonomy: attack_category {scenario.attack_category!r} is not a "
                "known taxonomy code (data/taxonomy.yaml)"
            )

    def _check_exact_decontam(
        self,
        scenario: Scenario,
        result: AdmissionResult,
        training_ids: set[str],
        live_ids: set[str],
    ) -> None:
        """BLOCK if the draft stable_id collides with training OR live."""
        sid = scenario.stable_id or ""
        if sid in training_ids:
            result.checks_failed.append("exact_decontam")
            result.reasons.append(
                f"exact_decontam: stable_id {sid[:12]}... collides with a TRAINING "
                "sample (train/eval leak)"
            )
            return
        if sid in live_ids:
            result.checks_failed.append("exact_decontam")
            result.reasons.append(
                f"exact_decontam: stable_id {sid[:12]}... already present in the live "
                "v0.1 set (duplicate scenario)"
            )
            return
        result.checks_passed.append("exact_decontam")

    def _check_near_dup_decontam(
        self,
        scenario: Scenario,
        result: AdmissionResult,
        corpus_sigs: list[list[int]],
        corpus_emb: Optional[list[Sequence[float]]],
    ) -> None:
        """BLOCK on lexical near-duplicate (MinHash proxy) or semantic cosine.

        The MinHash leg is a *Jaccard/MinHash PROXY* for decontamination, not a
        true embedding cosine. If ``embedding_fn`` was supplied, an additional
        cosine leg runs and either leg crossing the threshold blocks.
        """
        text = _scenario_text(scenario)
        draft_sig = minhash_signature(text)

        max_jac = 0.0
        for sig in corpus_sigs:
            jac = jaccard_from_minhash(draft_sig, sig)
            if jac > max_jac:
                max_jac = jac
                if max_jac >= 1.0:
                    break

        max_cos: Optional[float] = None
        if self.embedding_fn is not None and corpus_emb is not None:
            try:
                draft_vec = self.embedding_fn(text)
                max_cos = 0.0
                for vec in corpus_emb:
                    cos = _cosine(draft_vec, vec)
                    if cos > max_cos:
                        max_cos = cos
            except Exception as exc:  # pragma: no cover - depends on user fn
                result.warnings.append(
                    f"near_dup_decontam: embedding_fn raised ({exc}); semantic "
                    "cosine leg skipped, MinHash proxy still applied"
                )
                max_cos = None

        blocked = max_jac >= self.near_dup_threshold or (
            max_cos is not None and max_cos >= self.near_dup_threshold
        )
        if blocked:
            result.checks_failed.append("near_dup_decontam")
            detail = (
                f"MinHash/Jaccard proxy={max_jac:.3f}"
                + (f", embedding cosine={max_cos:.3f}" if max_cos is not None else "")
            )
            result.reasons.append(
                f"near_dup_decontam: near-duplicate of training/live corpus "
                f"({detail}) >= threshold {self.near_dup_threshold:.2f}"
            )
        else:
            result.checks_passed.append("near_dup_decontam")

    def _check_benign_twin(
        self,
        scenario: Scenario,
        result: AdmissionResult,
        twin_index: dict[str, Scenario],
    ) -> None:
        """BLOCK if paired_benign_id is set but the twin is missing/non-allowed."""
        twin_id = scenario.paired_benign_id
        if not twin_id:
            # No paired twin declared — nothing to validate.
            result.checks_passed.append("benign_twin")
            return
        twin = twin_index.get(twin_id)
        if twin is None:
            result.checks_failed.append("benign_twin")
            result.reasons.append(
                f"benign_twin: paired_benign_id {twin_id[:12]}... has no matching "
                "scenario in the draft or live set"
            )
            return
        if twin.expected_verdict != "allowed":
            result.checks_failed.append("benign_twin")
            result.reasons.append(
                f"benign_twin: paired twin {twin_id[:12]}... exists but its "
                f"expected_verdict is {twin.expected_verdict!r}, not 'allowed'"
            )
            return
        result.checks_passed.append("benign_twin")

    def _check_trust(self, scenario: Scenario, result: AdmissionResult) -> None:
        """SOFT trust signal from provenance/source. WARNING only — never blocks.

        Drafts are human-reviewed before promotion, so an untrusted provenance is
        information for the reviewer, not a hard gate. We resolve the scenario's
        ``source`` against ``data/trust_tiers.yaml`` when available and emit a
        warning for low-trust tiers; absence of the registry is itself a (quiet)
        note, not a failure.
        """
        tier = _resolve_trust_tier(scenario.source)
        if tier is None:
            return  # registry not resolvable — stay silent, this is advisory only
        # tier1 (Trusted) / tier2 (Community) are fine; tier3+ get a soft flag.
        if tier not in ("tier1", "tier2"):
            result.warnings.append(
                f"trust: source {scenario.source!r} resolves to {tier} "
                "(low-trust provenance) — verify origin before promoting"
            )

    # ── Orchestration ───────────────────────────────────────────────

    def admit(
        self,
        scenario: Scenario,
        *,
        training_ids: Optional[set[str]] = None,
        live_ids: Optional[set[str]] = None,
        corpus_sigs: Optional[list[list[int]]] = None,
        corpus_emb: Optional[list[Sequence[float]]] = None,
        twin_index: Optional[dict[str, Scenario]] = None,
    ) -> AdmissionResult:
        """Run all checks against one already-loaded scenario.

        Corpora may be passed in (so :meth:`run` builds them once and reuses
        across drafts) or omitted (computed on demand for standalone use).
        Schema is assumed to have passed already since the scenario object
        exists; :meth:`run` records the schema check explicitly for files that
        fail to load.
        """
        result = AdmissionResult(name=scenario.name, status="ADMIT")
        result.checks_passed.append("schema")

        if training_ids is None or corpus_sigs is None:
            t_ids, t_texts = self._load_training_stable_ids()
            training_ids = training_ids if training_ids is not None else t_ids
            if corpus_sigs is None:
                live = self._load_live_scenarios()
                corpus_texts = list(t_texts) + [_scenario_text(s) for s in live]
                corpus_sigs = [minhash_signature(t) for t in corpus_texts]
        if live_ids is None:
            live = self._load_live_scenarios()
            live_ids = {s.stable_id for s in live if s.stable_id}
        if twin_index is None:
            live = self._load_live_scenarios()
            twin_index = {s.stable_id: s for s in live if s.stable_id}

        self._check_taxonomy(scenario, result)
        self._check_exact_decontam(scenario, result, training_ids, live_ids)
        self._check_near_dup_decontam(scenario, result, corpus_sigs, corpus_emb)
        self._check_benign_twin(scenario, result, twin_index)
        self._check_trust(scenario, result)

        if result.checks_failed:
            result.status = "REJECT"
        return result

    def run(self, drafts_dir: Path) -> AdmissionReport:
        """Validate every draft scenario in ``drafts_dir``; return a report.

        Schema failures are recorded per-file as REJECT results (the gate cannot
        introspect a scenario it cannot load). All other checks run against the
        scenarios that did load.
        """
        report = AdmissionReport()
        drafts_dir = Path(drafts_dir)

        # ── Build shared corpora once ──
        training_ids, training_texts = self._load_training_stable_ids()
        live = self._load_live_scenarios()
        live_ids = {s.stable_id for s in live if s.stable_id}
        report.training_sample_count = len(training_ids)
        report.live_scenario_count = len(live)
        if report.training_sample_count == 0:
            report.notes.append(
                "decontam: 0 training samples found "
                f"(searched {', '.join(str(p) for p in self.training_dirs)})"
            )

        corpus_texts = list(training_texts) + [_scenario_text(s) for s in live]
        corpus_sigs = [minhash_signature(t) for t in corpus_texts]
        report.decontam_methods.append(
            f"minhash_jaccard_proxy(n=3,num_hashes={MINHASH_NUM_HASHES},"
            f"threshold={self.near_dup_threshold:.2f},corpus={len(corpus_texts)})"
        )

        corpus_emb: Optional[list[Sequence[float]]] = None
        if self.embedding_fn is not None:
            try:
                corpus_emb = [self.embedding_fn(t) for t in corpus_texts]
                report.decontam_methods.append(
                    f"embedding_cosine(threshold={self.near_dup_threshold:.2f})"
                )
            except Exception as exc:  # pragma: no cover - depends on user fn
                report.notes.append(
                    f"decontam: embedding_fn failed during corpus encode ({exc}); "
                    "semantic cosine leg disabled"
                )
                corpus_emb = None
        else:
            report.notes.append(
                "decontam: lexical MinHash/Jaccard PROXY only (no embedding_fn); "
                "TODO: supply embedding_fn for true semantic cosine"
            )

        # ── Load drafts (schema gate) ──
        try:
            drafts = load_scenarios_dir(drafts_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(str(exc)) from exc
        except Exception as exc:
            # One malformed file fails the whole directory load; record it as a
            # single schema REJECT so the gate exits non-zero rather than 0.
            report.results.append(
                AdmissionResult(
                    name=str(drafts_dir),
                    status="REJECT",
                    checks_failed=["schema"],
                    reasons=[f"schema: failed to load drafts dir — {exc}"],
                )
            )
            return report

        # Benign-twin resolution spans BOTH the draft set and the live set.
        twin_index: dict[str, Scenario] = {
            s.stable_id: s for s in live if s.stable_id
        }
        for s in drafts:
            if s.stable_id:
                twin_index.setdefault(s.stable_id, s)

        for scenario in drafts:
            report.results.append(
                self.admit(
                    scenario,
                    training_ids=training_ids,
                    live_ids=live_ids,
                    corpus_sigs=corpus_sigs,
                    corpus_emb=corpus_emb,
                    twin_index=twin_index,
                )
            )
        return report


# ── Module-level helpers ────────────────────────────────────────────


def _scenario_text(scenario: Scenario) -> str:
    """Canonical text of a scenario for lexical decontam (payload or turns)."""
    if scenario.payload:
        return scenario.payload
    return " ".join(turn.text for turn in scenario.turns)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two dense vectors; 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _safe_default_label():
    """Return a benign DataLabel for rows whose label can't be parsed."""
    from na0s.dataset.schema import DataLabel

    # Prefer an explicit benign/safe member; fall back to the first member.
    for candidate in ("BENIGN", "SAFE", "CLEAN", "LEGITIMATE"):
        if hasattr(DataLabel, candidate):
            return getattr(DataLabel, candidate)
    return list(DataLabel)[0]


_TRUST_CONFIG_CACHE: dict[str, Optional[dict]] = {}


def _resolve_trust_tier(source: Optional[str]) -> Optional[str]:
    """Resolve a scenario ``source`` to a trust tier via scripts/quarantine.

    Returns the tier key (e.g. ``"tier1"``) or None if the registry / resolver
    is unavailable. Best-effort + cached; this is a soft advisory signal only.
    """
    if not source:
        return None
    if "config" not in _TRUST_CONFIG_CACHE:
        cfg = None
        try:
            import importlib.util

            qpath = _REPO_ROOT / "scripts" / "quarantine.py"
            if qpath.is_file():
                spec = importlib.util.spec_from_file_location(
                    "_na0s_quarantine_for_admission", qpath
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    cfg = (mod, mod.load_trust_config())
        except Exception:
            cfg = None
        _TRUST_CONFIG_CACHE["config"] = cfg

    cfg = _TRUST_CONFIG_CACHE.get("config")
    if not cfg:
        return None
    mod, config = cfg  # type: ignore[misc]
    try:
        return mod.resolve_tier(source, config)
    except Exception:
        return None


__all__ = [
    "AdmissionResult",
    "AdmissionReport",
    "ScenarioAdmissionGate",
    "DEFAULT_NEAR_DUP_THRESHOLD",
]
