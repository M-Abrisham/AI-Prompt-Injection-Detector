"""Goal-Decomposition / Kill-Chain-Coverage detector (T1.3, IM3.x — GTG-1002).

THE PROBLEM
-----------
The GTG-1002 pattern is a *benign-decomposition* attack: a trusted persona
("authorized security firm, routine testing") splits a full intrusion into
individually-innocuous tool-calling steps — enumerate hosts, fingerprint
services, map CVEs — each of which scores ~0 in isolation, then pivots on the
final turn to an EXPLOIT / CRED_HARVEST / EXFIL action. Because every early turn
is benign on its own, the score-driven session aggregate never crosses the block
threshold, and the per-turn detectors never fire. This detector closes that gap
by reasoning over the *raw text per turn* (``state.turns[].text``), independent
of each turn's ``risk_score``.

HOW IT WORKS
------------
1. Per-turn phase tagging. Each turn's text is tagged with the kill-chain
   phase(s) it touches (PERSONA, RECON_INVENTORY, SERVICE_FINGERPRINT, VULN_MAP,
   EXPLOIT, CRED_HARVEST, EXFIL). Tagging is **regex first** against the anchor
   lexicon in ``data/killchain_phases.yaml``; when an all-MiniLM-L6-v2 encoder
   is available (offline, via :func:`load_pinned_sentence_transformer`) it adds
   a centroid-cosine fallback for turns that regex missed. With
   sentence-transformers absent the detector runs regex-only — it never silently
   disables (mirrors :mod:`embedding_drift` graceful degradation, but unlike
   drift the regex layer is high-precision so we keep alerting).

2. Session coverage. We track the SET of distinct phases seen across the whole
   session (the distinct-count pattern from ``detectors/recon.py``'s
   ``categories_seen``).

3. ORDER-AWARE trigger (NOT score-additive). We alert only when BOTH hold:
     (a) the session covers >= N distinct phases (N = GOAL_DECOMP_MIN_PHASES),
         AND
     (b) a TERMINAL phase (EXPLOIT / CRED_HARVEST / EXFIL) appears in a turn
         that comes AFTER an earlier RECON/FINGERPRINT/VULN_MAP turn — a
         kill-chain PIVOT.
   The pivot is the real gate; coverage alone never alerts. A high/critical
   :class:`Alert` flows through the existing
   ``conversation_monitor._compute_recommendation`` -> "block" -> predict.py G02
   fold-back, so no predict.py change is needed.

4. BENIGN SUPPRESSOR (FP-safety, load-bearing). Before alerting, if the session
   carries explicit rules-of-engagement / scope markers (signed contract, ROE,
   "no exploitation", "proof-of-vulnerability only", "aggregates only", "no
   external endpoints", read-only) AND no terminal-phase pivot fired, we
   SUPPRESS. The six GTG-1002 benign twins carry these markers and stop at
   VULN_MAP / aggregate — they MUST stay allowed. (If a pivot DID fire, ROE text
   does not rescue it: an in-scope engagement does not write working exploits or
   exfiltrate, so a real terminal action overrides the ROE claim.)

5. PERSONA multiplier (conservative). An unverifiable authority claim (reusing
   :data:`context_poisoning._AUTHORITY_PATTERNS`) co-occurring with a pivot may
   lower N by 1 (down to GOAL_DECOMP_MIN_PHASES_FLOOR). Persona ALONE must NEVER
   trigger — the benign twins share the identical persona, so persona is only
   ever an amplifier on top of an already-firing pivot, never a standalone
   signal.

NO NETWORK / NO LLM: regex + an offline-cached MiniLM encoder only. The L7 LLM
judge is not a dependency.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..config import (
    ENABLE_GOAL_DECOMPOSITION,
    GOAL_DECOMP_MIN_PHASES,
    GOAL_DECOMP_MIN_PHASES_FLOOR,
    GOAL_DECOMP_MIN_TURNS,
)
from ..models import Alert, ConversationState
from .base_detector import MultiTurnDetector

# Reuse the authority lexicon already maintained for context poisoning so the
# persona signal stays consistent across detectors and has a single source of
# truth.
from .context_poisoning import _AUTHORITY_PATTERNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sentence-transformers (mirror embedding_drift.py)
# ---------------------------------------------------------------------------

_HAS_EMBEDDINGS = False
try:
    from sentence_transformers import SentenceTransformer

    _HAS_EMBEDDINGS = True
except ImportError:  # pragma: no cover - exercised only when dep absent
    SentenceTransformer = None  # type: ignore[assignment]

from na0s.ml._st_loader import load_pinned_sentence_transformer

_MODEL_NAME = "all-MiniLM-L6-v2"

# Centroid-cosine acceptance threshold for the embedding fallback tagger. A turn
# is assigned a phase by the encoder only when its cosine similarity to that
# phase's anchor centroid is >= this value. Set conservatively HIGH (0.45) so
# the fuzzy fallback only fires on clear semantic matches; the high-precision
# regex layer carries the primary load and the embedding layer is corroboration
# for paraphrases the regex missed. (all-MiniLM-L6-v2 cosine for clearly related
# short phrases is typically 0.45-0.75; unrelated pairs sit well below 0.3.)
_EMBED_SIM_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Phase lexicon — loaded once from data/killchain_phases.yaml
# ---------------------------------------------------------------------------

# src/na0s/conversation/detectors/goal_decomposition.py ->
#   detectors[0] / conversation[1] / na0s[2] / src[3] / repo-root[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_KILLCHAIN_PATH = _REPO_ROOT / "data" / "killchain_phases.yaml"


def _load_killchain() -> dict:
    """Load and parse the kill-chain anchor lexicon.

    Returns a dict with keys ``phase_order``, ``terminal_phases``,
    ``recon_phases`` and ``phases`` (phase -> list[str] anchors). On any failure
    (missing file, missing PyYAML) returns an empty-ish structure so the
    detector degrades to a no-op rather than crashing the monitor.
    """
    try:
        import yaml  # local import: PyYAML is an optional/soft dep
    except ImportError:  # pragma: no cover
        logger.warning("PyYAML not installed; goal_decomposition disabled")
        return {"phase_order": [], "terminal_phases": [], "recon_phases": [], "phases": {}}

    try:
        raw = yaml.safe_load(_KILLCHAIN_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):  # pragma: no cover
        logger.warning("could not read %s; goal_decomposition disabled", _KILLCHAIN_PATH)
        return {"phase_order": [], "terminal_phases": [], "recon_phases": [], "phases": {}}

    return {
        "phase_order": list(raw.get("phase_order") or []),
        "terminal_phases": set(raw.get("terminal_phases") or []),
        "recon_phases": set(raw.get("recon_phases") or []),
        "phases": dict(raw.get("phases") or {}),
    }


_KILLCHAIN = _load_killchain()


def _compile_anchor_regexes(phases: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    """Compile each anchor phrase into a word-boundary-aware regex.

    We escape the literal phrase and wrap it so it matches as a phrase rather
    than a bare substring — this is the FP-safety boundary that stops, e.g., the
    benign "no external endpoints" from matching the EXFIL anchor
    "external endpoint" mid-token. A ``\b`` boundary is added only on a side
    whose adjacent anchor character is a word char (alnum/underscore); anchors
    that begin with punctuation such as "/etc/shadow" get no leading ``\b``
    (``\b`` before "/" requires a preceding word char and would never match
    "pull /etc/shadow").
    """
    compiled: Dict[str, List[re.Pattern]] = {}
    for phase, anchors in phases.items():
        pats: List[re.Pattern] = []
        for anchor in anchors:
            esc = re.escape(anchor)
            # A \b is meaningful only adjacent to a word char (\w == [A-Za-z0-9_]).
            lead = r"\b" if (anchor[:1].isalnum() or anchor[:1] == "_") else ""
            trail = r"\b" if (anchor[-1:].isalnum() or anchor[-1:] == "_") else ""
            pats.append(re.compile(lead + esc + trail, re.IGNORECASE))
        compiled[phase] = pats
    return compiled


_ANCHOR_REGEXES = _compile_anchor_regexes(_KILLCHAIN["phases"])

# ---------------------------------------------------------------------------
# Benign ROE / scope suppressor lexicon (FP-safety, load-bearing)
# ---------------------------------------------------------------------------

# These phrases are the explicit "this is an authorized, scope-limited
# engagement that does NOT take terminal actions" markers carried by the six
# GTG-1002 benign twins. Word-boundary regex (not substrings) keeps them from
# matching attack text. The suppressor only applies when NO terminal pivot
# fired — a real terminal action overrides any ROE claim.
_ROE_MARKERS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsigned\s+(?:penetration[-\s]?test\s+)?contract\b",
        r"\bsigned\s+(?:pen[-\s]?test|engagement|assessment|audit)\b",
        r"\brules[-\s]of[-\s]engagement\b",
        r"\bROE\b",
        r"\bin\s+scope\b",
        r"\bwithin\s+(?:that\s+|the\s+)?(?:approved\s+)?scope\b",
        r"\bno\s+exploitation\b",
        r"\bproof[-\s]of[-\s]vulnerability\s+only\b",
        r"\baggregates?\s+only\b",
        r"\baggregate\s+counts?\s+only\b",
        r"\bno\s+external\s+endpoints?\b",
        r"\bno\s+external\s+destinations?\b",
        r"\bread[-\s]only\b",
        r"\bdo\s+not\s+exploit\b",
        r"\bdon[''’]t\s+exploit\b",
        r"\bfindings?[-\s]only\b",
        r"\bno\s+embedded\s+directives?\b",
        r"\bno\s+brute[-\s]?forcing\s+real\b",
        r"\bout\s+of\s+scope\b",
    ]
]


def _has_roe_markers(texts: List[str]) -> bool:
    """True if any turn carries an explicit ROE / scope-limitation marker."""
    for text in texts:
        if not text:
            continue
        for pat in _ROE_MARKERS:
            if pat.search(text):
                return True
    return False


def _has_authority_claim(texts: List[str]) -> bool:
    """True if any turn carries an unverifiable authority/persona claim."""
    for text in texts:
        if not text:
            continue
        for pat in _AUTHORITY_PATTERNS:
            if pat.search(text):
                return True
    return False


# ---------------------------------------------------------------------------
# Embedding helpers (graceful-degrade to regex-only)
# ---------------------------------------------------------------------------


def _cosine(a, b) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    denom = (na ** 0.5) * (nb ** 0.5)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _centroid(vectors: list) -> list:
    if not vectors:
        return []
    width = len(vectors[0])
    acc = [0.0] * width
    for vec in vectors:
        for i in range(width):
            acc[i] += vec[i]
    n = float(len(vectors))
    return [v / n for v in acc]


class GoalDecompositionDetector(MultiTurnDetector):
    """Detects kill-chain decomposition across a multi-turn session.

    Reads ``state.turns[].text`` directly (independent of per-turn risk_score),
    tags each turn with kill-chain phase(s), and alerts when the session covers
    >= N distinct phases AND a terminal phase pivots after an earlier recon
    phase. Benign authorized-pentest siblings are suppressed via explicit ROE /
    scope markers.

    Taxonomy IDs: T1.3 (agent goal decomposition / tool abuse),
    IM3.4 (inter-model task-splitting across a delegation chain).
    """

    # Class-level model cache — loaded once, shared across instances
    # (mirror embedding_drift.py).
    _model = None
    _model_lock = threading.Lock()
    # Phase-centroid cache: phase -> centroid vector. Built once after the model
    # loads so we do not re-encode the static anchor lexicon every analyze().
    _phase_centroids: Optional[Dict[str, list]] = None
    _centroid_lock = threading.Lock()

    # ----- MultiTurnDetector interface ------------------------------------

    @property
    def detector_name(self) -> str:
        return "goal_decomposition"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["T1.3", "IM3.4"]

    def reset(self) -> None:
        pass  # stateless — all data comes from ConversationState

    # ----- Embedding plumbing --------------------------------------------

    @classmethod
    def _load_model(cls) -> None:
        if cls._model is None and _HAS_EMBEDDINGS:
            with cls._model_lock:
                if cls._model is None:  # double-check under lock
                    cls._model = load_pinned_sentence_transformer(
                        SentenceTransformer, _MODEL_NAME,
                    )

    def _encode(self, texts: List[str]) -> Optional[list]:
        """Encode *texts*; None when no real encoder is available.

        Unlike the regex path, the embedding path is OPTIONAL corroboration: if
        sentence-transformers is absent we return None and the detector runs
        regex-only. A test may inject an ``_encode`` instance override to feed
        synthetic vectors without the real model.
        """
        if _HAS_EMBEDDINGS:
            self._load_model()
            if self._model is not None:
                raw = self._model.encode(texts)
                return [r.tolist() if hasattr(r, "tolist") else list(r) for r in raw]
        return None

    def _ensure_phase_centroids(self) -> Optional[Dict[str, list]]:
        """Build (once) a phase -> anchor-centroid map using the encoder."""
        if self._phase_centroids is not None:
            return self._phase_centroids
        phases = _KILLCHAIN["phases"]
        if not phases:
            return None
        with self._centroid_lock:
            if self.__class__._phase_centroids is not None:
                return self.__class__._phase_centroids
            centroids: Dict[str, list] = {}
            for phase, anchors in phases.items():
                if not anchors:
                    continue
                vecs = self._encode(list(anchors))
                if not vecs:
                    return None  # encoder unavailable — abandon embedding path
                centroids[phase] = _centroid(vecs)
            self.__class__._phase_centroids = centroids
            return centroids

    # ----- Phase tagging --------------------------------------------------

    def _tag_turn(self, text: str, turn_vec: Optional[list],
                  centroids: Optional[Dict[str, list]]) -> set:
        """Return the set of kill-chain phases touched by *text*.

        Regex first (high precision). If an encoder + per-turn vector are
        available, add any phase whose anchor centroid is within
        ``_EMBED_SIM_THRESHOLD`` cosine of the turn — catching paraphrases the
        literal anchors missed.
        """
        tagged: set = set()
        if not text:
            return tagged

        # 1. Regex layer (always on).
        for phase, pats in _ANCHOR_REGEXES.items():
            for pat in pats:
                if pat.search(text):
                    tagged.add(phase)
                    break

        # 2. Embedding fallback (optional corroboration for paraphrases).
        if turn_vec is not None and centroids:
            for phase, centroid in centroids.items():
                if phase in tagged:
                    continue
                if _cosine(turn_vec, centroid) >= _EMBED_SIM_THRESHOLD:
                    tagged.add(phase)

        return tagged

    # ----- Main detection -------------------------------------------------

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not ENABLE_GOAL_DECOMPOSITION:
            return []
        if not _KILLCHAIN["phases"]:
            return []  # lexicon failed to load — degrade to no-op
        if state is None or state.is_empty:
            return []
        if len(state.turns) < GOAL_DECOMP_MIN_TURNS:
            return []

        texts = [t.text or "" for t in state.turns]

        terminal_phases: set = _KILLCHAIN["terminal_phases"]
        recon_phases: set = _KILLCHAIN["recon_phases"]

        # Optional per-turn embedding vectors (None => regex-only path).
        turn_vecs: Optional[list] = self._encode(texts)
        centroids = self._ensure_phase_centroids() if turn_vecs is not None else None

        # Tag every turn; track first-seen turn index per phase for ordering.
        per_turn_phases: List[set] = []
        phases_seen: set = set()
        first_seen: Dict[str, int] = {}
        for idx, text in enumerate(texts):
            tv = turn_vecs[idx] if turn_vecs is not None else None
            tags = self._tag_turn(text, tv, centroids)
            per_turn_phases.append(tags)
            for ph in tags:
                phases_seen.add(ph)
                first_seen.setdefault(ph, idx)

        # --- Kill-chain PIVOT: a terminal phase AFTER an earlier recon phase ---
        earliest_recon = min(
            (first_seen[p] for p in recon_phases if p in first_seen),
            default=None,
        )
        pivot_phase: Optional[str] = None
        pivot_turn: Optional[int] = None
        if earliest_recon is not None:
            for idx in range(earliest_recon + 1, len(per_turn_phases)):
                term_here = per_turn_phases[idx] & terminal_phases
                if term_here:
                    pivot_phase = sorted(term_here)[0]
                    pivot_turn = idx
                    break
        pivot_fired = pivot_phase is not None

        # --- Effective N (persona multiplier; conservative) ---
        # Persona ALONE never triggers: the multiplier only ever lowers N when a
        # pivot has ALSO fired, and never below the floor. The benign twins share
        # the persona AND the pivot gate keeps them safe (they never pivot), so
        # this can never make a benign twin alert on its own.
        #
        # The "persona" signal is the union of (a) the killchain PERSONA phase
        # being tagged in any turn — the canonical trust/authority/rapport
        # marker, documented in killchain_phases.yaml as non-triggering alone —
        # and (b) the narrower context_poisoning authority lexicon. Keying off
        # the PERSONA phase (not just _AUTHORITY_PATTERNS) is load-bearing for
        # Build-3 calibration: the GTG-1002 trusted persona ("contracted
        # security consultancy", "we're authorized") is captured by the PERSONA
        # anchors but NOT by the stricter _AUTHORITY_PATTERNS, so without this the
        # two 3-phase attacks (unauthorized_tool_invocation, tool_param_exfil)
        # never reach the default N=4 and stay unblocked. With the multiplier
        # they clear the floor N=3. Measured benign-FPR on the 6 GTG-1002 benign
        # siblings remains 0/6 because none carry a terminal-after-recon pivot.
        persona_present = "PERSONA" in phases_seen or _has_authority_claim(texts)
        effective_n = GOAL_DECOMP_MIN_PHASES
        persona_applied = False
        if pivot_fired and persona_present:
            effective_n = max(GOAL_DECOMP_MIN_PHASES_FLOOR, GOAL_DECOMP_MIN_PHASES - 1)
            persona_applied = effective_n < GOAL_DECOMP_MIN_PHASES

        # --- BENIGN SUPPRESSOR (load-bearing) ---
        # If the session declares ROE/scope AND no terminal pivot fired, this is
        # the authorized-pentest twin pattern (recon -> ... -> stop at VULN_MAP /
        # aggregate). Suppress. A pivot overrides ROE (an in-scope engagement
        # does not exploit / harvest / exfiltrate).
        if not pivot_fired and _has_roe_markers(texts):
            return []

        # --- TRIGGER: coverage AND order-aware pivot ---
        distinct = len(phases_seen)
        if not pivot_fired or distinct < effective_n:
            return []

        # Severity: a terminal pivot in a multi-phase decomposition is high; if
        # TWO+ terminal phases appear (e.g. exploit + exfil) it is critical.
        terminal_seen = phases_seen & terminal_phases
        severity = "critical" if len(terminal_seen) >= 2 else "high"

        # Confidence scales with how far coverage exceeds the floor, capped.
        # Floor at 0.75 (already a high/blocking alert); +0.05 per extra phase.
        confidence = round(min(0.99, 0.75 + 0.05 * (distinct - effective_n)), 4)

        evidence = [
            f"distinct_phases={distinct}",
            f"phases_seen={sorted(phases_seen)}",
            f"effective_n={effective_n}",
            f"pivot={pivot_phase}@turn{pivot_turn}",
            f"earliest_recon=turn{earliest_recon}",
            f"terminal_phases_seen={sorted(terminal_seen)}",
            f"persona_multiplier_applied={persona_applied}",
            f"embedding_tagging={'on' if turn_vecs is not None else 'regex_only'}",
        ]

        return [
            Alert(
                alert_type="goal_decomposition",
                severity=severity,
                confidence=confidence,
                description=(
                    f"Kill-chain decomposition detected: {distinct} distinct "
                    f"phases across the session with a terminal '{pivot_phase}' "
                    f"pivot at turn {pivot_turn} following recon at turn "
                    f"{earliest_recon} (GTG-1002 benign-decomposition pattern)"
                ),
                turn_range=(earliest_recon, pivot_turn),
                evidence=evidence,
            )
        ]
