"""F14 Scenario schema — single-prompt + multi-turn scenarios for the promotion gate."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ScenarioType(str, Enum):
    """Whether a scenario is a single prompt or a multi-turn conversation.

    SINGLE_PROMPT: one text -> one verdict (maps to classic canary_eval.csv rows).
    MULTI_TURN: list of turns -> detection expected by turn N (maps to layer16
    conversation fixtures and the R6 "multi-turn jailbreak" coverage cell).
    """

    SINGLE_PROMPT = "single_prompt"
    MULTI_TURN = "multi_turn"


class EvaluatorType(str, Enum):
    """How to score a scenario's outcome.

    DETERMINISTIC: boolean/numeric check on the model's verdict (e.g.
    ``result.label == MALICIOUS`` or ``result.confidence >= 0.5``).
    LLM_JUDGE: an LLM (Claude, Mistral, local) scores the response against
    the expected behavior; useful for open-ended outputs where the verdict
    isn't a simple label. Reserved for F14-v0.3+ — not used in v0.1.
    """

    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"


@dataclass
class ScenarioTurn:
    """One turn within a multi-turn scenario.

    Attributes
    ----------
    text : str
        The user-side text for this turn.
    expected_label : str
        Ground-truth label for this turn in isolation (e.g. ``safe``,
        ``suspicious``, ``malicious``). Used for per-turn validation and
        to let the gate detect "missed an early warning sign" cases.
    risk_score : Optional[float]
        Optional ground-truth risk score in [0, 1]. Mirrors the layer16
        fixture convention. When absent, the gate derives pass/fail only
        from ``expected_label``.
    """

    text: str
    expected_label: str
    risk_score: Optional[float] = None


@dataclass
class ScenarioEvaluator:
    """How to score a scenario's outcome.

    Attributes
    ----------
    type : EvaluatorType
        DETERMINISTIC or LLM_JUDGE (only DETERMINISTIC used in v0.1).
    check : str
        For DETERMINISTIC: a short expression describing what to assert
        (e.g. ``"label == MALICIOUS"``, ``"confidence >= 0.5"``,
        ``"any(turn.alert for turn in result.turns)"``). v0.1 uses this
        as free-text documentation; v0.2 will parse and execute it.
    threshold : Optional[float]
        For LLM_JUDGE: the numeric threshold the judge score must exceed.
    """

    type: EvaluatorType = EvaluatorType.DETERMINISTIC
    check: str = "label == MALICIOUS"
    threshold: Optional[float] = None


@dataclass
class Scenario:
    """One scenario in the F14 evaluation library.

    Single-prompt: set ``type=SINGLE_PROMPT`` and ``payload`` (a string).
    Multi-turn: set ``type=MULTI_TURN`` and ``turns`` (list of ScenarioTurn).
    Exactly one of ``payload`` / ``turns`` must be non-empty — enforced by
    :meth:`__post_init__`.

    The A3-compatible fields (``stable_id``, ``paired_benign_id``,
    ``compliance_tags``) mirror :class:`na0s.dataset.schema.Na0SSample` so
    scenarios can round-trip through the dataset pipeline and be
    decontamination-checked against training data.

    Attributes
    ----------
    name : str
        Short unique identifier (e.g. ``"D1_ignore_previous_direct"``).
    type : ScenarioType
        Single prompt or multi-turn.
    expected_verdict : str
        ``"blocked"`` or ``"allowed"`` — what the defender should do.
    severity : str
        ``"critical"`` | ``"high"`` | ``"medium"`` | ``"low"`` — governs
        whether a regression on this scenario is a BLOCK or WARNING gate.
    attack_category : str
        Taxonomy category from ``data/taxonomy.yaml`` (e.g. ``"D1"``,
        ``"E1.3"``). Used for per-category TPR/FPR scoring.
    description : str
        Human-readable summary.
    payload : Optional[str]
        Single-prompt text. Required iff ``type == SINGLE_PROMPT``.
    turns : list[ScenarioTurn]
        Multi-turn conversation. Required iff ``type == MULTI_TURN``.
    customer_archetype : Optional[str]
        Target customer context (``"chatbot"``, ``"rag"``,
        ``"coding_agent"``, ``"internal_search"``). Used for multi-bucket
        scoring in F14-v0.6+; unused in v0.1.
    evaluator : ScenarioEvaluator
        How to score this scenario's outcome.
    source : str
        Where the scenario came from (``"manual"``, ``"shade_arena"``,
        ``"harmbench"``, ``"layer16_fixtures"``, ``"llm_generated"``,
        ``"matrix_composed"``, ``"harvest_pipeline"``, ``"synthesized"``).
        ``"synthesized"`` marks scenarios that are deliberately paraphrased /
        re-authored from a public incident report (never copied verbatim) to
        cover a taxonomy gap the report describes; provenance (origin URL +
        retrieval date) is folded into ``description`` exactly as for
        ``"harvest_pipeline"``.
    tags : list[str]
        Multi-label technique tags (e.g. ``["roleplay", "base64-encode"]``).
    difficulty : Optional[int]
        L12 probe difficulty score (100-400). Used by F14-v0.2 bandit.
    compliance_tags : list[str]
        OWASP / MITRE ATLAS / NIST / EU AI Act crosswalk (mirrors
        :attr:`Na0SSample.compliance_tags`).
    stable_id : Optional[str]
        SHA-256 of the normalized canonical content. Auto-computed if
        not provided. Used by the F14 decontamination gate to block
        scenarios from entering training data.
    paired_benign_id : Optional[str]
        stable_id of the benign sibling for over-refusal testing
        (mirrors :attr:`Na0SSample.paired_benign_id`).
    """

    name: str
    type: ScenarioType
    expected_verdict: str  # "blocked" | "allowed"
    severity: str          # "critical" | "high" | "medium" | "low"
    attack_category: str
    description: str = ""
    payload: Optional[str] = None
    turns: list[ScenarioTurn] = field(default_factory=list)
    customer_archetype: Optional[str] = None
    evaluator: ScenarioEvaluator = field(default_factory=ScenarioEvaluator)
    source: str = "manual"
    tags: list[str] = field(default_factory=list)
    difficulty: Optional[int] = None
    compliance_tags: list[str] = field(default_factory=list)
    stable_id: Optional[str] = None
    paired_benign_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Enforce exactly-one-of payload/turns based on type
        if self.type == ScenarioType.SINGLE_PROMPT:
            if not self.payload:
                raise ValueError(
                    f"Scenario {self.name!r} is SINGLE_PROMPT but payload is empty"
                )
            if self.turns:
                raise ValueError(
                    f"Scenario {self.name!r} is SINGLE_PROMPT; turns must be empty "
                    f"(got {len(self.turns)})"
                )
        elif self.type == ScenarioType.MULTI_TURN:
            if not self.turns:
                raise ValueError(
                    f"Scenario {self.name!r} is MULTI_TURN but turns list is empty"
                )
            if self.payload:
                raise ValueError(
                    f"Scenario {self.name!r} is MULTI_TURN; payload must be empty "
                    f"(got {len(self.payload)} chars)"
                )

        # Validate expected_verdict + severity vocabularies
        if self.expected_verdict not in ("blocked", "allowed"):
            raise ValueError(
                f"Scenario {self.name!r}: expected_verdict must be "
                f"'blocked' or 'allowed', got {self.expected_verdict!r}"
            )
        if self.severity not in ("critical", "high", "medium", "low"):
            raise ValueError(
                f"Scenario {self.name!r}: severity must be one of "
                f"critical/high/medium/low, got {self.severity!r}"
            )

        # Auto-compute stable_id from canonical content if not provided
        if self.stable_id is None:
            self.stable_id = self._compute_stable_id()

    def _canonical_content(self) -> str:
        """Return the canonical text content for hashing + dedup.

        For single-prompt: the payload itself.
        For multi-turn: the concatenation of turn texts separated by
        a null char (unlikely to appear naturally, gives stable boundary).
        """
        if self.type == ScenarioType.SINGLE_PROMPT:
            return self.payload or ""
        return "\x00".join(turn.text for turn in self.turns)

    def _compute_stable_id(self) -> str:
        """SHA-256 of NFKC-normalized whitespace-collapsed canonical content.

        Matches the normalization used by Na0SSample.stable_id so that a
        scenario and a training sample with the same text hash to the
        same stable_id — enabling decontamination checks across the two
        storage layers.
        """
        text = self._canonical_content()
        normalized = unicodedata.normalize("NFKC", text)
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for YAML round-trip."""
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type.value,
            "expected_verdict": self.expected_verdict,
            "severity": self.severity,
            "attack_category": self.attack_category,
            "description": self.description,
            "customer_archetype": self.customer_archetype,
            "source": self.source,
            "tags": list(self.tags),
            "difficulty": self.difficulty,
            "compliance_tags": list(self.compliance_tags),
            "stable_id": self.stable_id,
            "paired_benign_id": self.paired_benign_id,
            "evaluator": {
                "type": self.evaluator.type.value,
                "check": self.evaluator.check,
                "threshold": self.evaluator.threshold,
            },
        }
        if self.type == ScenarioType.SINGLE_PROMPT:
            out["payload"] = self.payload
        else:
            out["turns"] = [
                {
                    "text": turn.text,
                    "expected_label": turn.expected_label,
                    "risk_score": turn.risk_score,
                }
                for turn in self.turns
            ]
        return out
