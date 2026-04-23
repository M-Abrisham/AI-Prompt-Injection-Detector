"""Na0S data schema — canonical sample representation.

Provides :class:`Na0SSample`, :class:`DataLabel`, :class:`DataSplit` and
validation helpers so every pipeline stage speaks the same schema.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
The legacy CSV has three columns: ``text``, ``label``, ``augmentation_type``.
:meth:`Na0SSample.from_legacy_csv_row` normalises the many label spellings
found in the wild (``"1"``/``"0"``, ``"True"``/``"False"``,
``"injection"``/``"benign"``, ``"malicious"``/``"safe"``) into
:class:`DataLabel`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class DataLabel(str, Enum):
    """Binary label for a prompt sample."""

    INJECTION = "injection"
    BENIGN = "benign"


class DataSplit(str, Enum):
    """Dataset split assignment."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# Technique-ID pattern: letter(s) followed by digits, optionally dot-separated sub-IDs,
# with optional _suffix (e.g. _benign for counterexamples).
# e.g. D1, D1.3, C1MT.1, I2.1, D1.1_benign, A3.2.1
_TECHNIQUE_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(\.\d+)*(_[a-z]+)?$")

# Label aliases — maps normalised lower-case strings to DataLabel
_LABEL_MAP: dict[str, DataLabel] = {
    "injection": DataLabel.INJECTION,
    "malicious": DataLabel.INJECTION,
    "true": DataLabel.INJECTION,
    "1": DataLabel.INJECTION,
    "benign": DataLabel.BENIGN,
    "safe": DataLabel.BENIGN,
    "false": DataLabel.BENIGN,
    "0": DataLabel.BENIGN,
}

_VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


@dataclass
class Na0SSample:
    """Canonical data sample for the Na0S dataset.

    The first three fields (``text``, ``label``, ``augmentation_type``) are
    the legacy CSV core and MUST remain the first positional arguments for
    backward compatibility.
    """

    # ── Legacy core (CSV columns) ──────────────────────────────────
    text: str
    label: DataLabel
    augmentation_type: Optional[str] = None

    # ── Enrichment fields ──────────────────────────────────────────
    technique_id: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    language: str = "en"
    split: DataSplit = DataSplit.TRAIN
    difficulty: Optional[str] = None
    license: Optional[str] = None
    license_url: Optional[str] = None
    attribution: Optional[str] = None
    hf_dataset: Optional[str] = None
    quality_score: Optional[float] = None
    is_duplicate: bool = False
    near_dup_cluster: Optional[str] = None
    created_at: Optional[str] = field(default=None)

    # ── A3 interop + F14 decontamination fields (added 2026-04-23) ──
    # Cross-dataset compatibility with HarmBench / WildJailbreak /
    # TensorTrust / NotInject / promptfoo. All optional; existing
    # samples remain valid without them.
    tactics: list[str] = field(default_factory=list)
    """Multi-label technique tags (e.g. ['roleplay', 'base64-encode']).
    Maps to WildJailbreak.tactics + HarmBench.Tags."""

    semantic_category: Optional[str] = None
    """Harm-domain category (e.g. 'cybercrime_intrusion', 'misinformation').
    Maps to HarmBench.SemanticCategory."""

    functional_category: Optional[str] = None
    """Attack framing (e.g. 'standard', 'contextual', 'copyright',
    'extraction', 'hijacking'). Maps to HarmBench.FunctionalCategory."""

    compliance_tags: list[str] = field(default_factory=list)
    """OWASP / MITRE ATLAS / NIST AI RMF / EU AI Act crosswalk tags
    (e.g. ['owasp:llm:01', 'mitre:atlas:exfiltration']).
    promptfoo-style multi-framework mapping."""

    context_string: Optional[str] = None
    """System-prompt or RAG context the attack runs against.
    Maps to TensorTrust.pre_prompt / HarmBench.ContextString."""

    expected_completion: Optional[str] = None
    """Success oracle — what counts as a successful attack.
    Maps to HackaPrompt.expected_completion."""

    paired_benign_id: Optional[str] = None
    """Stable_id of the benign sibling for over-refusal eval.
    WildJailbreak pairs vanilla_harmful with vanilla_benign."""

    trigger_words: list[str] = field(default_factory=list)
    """Words that make this sample look adversarial despite being
    benign (for over-defense calibration). Maps to NotInject.word_list."""

    stable_id: Optional[str] = None
    """SHA-256(normalized text) — cross-dataset deduplication key.
    Auto-computed in __post_init__ if not provided. F14 decontamination
    gate uses this to block eval scenarios from entering training."""

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.stable_id is None and self.text:
            # SHA-256 of NFKC-normalized whitespace-collapsed text.
            # Same normalization used in near_duplicate + process_data
            # to ensure identical hashes across pipeline stages.
            import hashlib
            import unicodedata
            normalized = unicodedata.normalize("NFKC", self.text)
            normalized = " ".join(normalized.split())
            self.stable_id = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

    # ── Factory ────────────────────────────────────────────────────

    @classmethod
    def from_legacy_csv_row(cls, row: dict) -> Na0SSample:
        """Build a sample from a legacy CSV dict.

        Handles all known label spellings (case-insensitive) and the
        optional ``augmentation_type`` column.
        """
        text = row.get("text", "")
        raw_label = str(row.get("label", "")).strip().lower()
        label = _LABEL_MAP.get(raw_label)
        if label is None:
            raise ValueError(
                "Unrecognised label value: {!r}. "
                "Expected one of: {}".format(
                    row.get("label"), ", ".join(sorted(_LABEL_MAP))
                )
            )
        augmentation_type = row.get("augmentation_type") or None
        return cls(text=text, label=label, augmentation_type=augmentation_type)

    # ── Serialisation ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialisation."""
        return {
            "text": self.text,
            "label": self.label.value,
            "augmentation_type": self.augmentation_type,
            "technique_id": self.technique_id,
            "source": self.source,
            "source_id": self.source_id,
            "language": self.language,
            "split": self.split.value,
            "difficulty": self.difficulty,
            "license": self.license,
            "license_url": self.license_url,
            "attribution": self.attribution,
            "hf_dataset": self.hf_dataset,
            "quality_score": self.quality_score,
            "is_duplicate": self.is_duplicate,
            "near_dup_cluster": self.near_dup_cluster,
            "created_at": self.created_at,
            # A3 interop fields
            "tactics": list(self.tactics),
            "semantic_category": self.semantic_category,
            "functional_category": self.functional_category,
            "compliance_tags": list(self.compliance_tags),
            "context_string": self.context_string,
            "expected_completion": self.expected_completion,
            "paired_benign_id": self.paired_benign_id,
            "trigger_words": list(self.trigger_words),
            "stable_id": self.stable_id,
        }

    # ── External-format adapters (A3 interop) ──────────────────────

    def to_harmbench_row(self) -> dict:
        """Export this sample as a HarmBench-compatible row.

        HarmBench schema: ``Behavior, FunctionalCategory, SemanticCategory,
        Tags, ContextString, BehaviorID``.
        Reference: https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv
        """
        return {
            "Behavior": self.text,
            "FunctionalCategory": self.functional_category or "standard",
            "SemanticCategory": self.semantic_category or "",
            "Tags": ",".join(self.tactics),
            "ContextString": self.context_string or "",
            "BehaviorID": self.stable_id or "",
        }

    def to_wildjailbreak_row(self) -> dict:
        """Export this sample as a WildJailbreak-compatible row.

        WildJailbreak schema: ``vanilla, adversarial, tactics, completion,
        data_type`` where data_type ∈
        {vanilla_harmful, vanilla_benign, adversarial_harmful, adversarial_benign}.
        Reference: https://huggingface.co/datasets/allenai/wildjailbreak
        """
        is_malicious = self.label == DataLabel.INJECTION
        is_adversarial = bool(self.tactics)  # tactics present = adversarial transform applied
        if is_adversarial and is_malicious:
            data_type = "adversarial_harmful"
        elif is_adversarial and not is_malicious:
            data_type = "adversarial_benign"
        elif is_malicious:
            data_type = "vanilla_harmful"
        else:
            data_type = "vanilla_benign"
        return {
            "vanilla": self.text if not is_adversarial else "",
            "adversarial": self.text if is_adversarial else "",
            "tactics": list(self.tactics),
            "completion": self.expected_completion or "",
            "data_type": data_type,
        }


# ── Validators ─────────────────────────────────────────────────────


def validate_sample(s: Na0SSample) -> list[str]:
    """Return a list of validation error strings (empty = valid).

    Checks:
    - text not empty
    - text not longer than 50 000 characters
    - quality_score in [0.0, 1.0] when present
    - difficulty in {easy, medium, hard} when present
    - technique_id matches expected format when present
    """
    errors: list[str] = []

    if not s.text or not s.text.strip():
        errors.append("text is empty or whitespace-only")

    if len(s.text) > 50_000:
        errors.append(
            "text exceeds 50 000 characters ({})".format(len(s.text))
        )

    if s.quality_score is not None:
        if not (0.0 <= s.quality_score <= 1.0):
            errors.append(
                "quality_score must be in [0.0, 1.0], got {}".format(
                    s.quality_score
                )
            )

    if s.difficulty is not None:
        if s.difficulty not in _VALID_DIFFICULTIES:
            errors.append(
                "difficulty must be one of {}, got {!r}".format(
                    sorted(_VALID_DIFFICULTIES), s.difficulty
                )
            )

    if s.technique_id is not None:
        if not _TECHNIQUE_ID_RE.match(s.technique_id):
            errors.append(
                "technique_id {!r} does not match expected format "
                "(e.g. D1, D1.3, I2.1)".format(s.technique_id)
            )

    return errors


def validate_sample_strict(s: Na0SSample) -> list[str]:
    """Strict validation — base checks plus required enrichment fields.

    Additionally requires:
    - license is set
    - source is set
    """
    errors = validate_sample(s)

    if not s.license:
        errors.append("license is required (strict mode)")

    if not s.source:
        errors.append("source is required (strict mode)")

    return errors
