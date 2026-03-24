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

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

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
