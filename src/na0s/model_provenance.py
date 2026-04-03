"""Model provenance tracking for supply-chain integrity (Layer 11).

Records and verifies metadata about trained models via JSON sidecar files.
Gated behind NA0S_MODEL_PROVENANCE=1 environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _is_enabled() -> bool:
    return os.environ.get("NA0S_MODEL_PROVENANCE", "0") == "1"


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelProvenance:
    model_path: str = ""
    sha256: str = ""
    training_date: str = ""
    training_script: str = ""
    dataset_version: str = ""
    feature_count: int = 0
    sample_count: int = 0
    accuracy: float = 0.0
    framework: str = "scikit-learn"
    python_version: str = field(default_factory=lambda: platform.python_version())

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelProvenance":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    # --- persistence ---

    @staticmethod
    def _meta_path(path: str | Path) -> Path:
        return Path(str(path) + ".meta.json")

    def save(self, path: str | Path | None = None) -> Path:
        """Write sidecar .meta.json next to the model file."""
        if not _is_enabled():
            raise RuntimeError(
                "Model provenance is disabled. Set NA0S_MODEL_PROVENANCE=1."
            )
        target = path or self.model_path
        if not target:
            raise ValueError("No model path specified")
        meta = self._meta_path(target)
        meta.write_text(json.dumps(self.to_dict(), indent=2))
        return meta

    @classmethod
    def load(cls, path: str | Path) -> "ModelProvenance":
        """Load provenance from a .meta.json sidecar."""
        if not _is_enabled():
            raise RuntimeError(
                "Model provenance is disabled. Set NA0S_MODEL_PROVENANCE=1."
            )
        meta = cls._meta_path(path)
        data = json.loads(meta.read_text())
        return cls.from_dict(data)

    # --- verification ---

    def verify(self, model_path: str | Path | None = None) -> bool:
        """Check that sha256 in metadata matches the actual file hash."""
        if not _is_enabled():
            raise RuntimeError(
                "Model provenance is disabled. Set NA0S_MODEL_PROVENANCE=1."
            )
        target = model_path or self.model_path
        if not target:
            raise ValueError("No model path specified")
        actual = _sha256(target)
        return actual == self.sha256

    @classmethod
    def create(
        cls,
        model_path: str | Path,
        training_script: str = "",
        dataset_version: str = "",
        feature_count: int = 0,
        sample_count: int = 0,
        accuracy: float = 0.0,
        framework: str = "scikit-learn",
    ) -> "ModelProvenance":
        """Convenience factory that auto-fills hash and timestamp."""
        return cls(
            model_path=str(model_path),
            sha256=_sha256(model_path),
            training_date=datetime.now(timezone.utc).isoformat(),
            training_script=training_script,
            dataset_version=dataset_version,
            feature_count=feature_count,
            sample_count=sample_count,
            accuracy=accuracy,
            framework=framework,
        )
