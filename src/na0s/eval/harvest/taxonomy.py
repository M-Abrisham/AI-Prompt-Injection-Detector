"""TaxonomyValidator — validate attack_category codes against data/taxonomy.yaml.

The F14 eval library's ``attack_category`` field must reference a real code from
the single-source-of-truth taxonomy. A harvest pipeline that accepted arbitrary
codes would be an injection vector into the eval library (a fabricated category
could silently shift per-category TPR/FPR scoring). This validator loads the
taxonomy once and answers two questions:

- :meth:`validate_code` — is ``code`` a real category *or* technique code?
- :meth:`get_severity` — what severity does the taxonomy assign that code?

Both category-level codes (e.g. ``"D1"``, ``"E"``, ``"CT"``) and technique-level
codes (e.g. ``"D1.1"``, ``"E1.1"``, ``"C1MT.3"``) are accepted, because finer-grained
intel may map to a specific technique while the live ``v0.1/`` set uses category-level
codes (``attack_category: D1``).

NOTE: this validator enforces the *canonical* taxonomy only. A few codes used by the
current ``v0.1/`` scenarios are deliberately NOT accepted because they are absent from
``data/taxonomy.yaml``: ``C2`` and ``M1`` (non-canonical — the real top-level codes are
``C`` / ``M``) and ``BEN`` (the benign sentinel, not an attack category). That is a
pre-existing data-quality inconsistency in the eval set, tracked separately; the
harvester refuses to propagate it rather than emit non-canonical codes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

# Resolve data/taxonomy.yaml relative to the repo root. This module lives at
# src/na0s/eval/harvest/taxonomy.py, so the repo root is four parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TAXONOMY_PATH = _REPO_ROOT / "data" / "taxonomy.yaml"


class TaxonomyValidator:
    """Validate and look up attack-category codes from ``data/taxonomy.yaml``.

    Parameters
    ----------
    taxonomy_path : Path | None
        Path to the taxonomy YAML. Defaults to ``data/taxonomy.yaml`` at the
        repository root.

    Raises
    ------
    FileNotFoundError
        If ``taxonomy_path`` does not exist.
    ValueError
        If the taxonomy YAML is malformed (no ``categories`` mapping).
    """

    def __init__(self, taxonomy_path: Optional[Path] = None) -> None:
        self.taxonomy_path = Path(taxonomy_path or _DEFAULT_TAXONOMY_PATH)
        if not self.taxonomy_path.is_file():
            raise FileNotFoundError(
                f"Taxonomy file not found: {self.taxonomy_path}"
            )
        raw = yaml.safe_load(self.taxonomy_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(
            raw.get("categories"), dict
        ):
            raise ValueError(
                f"Taxonomy {self.taxonomy_path} has no 'categories' mapping"
            )
        # Flatten into {code: severity}. Category codes and technique codes both
        # become keys; severity falls back through technique -> category -> "".
        self._severity: dict[str, str] = {}
        categories: dict[str, Any] = raw["categories"]
        for cat_code, cat_body in categories.items():
            if not isinstance(cat_body, dict):
                continue
            self._severity[cat_code] = str(cat_body.get("severity", ""))
            techniques = cat_body.get("techniques")
            if not isinstance(techniques, dict):
                continue
            for tech_code, tech_body in techniques.items():
                if isinstance(tech_body, dict):
                    severity = str(
                        tech_body.get("severity")
                        or cat_body.get("severity", "")
                    )
                else:
                    severity = str(cat_body.get("severity", ""))
                self._severity[tech_code] = severity

    def validate_code(self, code: str) -> bool:
        """Return True iff ``code`` is a known category or technique code.

        Examples
        --------
        >>> v = TaxonomyValidator()
        >>> v.validate_code("E1.1")
        True
        >>> v.validate_code("ZZ9.9")
        False
        """
        return code in self._severity

    def get_severity(self, code: str) -> Optional[str]:
        """Return the taxonomy's severity for ``code``, or None if unknown.

        Returns an empty string if the code exists but the taxonomy records no
        severity for it; returns ``None`` only when the code is not present at
        all (so callers can distinguish "missing" from "blank").
        """
        return self._severity.get(code)

    def known_codes(self) -> frozenset[str]:
        """Return the full set of valid codes (categories + techniques)."""
        return frozenset(self._severity)
