"""TaxonomyValidator — validate attack_category codes against data/taxonomy.yaml.

The F14 eval library's ``attack_category`` field must reference a real code from
the single-source-of-truth taxonomy. A harvest pipeline that accepted arbitrary
codes would be an injection vector into the eval library (a fabricated category
could silently shift per-category TPR/FPR scoring). This validator loads the
taxonomy once and answers:

- :meth:`validate_code` — is ``code`` a real category/technique code (or a
  resolvable MITRE ATLAS ID)?
- :meth:`get_severity` — what severity does the taxonomy assign that code?
- :meth:`resolve_to_na0s` — what Na0S code does an ATLAS ``AML.Txxxx`` map to?

Both category-level codes (e.g. ``"D1"``, ``"E"``, ``"CT"``) and technique-level
codes (e.g. ``"D1.1"``, ``"E1.1"``, ``"C1MT.3"``) are accepted, because finer-grained
intel may map to a specific technique while the live ``v0.1/`` set uses category-level
codes (``attack_category: D1``).

NOTE: ``BEN`` (the benign-control sentinel) IS canonical — it is a first-class
category in ``data/taxonomy.yaml`` (``BEN``/``BEN.1``/``BEN.2``), so the benign
hard-negative scenarios that drive FPR measurement validate cleanly. ``C2`` and
``M1`` remain phantom (non-canonical — the real top-level codes are ``C`` / ``M``)
and are intentionally NOT accepted; the harvester refuses to propagate them rather
than emit non-canonical codes.

MITRE ATLAS bridge (optional)
-----------------------------
If ``data/threat_intel_snapshots/atlas_to_na0s_mapping.yaml`` is present, ATLAS
technique IDs (``AML.Txxxx``) can be *resolved* to their mapped Na0S code via
:meth:`resolve_to_na0s`, and :meth:`validate_code` will accept an ATLAS ID whose
mapping target is itself a valid Na0S code. The mapping file is optional: when it
is absent (the default, until a human-reviewed mapping is committed),
:meth:`resolve_to_na0s` returns ``None`` and ATLAS IDs are rejected by
:meth:`validate_code` — exactly the pre-bridge behavior. Validation of Na0S codes
is unchanged either way, so existing callers (``IntelScenarioExtractor`` and PR
#437's ``_validated_technique``) keep their semantics.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Resolve data/taxonomy.yaml relative to the repo root. This module lives at
# src/na0s/eval/harvest/taxonomy.py, so the repo root is four parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TAXONOMY_PATH = _REPO_ROOT / "data" / "taxonomy.yaml"

# Optional MITRE ATLAS -> Na0S mapping. Absent by default; an ATLAS sync may
# write it later, and any change to taxonomy.yaml from ATLAS stays human/PR
# reviewed (atlas_sync.apply refuses to auto-write the taxonomy).
_DEFAULT_ATLAS_MAPPING_PATH = (
    _REPO_ROOT / "data" / "threat_intel_snapshots" / "atlas_to_na0s_mapping.yaml"
)

# MITRE ATLAS technique IDs look like ``AML.T0043`` (optionally with a dotted
# sub-technique suffix, ``AML.T0043.001``). Used only to recognize an ATLAS ID
# so it can be routed through the mapping; never to validate it on its own.
_ATLAS_ID_RE = re.compile(r"^AML\.T\d{4}(?:\.\d{3})?$")


class TaxonomyValidator:
    """Validate and look up attack-category codes from ``data/taxonomy.yaml``.

    Parameters
    ----------
    taxonomy_path : Path | None
        Path to the taxonomy YAML. Defaults to ``data/taxonomy.yaml`` at the
        repository root.
    atlas_mapping_path : Path | None
        Path to the optional MITRE ATLAS -> Na0S mapping YAML
        (``{atlas_id: na0s_code}``). Defaults to
        ``data/threat_intel_snapshots/atlas_to_na0s_mapping.yaml``. If the file
        is absent the validator behaves exactly as before — ATLAS resolution is
        a no-op. A malformed mapping file is logged and ignored (it must never
        be able to break validation of canonical Na0S codes).

    Raises
    ------
    FileNotFoundError
        If ``taxonomy_path`` does not exist.
    ValueError
        If the taxonomy YAML is malformed (no ``categories`` mapping).
    """

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        atlas_mapping_path: Optional[Path] = None,
    ) -> None:
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

        # Optional MITRE ATLAS -> Na0S mapping. Load lazily-tolerantly: a
        # missing file is the normal case, and a malformed file must never be
        # able to break canonical Na0S-code validation.
        self.atlas_mapping_path = Path(
            atlas_mapping_path or _DEFAULT_ATLAS_MAPPING_PATH
        )
        self._atlas_mapping: dict[str, str] = self._load_atlas_mapping()

    def _load_atlas_mapping(self) -> dict[str, str]:
        """Load the ATLAS->Na0S mapping, or return {} if absent/malformed.

        Expected file format mirrors ``atlas_sync._load_mapping``::

            AML.T0043: D1.5
            AML.T0051: E1.1

        Only entries whose ATLAS key looks like an ATLAS ID *and* whose target
        resolves to a known Na0S code are kept, so a stale or partially-filled
        mapping can never widen :meth:`validate_code` to accept junk.
        """
        if not self.atlas_mapping_path.is_file():
            return {}
        try:
            raw = yaml.safe_load(
                self.atlas_mapping_path.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            logger.warning(
                "Ignoring malformed ATLAS mapping %s: %r",
                self.atlas_mapping_path,
                exc,
            )
            return {}
        if not isinstance(raw, dict):
            logger.warning(
                "Ignoring ATLAS mapping %s: expected a mapping, got %s",
                self.atlas_mapping_path,
                type(raw).__name__,
            )
            return {}
        mapping: dict[str, str] = {}
        for atlas_id, na0s_code in raw.items():
            atlas_id = str(atlas_id)
            na0s_code = str(na0s_code)
            if not _ATLAS_ID_RE.match(atlas_id):
                logger.warning(
                    "Ignoring ATLAS mapping entry with non-ATLAS key %r",
                    atlas_id,
                )
                continue
            if na0s_code not in self._severity:
                logger.warning(
                    "Ignoring ATLAS mapping %s -> %r: target is not a known "
                    "Na0S code",
                    atlas_id,
                    na0s_code,
                )
                continue
            mapping[atlas_id] = na0s_code
        return mapping

    def resolve_to_na0s(self, code: str) -> Optional[str]:
        """Resolve a MITRE ATLAS technique ID to its mapped Na0S code.

        Returns the mapped Na0S code for an ``AML.Txxxx`` ATLAS ID when the
        optional mapping file is present and contains a valid entry; otherwise
        returns ``None``. Non-ATLAS input (including valid Na0S codes) always
        returns ``None`` — this method only bridges *from* ATLAS *to* Na0S.

        Examples
        --------
        >>> v = TaxonomyValidator()           # no mapping file committed
        >>> v.resolve_to_na0s("AML.T0051") is None
        True
        """
        if not _ATLAS_ID_RE.match(code):
            return None
        return self._atlas_mapping.get(code)

    def validate_code(self, code: str) -> bool:
        """Return True iff ``code`` is a valid taxonomy or resolvable ATLAS code.

        A code is valid when it is a known Na0S category/technique code, OR it
        is a MITRE ATLAS technique ID that resolves (via the optional mapping
        file) to a known Na0S code. Na0S-code semantics are unchanged: when no
        mapping file is present, only canonical Na0S codes validate — exactly
        the pre-ATLAS behavior relied on by ``IntelScenarioExtractor`` and
        PR #437's ``_validated_technique``.

        Examples
        --------
        >>> v = TaxonomyValidator()
        >>> v.validate_code("E1.1")
        True
        >>> v.validate_code("ZZ9.9")
        False
        """
        if code in self._severity:
            return True
        return self.resolve_to_na0s(code) is not None

    def get_severity(self, code: str) -> Optional[str]:
        """Return the taxonomy's severity for ``code``, or None if unknown.

        Returns an empty string if the code exists but the taxonomy records no
        severity for it; returns ``None`` only when the code is not present at
        all (so callers can distinguish "missing" from "blank"). A resolvable
        ATLAS ID returns the severity of its mapped Na0S code.
        """
        if code in self._severity:
            return self._severity[code]
        resolved = self.resolve_to_na0s(code)
        if resolved is not None:
            return self._severity.get(resolved)
        return None

    def known_codes(self) -> frozenset[str]:
        """Return the full set of valid codes (categories + techniques)."""
        return frozenset(self._severity)
