"""Validate + (de)serialize per-source taxonomy codes for ``data/datasets.yaml``.

Each source in the dataset registry may declare a ``taxonomy_codes:`` list of
CANONICAL ``data/taxonomy.yaml`` codes describing the attack class(es) its rows
provide provenance for. Benign/baseline corpora declare ``["BEN"]`` (the
canonical benign-control sentinel) — never an invented attack code.

This module is pure read + validate. It REUSES
:class:`na0s.eval.harvest.taxonomy.TaxonomyValidator` (the ATLAS-anchored gate
PR #449 shipped) rather than re-deriving the canonical-code set, so a taxonomy
edit can never leave this validator out of sync. YAML is loaded via a local
size-guarded ``yaml.safe_load`` (mirroring ``scripts.safe_yaml``) so the module
imports cleanly whether driven from pytest, the package, or a bare CLI run.

Security contract (mirrors ``discovery_tagging`` / ``_validated_technique``):

- **Never invent a code.** :func:`validate_registry_codes` flags — and the
  registry-canonical test rejects — any code on any source that is not in the
  live taxonomy. An invented code therefore cannot reach the training corpus
  via the registry.
- **Absent / empty is allowed.** A source with no ``taxonomy_codes`` (or an
  empty list) is *untagged*, a valid state — exactly the ``None``-return
  contract of ``tag_discovery``. Untagged is never an error; only a code that
  is PRESENT and non-canonical is.
- **Pure / local / keyless.** No network, no LLM. Deterministic given the
  registry + the committed taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from na0s.eval.harvest.taxonomy import TaxonomyValidator

# 10 MB size guard mirrors scripts.safe_yaml (billion-laughs / zip-bomb DoS).
# We re-implement the guard here rather than importing scripts.safe_yaml so this
# library module stays importable however it is loaded — a bare CLI run puts
# scripts/ itself (not the repo root) on sys.path, which would make
# ``import scripts.safe_yaml`` fail. (Verified: that exact ImportError surfaced
# when scripts/taxonomy_coverage.py imported this module.)
_MAX_YAML_BYTES = 10 * 1024 * 1024


def _safe_load_yaml(path: Path) -> Any:
    """Size-guarded ``yaml.safe_load`` of a file (never ``yaml.load``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    size = path.stat().st_size
    if size > _MAX_YAML_BYTES:
        raise ValueError(
            f"YAML file too large ({size:,} bytes, limit "
            f"{_MAX_YAML_BYTES:,}): {path}"
        )
    try:
        with path.open("r", encoding="utf-8-sig") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

# This module lives at src/na0s/eval/registry/taxonomy_labels.py, so the repo
# root is four parents up (mirrors taxonomy.py's _REPO_ROOT derivation).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_REGISTRY_PATH = _REPO_ROOT / "data" / "datasets.yaml"

# The single owner of the flat-CSV-cell encoding. A list of codes is joined
# with this separator so it survives a flat ``combined_data.csv`` cell. Chosen
# because no canonical taxonomy code contains ';' (asserted in serialize_codes,
# so a future code that did would fail loudly rather than silently corrupt the
# round-trip).
CODE_SEPARATOR = ";"


@dataclass(frozen=True)
class RegistryCodeError:
    """One non-canonical / malformed ``taxonomy_codes`` finding.

    Attributes
    ----------
    source : str
        The ``data/datasets.yaml`` source name (e.g. ``"deepset_injections"``).
    code : str
        The offending code as written in the registry.
    reason : str
        Why it was flagged (e.g. ``"not a known taxonomy code"``).
    """

    source: str
    code: str
    reason: str


def load_registry(path: Optional[Path] = None) -> dict[str, Any]:
    """Load ``data/datasets.yaml`` via the size-guarded safe loader.

    Parameters
    ----------
    path : Path | None
        Registry path. Defaults to ``data/datasets.yaml`` at the repo root.

    Returns
    -------
    dict
        The parsed registry mapping (``version`` / ``output_dir`` / ``sources``).

    Raises
    ------
    FileNotFoundError
        If the registry file does not exist.
    ValueError
        If the registry YAML is malformed or has no ``sources`` mapping.
    """
    registry = _safe_load_yaml(Path(path or _DEFAULT_REGISTRY_PATH))
    if not isinstance(registry, dict) or not isinstance(
        registry.get("sources"), dict
    ):
        raise ValueError(
            f"Registry {path or _DEFAULT_REGISTRY_PATH} has no 'sources' mapping"
        )
    return registry


def _coerce_codes(value: Any) -> list[str]:
    """Normalize a source's ``taxonomy_codes`` value to a list of strings.

    Accepts a list (the canonical form), a bare scalar (coerced to a
    1-element list, defensively — but never invented), or absent/None (empty
    list). Non-string list members are coerced to ``str`` so a malformed entry
    is *flagged* by the validator rather than crashing it.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    # A bare scalar (e.g. ``taxonomy_codes: BEN``) — coerce, do not invent.
    return [str(value)]


def iter_source_codes(registry: dict[str, Any]) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(source_name, codes_list)`` for every source in the registry.

    ``codes_list`` is the source's ``taxonomy_codes`` coerced to a list of
    strings (empty when absent). Sources are yielded in registry order.
    """
    sources = registry.get("sources", {})
    for name, cfg in sources.items():
        if not isinstance(cfg, dict):
            continue
        yield name, _coerce_codes(cfg.get("taxonomy_codes"))


def validate_registry_codes(
    registry: dict[str, Any],
    validator: Optional[TaxonomyValidator] = None,
) -> list[RegistryCodeError]:
    """Flag every non-canonical / malformed ``taxonomy_codes`` entry.

    For each source, each declared code is checked against the live taxonomy
    via :meth:`TaxonomyValidator.validate_code`. A blank/whitespace code is
    flagged as malformed. Absent or empty ``taxonomy_codes`` is NOT an error
    (untagged is a valid state).

    Parameters
    ----------
    registry : dict
        A parsed registry (from :func:`load_registry`).
    validator : TaxonomyValidator | None
        The canonical-code gate. Defaults to a fresh ``TaxonomyValidator()``.

    Returns
    -------
    list[RegistryCodeError]
        One finding per offending ``(source, code)``. Empty when every
        declared code on every source is canonical — the invariant the
        registry-canonical test asserts.
    """
    validator = validator or TaxonomyValidator()
    errors: list[RegistryCodeError] = []
    for source, codes in iter_source_codes(registry):
        for code in codes:
            if not code or not code.strip():
                errors.append(
                    RegistryCodeError(source, code, "empty or blank code")
                )
                continue
            if not validator.validate_code(code):
                errors.append(
                    RegistryCodeError(
                        source, code, "not a known taxonomy code"
                    )
                )
    return errors


def serialize_codes(codes: list[str]) -> str:
    """Join a list of codes into a single ``;``-separated CSV cell.

    The inverse of :func:`parse_codes`. An empty list serializes to ``""``.

    Raises
    ------
    ValueError
        If any code contains :data:`CODE_SEPARATOR` — no canonical code does,
        so this guards the round-trip against a future taxonomy code that
        would otherwise corrupt the flat-cell encoding silently.
    """
    out: list[str] = []
    for code in codes or []:
        code = str(code)
        if CODE_SEPARATOR in code:
            raise ValueError(
                f"taxonomy code {code!r} contains the reserved separator "
                f"{CODE_SEPARATOR!r}; cannot serialize"
            )
        out.append(code)
    return CODE_SEPARATOR.join(out)


def parse_codes(cell: Optional[str]) -> list[str]:
    """Split a ``;``-separated CSV cell back into a list of codes.

    The inverse of :func:`serialize_codes`. An empty / blank / None cell parses
    to ``[]``. Blank fragments (e.g. from a trailing separator) are dropped.
    """
    if not cell:
        return []
    return [frag for frag in str(cell).split(CODE_SEPARATOR) if frag.strip()]
