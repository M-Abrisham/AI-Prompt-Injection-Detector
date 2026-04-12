"""Software Bill of Materials generation for Na0S (Layer 11).

Produces a simplified CycloneDX-lite SBOM covering the Na0S package, its
runtime dependencies, and any deployed model artefacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_na0s_version() -> str:
    """Best-effort retrieval of the installed na0s version."""
    try:
        from na0s._version import __version__

        return __version__
    except Exception:
        return "unknown"


def _get_dependencies() -> List[Dict[str, str]]:
    """Return a list of ``{name, version}`` dicts for installed na0s deps.

    Uses ``importlib.metadata`` to read the package's requirements, then
    resolves each to its installed version.
    """
    deps: list[Dict[str, str]] = []
    try:
        from importlib.metadata import distribution, requires

        reqs = requires("na0s") or []
        for req_line in reqs:
            # Lines look like  "numpy>=1.20" or "scikit-learn ; extra == ..."
            if "extra ==" in req_line:
                continue  # skip optional extras
            name = req_line.split(";")[0].strip()
            # Strip version specifiers to get bare package name
            for sep in (">=", "<=", "==", "!=", ">", "<", "~="):
                name = name.split(sep)[0]
            name = name.strip()
            try:
                ver = distribution(name).version
            except Exception:
                ver = "unknown"
            deps.append({"name": name, "version": ver})
    except Exception:
        pass
    return deps


def _discover_models(models_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Find ``.pkl`` model files and return metadata for each."""
    if models_dir is None:
        # Default: look in src/na0s/models
        models_dir = Path(__file__).resolve().parent / "models"
    models: list[Dict[str, Any]] = []
    if not models_dir.is_dir():
        return models
    for p in sorted(models_dir.glob("*.pkl")):
        models.append(
            {
                "filename": p.name,
                "sha256": _sha256(p),
                "size_bytes": p.stat().st_size,
            }
        )
    return models


class SBOMGenerator:
    """Generate, save, load and verify a CycloneDX-lite SBOM.

    Parameters
    ----------
    models_dir : Path | str | None
        Directory to scan for ``.pkl`` model files.  Defaults to
        ``src/na0s/models`` relative to the package.
    """

    def __init__(self, models_dir: Optional[str | Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else None

    def generate(self) -> Dict[str, Any]:
        """Build and return the SBOM dict."""
        return {
            "format": "CycloneDX-lite",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {
                "name": "na0s",
                "version": _get_na0s_version(),
            },
            "dependencies": _get_dependencies(),
            "models": _discover_models(self.models_dir),
            "python_version": sys.version,
        }

    def save(self, output_path: str | Path) -> Path:
        """Generate the SBOM and write it as JSON to *output_path*."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sbom = self.generate()
        path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> Dict[str, Any]:
        """Read an SBOM JSON file and return its contents."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def verify_models(self, sbom: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check model hashes in *sbom* against files on disk.

        Returns a list of dicts with keys ``filename``, ``expected``,
        ``actual``, and ``match`` for every model entry in the SBOM.
        """
        models_dir = self.models_dir
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"

        results: list[Dict[str, Any]] = []
        for entry in sbom.get("models", []):
            model_path = models_dir / entry["filename"]
            expected = entry["sha256"]
            if model_path.exists():
                actual = _sha256(model_path)
            else:
                actual = None
            results.append(
                {
                    "filename": entry["filename"],
                    "expected": expected,
                    "actual": actual,
                    "match": actual == expected,
                }
            )
        return results
