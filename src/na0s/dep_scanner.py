"""Lightweight dependency vulnerability scanner (Layer 11).

Analyses installed packages vs requirements.txt for pinning issues.
Gated behind NA0S_DEP_SCAN=1 environment variable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_enabled() -> bool:
    return os.environ.get("NA0S_DEP_SCAN", "0") == "1"


class DependencyScanner:
    """Scan installed Python packages and compare to requirements."""

    def scan_installed(self) -> List[Dict[str, str]]:
        """Return list of installed packages as {name, version} dicts."""
        if not _is_enabled():
            raise RuntimeError("Dep scanning disabled. Set NA0S_DEP_SCAN=1.")
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return []
            packages = json.loads(result.stdout)
            return [{"name": p["name"], "version": p["version"]} for p in packages]
        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            return []

    def _parse_requirements(self, req_file: str | Path) -> List[Dict[str, str]]:
        """Parse a requirements.txt into list of {name, specifier, version}."""
        entries: List[Dict[str, str]] = []
        path = Path(req_file)
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Match: package==1.0, package>=1.0, package~=1.0, package (no pin)
            m = re.match(r"^([A-Za-z0-9_.-]+)\s*(==|>=|<=|~=|!=|>|<)?\s*([\w.*]+)?", line)
            if m:
                name = m.group(1)
                op = m.group(2) or ""
                ver = m.group(3) or ""
                entries.append({"name": name, "specifier": op, "version": ver})
        return entries

    def check_requirements(
        self, req_file: str | Path, installed: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Compare requirements to installed packages.

        Returns list of {name, pinned_version, installed_version, matches}.
        """
        if not _is_enabled():
            raise RuntimeError("Dep scanning disabled. Set NA0S_DEP_SCAN=1.")

        if installed is None:
            installed = self.scan_installed()

        installed_map = {p["name"].lower(): p["version"] for p in installed}
        reqs = self._parse_requirements(req_file)
        results: List[Dict[str, Any]] = []
        for req in reqs:
            name_lower = req["name"].lower()
            inst_ver = installed_map.get(name_lower, "")
            pinned = req["version"] if req["specifier"] == "==" else ""
            matches = (pinned == inst_ver) if pinned else True
            results.append(
                {
                    "name": req["name"],
                    "pinned_version": pinned,
                    "installed_version": inst_ver,
                    "matches": matches,
                }
            )
        return results

    def find_unpinned(self, req_file: str | Path) -> List[str]:
        """Return names of dependencies not pinned with ==."""
        if not _is_enabled():
            raise RuntimeError("Dep scanning disabled. Set NA0S_DEP_SCAN=1.")
        reqs = self._parse_requirements(req_file)
        return [r["name"] for r in reqs if r["specifier"] != "=="]

    def audit_report(
        self, req_file: str | Path, installed: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, int]:
        """Summary counts: total, pinned, unpinned, mismatched."""
        if not _is_enabled():
            raise RuntimeError("Dep scanning disabled. Set NA0S_DEP_SCAN=1.")
        checks = self.check_requirements(req_file, installed=installed)
        pinned = sum(1 for c in checks if c["pinned_version"])
        unpinned = sum(1 for c in checks if not c["pinned_version"])
        mismatched = sum(1 for c in checks if not c["matches"])
        return {
            "total": len(checks),
            "pinned": pinned,
            "unpinned": unpinned,
            "mismatched": mismatched,
        }
