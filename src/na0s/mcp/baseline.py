"""Approved-tool baseline store for MCP rug-pull detection.

A *rug pull* is an MCP tool that ships a benign description, gets approved by the
host, then has its description silently mutated to something malicious.  Catching
it requires remembering what was approved.  :class:`ToolBaselineStore` is that
memory: a tiny, pure-stdlib record of approved tool-description hashes (and,
optionally, the approved descriptions themselves for semantic-drift scoring).

It is deliberately dependency-free (stdlib ``hashlib`` / ``json`` only) so it can
be embedded anywhere the guard runs.  Persistence to a JSON file is optional; the
default is purely in-memory.

The store hands the supply-chain detector a baseline mapping in the exact shape
``na0s.detectors.mcp_supply_chain.detect_rug_pull`` accepts
(``{tool_name: {"hash": ..., "description": ...}}``) via :meth:`as_baseline`, so
no shape-translation glue is needed at the call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger("na0s.mcp.baseline")

__all__ = ["ToolBaselineStore"]


def _hash_desc(desc: str) -> str:
    """SHA-256 of a (possibly empty) description.

    Identical primitive to ``na0s.detectors.mcp_supply_chain._hash_desc`` (itself
    copied from ``worm/advanced.py``); restated here so the store has no
    cross-tier import and the approved hash matches what ``detect_rug_pull``
    re-derives byte-for-byte.
    """
    return hashlib.sha256((desc or "").encode("utf-8")).hexdigest()


class ToolBaselineStore:
    """In-memory (optionally JSON-backed) store of approved tool descriptions.

    Parameters
    ----------
    path : str or None
        Optional path to a JSON file.  When given and the file exists, the store
        is loaded from it on construction; :meth:`save` (and, when
        ``autosave=True``, every :meth:`approve`) persists back to it atomically.
        When ``None`` (default) the store is purely in-memory.
    autosave : bool
        When ``True`` and ``path`` is set, persist on every mutation.  Default
        ``False`` — the caller controls when to flush via :meth:`save`.
    """

    def __init__(self, path: Optional[str] = None, autosave: bool = False) -> None:
        self.path = path
        self.autosave = autosave
        # name -> {"hash": <sha256>, "description": <approved text>}
        self._tools: Dict[str, Dict[str, str]] = {}
        if path and os.path.exists(path):
            self.load(path)

    # -- mutation ----------------------------------------------------------

    def approve(self, name: str, description: str) -> str:
        """Record ``name``'s ``description`` as the approved baseline.

        Returns the stored SHA-256 hash.  Re-approving overwrites the prior
        record (the host has explicitly re-blessed the new description).
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("tool name must be non-empty to approve a baseline")
        desc = description or ""
        digest = _hash_desc(desc)
        self._tools[name] = {"hash": digest, "description": desc}
        if self.autosave and self.path:
            self.save()
        return digest

    def approve_tools(self, tools: List[Dict]) -> None:
        """Approve a batch of tool defs (each ``{"name", "description"}``)."""
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            name = (tool.get("name") or "").strip()
            if name:
                self.approve(name, tool.get("description", ""))

    def forget(self, name: str) -> bool:
        """Drop ``name``'s baseline.  Returns ``True`` if a record was removed."""
        removed = self._tools.pop((name or "").strip(), None) is not None
        if removed and self.autosave and self.path:
            self.save()
        return removed

    def clear(self) -> None:
        """Remove all approved baselines."""
        self._tools.clear()
        if self.autosave and self.path:
            self.save()

    # -- query -------------------------------------------------------------

    def is_approved(self, name: str) -> bool:
        """Whether a baseline exists for ``name``."""
        return (name or "").strip() in self._tools

    def approved_hash(self, name: str) -> Optional[str]:
        """The approved SHA-256 hash for ``name``, or ``None``."""
        entry = self._tools.get((name or "").strip())
        return entry.get("hash") if entry else None

    def known_names(self) -> List[str]:
        """All approved tool names (a typosquat known-name set)."""
        return list(self._tools.keys())

    def as_baseline(self) -> Dict[str, Dict[str, str]]:
        """Return a baseline mapping for ``detect_rug_pull`` / ``scan_tool_supply_chain``.

        Shape: ``{name: {"hash": ..., "description": ...}}`` — exactly the
        ``{tool_name: {"hash"|"description": ...}}`` form
        ``mcp_supply_chain._resolve_baseline`` tolerates.  A shallow copy so the
        caller cannot mutate the store's internals.
        """
        return {name: dict(entry) for name, entry in self._tools.items()}

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.is_approved(name)

    # -- persistence -------------------------------------------------------

    def load(self, path: Optional[str] = None) -> None:
        """Load (replace) the store from a JSON file.

        Tolerates a legacy ``{name: "description"}`` shape (the hash is
        recomputed) as well as the canonical ``{name: {"hash", "description"}}``.
        A malformed/unreadable file logs and leaves the store empty rather than
        raising, so a corrupt baseline file never bricks the guard.
        """
        target = path or self.path
        if not target:
            raise ValueError("no path to load from")
        try:
            with open(target, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("could not load tool baseline from %s: %s", target, exc)
            self._tools = {}
            return

        tools: Dict[str, Dict[str, str]] = {}
        if isinstance(raw, dict):
            entries = raw.get("tools", raw) if "tools" in raw else raw
            if isinstance(entries, dict):
                for name, entry in entries.items():
                    if not isinstance(name, str) or not name.strip():
                        continue
                    if isinstance(entry, dict):
                        desc = entry.get("description", "") or ""
                        digest = entry.get("hash") or _hash_desc(desc)
                        tools[name.strip()] = {"hash": str(digest), "description": desc}
                    elif isinstance(entry, str):
                        tools[name.strip()] = {
                            "hash": _hash_desc(entry),
                            "description": entry,
                        }
        self._tools = tools

    def save(self, path: Optional[str] = None) -> None:
        """Persist the store to a JSON file atomically (temp-file + rename)."""
        target = path or self.path
        if not target:
            raise ValueError("no path to save to")
        payload = {"tools": self.as_baseline()}
        directory = os.path.dirname(os.path.abspath(target)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".tool_baseline.", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, target)
        except Exception:
            # Clean up the temp file on any failure; never leave a partial.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
