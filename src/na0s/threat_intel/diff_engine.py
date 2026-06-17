"""Taxonomy Diff Engine — compares two taxonomy snapshots.

This is the core utility for Layer 15. Every sync module produces a
SourceSnapshot; the diff engine compares old vs. new and outputs a
structured TaxonomyDiff plus a human-readable Markdown changelog.

Design notes:
- Diffing is done by technique ID (the canonical key).
- "Modified" means same ID but different name, description, or severity.
- "Reclassified" is treated as a subtype of "modified" (category changed).
- The engine is source-agnostic — it works with any SourceSnapshot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from na0s.threat_intel.base import (
    DiffItem,
    SourceSnapshot,
    TaxonomyDiff,
    TechniqueEntry,
)

logger = logging.getLogger(__name__)


class TaxonomyDiffEngine:
    """Compares two SourceSnapshots and produces a TaxonomyDiff.

    Usage::

        engine = TaxonomyDiffEngine()
        diff = engine.compute(old_snapshot, new_snapshot)
        md = engine.to_markdown(diff)
        js = engine.to_json(diff)
    """

    def compute(
        self,
        old: SourceSnapshot,
        new: SourceSnapshot,
    ) -> TaxonomyDiff:
        """Compute the diff between two snapshots.

        Parameters
        ----------
        old : SourceSnapshot
            The previous (baseline) snapshot. May have an empty techniques
            list on the first sync.
        new : SourceSnapshot
            The freshly fetched snapshot.

        Returns
        -------
        TaxonomyDiff
            Structured diff with added/removed/modified items.
        """
        old_by_id: Dict[str, TechniqueEntry] = {
            t.id: t for t in old.techniques
        }
        new_by_id: Dict[str, TechniqueEntry] = {
            t.id: t for t in new.techniques
        }

        old_ids = set(old_by_id.keys())
        new_ids = set(new_by_id.keys())

        items: List[DiffItem] = []

        # --- Added techniques ---
        for tid in sorted(new_ids - old_ids):
            t = new_by_id[tid]
            items.append(
                DiffItem(
                    change_type="added",
                    technique_id=tid,
                    technique_name=t.name,
                    new_value=_technique_to_dict(t),
                    na0s_mapping=t.category,
                    needs_review=not t.category,
                )
            )

        # --- Removed techniques ---
        for tid in sorted(old_ids - new_ids):
            t = old_by_id[tid]
            items.append(
                DiffItem(
                    change_type="removed",
                    technique_id=tid,
                    technique_name=t.name,
                    old_value=_technique_to_dict(t),
                    na0s_mapping=t.category,
                )
            )

        # --- Modified / reclassified techniques ---
        for tid in sorted(old_ids & new_ids):
            old_t = old_by_id[tid]
            new_t = new_by_id[tid]
            changes = _detect_changes(old_t, new_t)
            if changes:
                # If category changed, it's a reclassification
                change_type = (
                    "reclassified"
                    if "category" in changes
                    else "modified"
                )
                items.append(
                    DiffItem(
                        change_type=change_type,
                        technique_id=tid,
                        technique_name=new_t.name,
                        old_value=_technique_to_dict(old_t),
                        new_value=_technique_to_dict(new_t),
                        na0s_mapping=new_t.category,
                        needs_review="category" in changes,
                    )
                )

        diff = TaxonomyDiff(
            source_name=new.source_name,
            old_version=old.version,
            new_version=new.version,
            items=items,
        )

        logger.info(
            "Diff computed for %s: %d added, %d removed, %d modified",
            new.source_name,
            len(diff.added),
            len(diff.removed),
            len(diff.modified),
        )
        return diff

    def to_markdown(self, diff: TaxonomyDiff) -> str:
        """Render a TaxonomyDiff as a human-readable Markdown changelog.

        Suitable for inclusion in a GitHub PR body or issue comment.
        """
        lines: List[str] = []
        lines.append(f"# Threat Intel Sync — {diff.source_name}")
        lines.append("")
        lines.append(
            f"**Versions**: `{diff.old_version or '(none)'}` → `{diff.new_version}`"
        )
        lines.append(
            f"**Timestamp**: {diff.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        lines.append("")

        if not diff.has_changes:
            lines.append("No changes detected.")
            return "\n".join(lines)

        lines.append(
            f"**Summary**: {len(diff.added)} added, "
            f"{len(diff.removed)} removed, "
            f"{len(diff.modified)} modified"
        )
        if diff.unmapped:
            lines.append(
                f"  ⚠️ {len(diff.unmapped)} items need manual review (unmapped)"
            )
        lines.append("")

        if diff.added:
            lines.append("## ➕ New Techniques")
            lines.append("")
            lines.append("| ID | Name | Na0S Mapping | Review Needed |")
            lines.append("|---|---|---|---|")
            for item in diff.added:
                mapping = item.na0s_mapping or "UNMAPPED"
                review = "⚠️ Yes" if item.needs_review else "No"
                lines.append(
                    f"| `{item.technique_id}` | {item.technique_name} "
                    f"| {mapping} | {review} |"
                )
            lines.append("")

        if diff.removed:
            lines.append("## ➖ Removed Techniques")
            lines.append("")
            lines.append("| ID | Name |")
            lines.append("|---|---|")
            for item in diff.removed:
                lines.append(
                    f"| `{item.technique_id}` | {item.technique_name} |"
                )
            lines.append("")

        if diff.modified:
            lines.append("## ✏️ Modified Techniques")
            lines.append("")
            for item in diff.modified:
                lines.append(f"### `{item.technique_id}` — {item.technique_name}")
                if item.old_value and item.new_value:
                    changes = _dict_diff(item.old_value, item.new_value)
                    for field_name, (old_val, new_val) in changes.items():
                        lines.append(f"- **{field_name}**: `{old_val}` → `{new_val}`")
                lines.append("")

        return "\n".join(lines)

    def to_json(self, diff: TaxonomyDiff, indent: int = 2) -> str:
        """Render a TaxonomyDiff as machine-readable JSON."""
        return json.dumps(diff.to_dict(), indent=indent)

    def save_report(
        self,
        diff: TaxonomyDiff,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Save both Markdown and JSON reports to disk.

        Returns a dict with keys 'markdown' and 'json' pointing to the
        saved file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = diff.timestamp.strftime("%Y%m%d_%H%M%S")
        base = f"{diff.source_name}_{ts}"

        md_path = output_dir / f"{base}.md"
        json_path = output_dir / f"{base}.json"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown(diff))
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(diff))

        logger.info("Reports saved: %s, %s", md_path, json_path)
        return {"markdown": md_path, "json": json_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _technique_to_dict(t: TechniqueEntry) -> Dict[str, Any]:
    """Convert a TechniqueEntry to a flat dict for diff comparison."""
    return {
        "id": t.id,
        "name": t.name,
        "description": t.description,
        "severity": t.severity,
        "category": t.category,
    }


def _detect_changes(
    old: TechniqueEntry, new: TechniqueEntry
) -> Dict[str, tuple]:
    """Return a dict of {field: (old_value, new_value)} for changed fields."""
    changes: Dict[str, tuple] = {}
    for field_name in ("name", "description", "severity", "category"):
        old_val = getattr(old, field_name)
        new_val = getattr(new, field_name)
        if old_val != new_val:
            changes[field_name] = (old_val, new_val)
    return changes


def _dict_diff(
    old: Dict[str, Any], new: Dict[str, Any]
) -> Dict[str, tuple]:
    """Return fields that differ between two flat dicts."""
    changes: Dict[str, tuple] = {}
    all_keys = set(old.keys()) | set(new.keys())
    for k in sorted(all_keys):
        old_val = old.get(k)
        new_val = new.get(k)
        if old_val != new_val:
            changes[k] = (old_val, new_val)
    return changes
