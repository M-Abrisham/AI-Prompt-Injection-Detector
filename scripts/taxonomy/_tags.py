"""MISP tag loading and aggregation helpers."""

import functools
import importlib.resources
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_project_root():
    """Locate the project root using importlib.resources with Path fallback.

    Tries importlib.resources first (works in zipped/installed packages),
    then falls back to __file__-based resolution for script-mode execution.
    """
    try:
        pkg_dir = importlib.resources.files("scripts.taxonomy")
        root = Path(str(pkg_dir)).resolve().parent.parent
        if (root / "data").is_dir():
            return root
    except (TypeError, FileNotFoundError, ModuleNotFoundError, Exception):
        pass
    return Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _find_project_root()
_DEFAULT_TAGS = str(_PROJECT_ROOT / "data" / "tags.misp.tsv")
_TAGS_PATH = Path(os.environ.get("TAGS_MISP_PATH", _DEFAULT_TAGS))

@functools.lru_cache(maxsize=1)
def load_tags():
    """Load data/tags.misp.tsv into a dict {tag: description} (thread-safe via lru_cache)."""
    if not _TAGS_PATH.exists():
        raise FileNotFoundError(
            f"MISP tags file not found: {_TAGS_PATH}. "
            "Set TAGS_MISP_PATH env var or ensure data/tags.misp.tsv exists."
        )
    result = {}
    with _TAGS_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                tag_key = parts[0]
                if tag_key in result:
                    logger.warning(
                        "tags.misp.tsv line %d: duplicate tag '%s', "
                        "keeping first occurrence", line_num, tag_key
                    )
                    continue
                result[tag_key] = parts[1]
            else:
                logger.warning(
                    "tags.misp.tsv line %d: expected tab-separated "
                    "'tag\\tdescription', got: %s", line_num, line[:80]
                )
    return result


def clear_tag_cache():
    """Reset cached tag data (for tests and live-reload)."""
    load_tags.cache_clear()


def aggregate_by_taxonomy(probe_results, namespace):
    """Group probe results by a taxonomy namespace.

    Args:
        probe_results: list of dicts from Probe.evaluate()
        namespace: prefix to filter tags, e.g. "owasp-llm", "avid-effect",
                   "risk-cards".  A ":" delimiter is enforced so "owasp"
                   won't accidentally match "owasp-llm:..." tags.

    Returns:
        dict {tag: {"description": str, "total": int, "detected": int,
                     "recall": float, "probes": list}}
    """
    prefix = namespace if namespace.endswith(":") else namespace + ":"
    tags = load_tags()
    groups = {}
    for result in probe_results:
        for tag in result.get("tags", []):
            if not tag.startswith(prefix):
                continue
            if tag not in groups:
                desc = tags.get(tag)
                if desc is None:
                    logger.warning(
                        "Tag '%s' from probe '%s' not found in MISP tags",
                        tag, result.get("probe", "?"),
                    )
                    desc = tag
                groups[tag] = {
                    "description": desc,
                    "total": 0,
                    "detected": 0,
                    "attributed": 0,
                    "probes": [],
                }
            groups[tag]["total"] += result["total"]
            groups[tag]["detected"] += result["detected"]
            groups[tag]["attributed"] += result.get("attributed", 0)
            groups[tag]["probes"].append(result["probe"])

    for g in groups.values():
        g["missed"] = g["total"] - g["detected"]
        g["recall"] = g["detected"] / g["total"] if g["total"] else 0.0
        g["attribution_rate"] = (
            g["attributed"] / g["detected"] if g["detected"] else 0.0
        )

    return groups


def summarize_groups(groups, namespace=None):
    """Compute aggregate stats across tag groups from aggregate_by_taxonomy().

    Returns:
        dict with namespace, tag_count, total, detected, missed, recall.
    """
    total = sum(g["total"] for g in groups.values())
    detected = sum(g["detected"] for g in groups.values())
    attributed = sum(g["attributed"] for g in groups.values())
    return {
        "namespace": namespace,
        "tag_count": len(groups),
        "total": total,
        "detected": detected,
        "attributed": attributed,
        "missed": total - detected,
        "recall": detected / total if total else 0.0,
        "attribution_rate": attributed / detected if detected else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-probe and per-technique aggregation helpers
# ---------------------------------------------------------------------------

def count_by_probe(results):
    """Count detection stats per probe category.

    Args:
        results: list of probe result dicts as returned by Probe.evaluate().
            Each dict must have at minimum: ``probe`` (category_id),
            ``total``, ``detected``, ``missed``.  ``false_positives`` is
            optional (defaults to 0).

    Returns:
        dict mapping each probe's category_id to:
            total_samples (int), detected (int), missed (int),
            recall (float 0.0-1.0), false_positives (int).
    """
    counts = {}
    for r in results:
        pid = r.get("probe", "unknown")
        total = r.get("total", 0)
        detected = r.get("detected", 0)
        missed = r.get("missed", total - detected)
        fp = r.get("false_positives", 0)
        if pid in counts:
            counts[pid]["total_samples"] += total
            counts[pid]["detected"] += detected
            counts[pid]["missed"] += missed
            counts[pid]["false_positives"] += fp
        else:
            counts[pid] = {
                "total_samples": total,
                "detected": detected,
                "missed": missed,
                "false_positives": fp,
            }
    # Compute recall after accumulation
    for v in counts.values():
        denom = v["detected"] + v["missed"]
        v["recall"] = v["detected"] / denom if denom else 0.0
    return counts


def top_missed_techniques(results, n=10):
    """Return the top-N technique IDs with the most misses.

    Args:
        results: list of probe result dicts (each must contain
            ``by_technique``).
        n: number of top entries to return (default 10).

    Returns:
        list of dicts sorted by missed_count descending::

            [{"technique_id": str, "missed_count": int,
              "total_count": int, "miss_rate": float}, ...]
    """
    merged = {}  # technique_id -> {detected, missed}
    for r in results:
        by_tech = r.get("by_technique", {})
        for tid, stats in by_tech.items():
            if tid not in merged:
                merged[tid] = {"detected": 0, "missed": 0}
            merged[tid]["detected"] += stats.get("detected", 0)
            merged[tid]["missed"] += stats.get("missed", 0)

    entries = []
    for tid, s in merged.items():
        if s["missed"] == 0:
            continue
        total = s["detected"] + s["missed"]
        entries.append({
            "technique_id": tid,
            "missed_count": s["missed"],
            "total_count": total,
            "miss_rate": s["missed"] / total if total else 0.0,
        })
    entries.sort(key=lambda e: (-e["missed_count"], e["technique_id"]))
    return entries[:n]


def aggregation_summary(results):
    """Enhanced aggregation summary across multiple probe results.

    Combines per-probe counts, top missed techniques, per-difficulty
    breakdown, and per-evasion-type breakdown into a single report.

    Args:
        results: list of probe result dicts from Probe.evaluate().

    Returns:
        dict with keys:
            per_probe: output of count_by_probe()
            top_missed: output of top_missed_techniques(n=10)
            by_difficulty: merged per-difficulty-level breakdown
            by_evasion_type: merged per-evasion-type breakdown
            overall: {total, detected, missed, recall, false_positives}
    """
    per_probe = count_by_probe(results)
    top_missed = top_missed_techniques(results, n=10)

    # Merge difficulty breakdowns across probes
    by_diff = {}
    for r in results:
        for level, stats in r.get("by_difficulty", {}).items():
            if level not in by_diff:
                by_diff[level] = {"detected": 0, "missed": 0, "total": 0}
            by_diff[level]["detected"] += stats.get("detected", 0)
            by_diff[level]["missed"] += stats.get("missed", 0)
            by_diff[level]["total"] += stats.get("total", 0)
    for v in by_diff.values():
        v["recall"] = v["detected"] / v["total"] if v["total"] else 0.0

    # Merge evasion-type breakdowns across probes
    by_evasion = {}
    for r in results:
        for etype, stats in r.get("by_evasion_type", {}).items():
            if etype not in by_evasion:
                by_evasion[etype] = {"detected": 0, "missed": 0, "total": 0}
            by_evasion[etype]["detected"] += stats.get("detected", 0)
            by_evasion[etype]["missed"] += stats.get("missed", 0)
            by_evasion[etype]["total"] += stats.get("total", 0)
    for v in by_evasion.values():
        v["recall"] = v["detected"] / v["total"] if v["total"] else 0.0

    # Overall totals
    total_detected = sum(p["detected"] for p in per_probe.values())
    total_missed = sum(p["missed"] for p in per_probe.values())
    total_fp = sum(p["false_positives"] for p in per_probe.values())
    total_all = total_detected + total_missed

    return {
        "per_probe": per_probe,
        "top_missed": top_missed,
        "by_difficulty": by_diff,
        "by_evasion_type": by_evasion,
        "overall": {
            "total": total_all,
            "detected": total_detected,
            "missed": total_missed,
            "recall": total_detected / total_all if total_all else 0.0,
            "false_positives": total_fp,
        },
    }
