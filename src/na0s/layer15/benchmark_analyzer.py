"""Cross-benchmark validation analyzer.

Computes overlap and gaps between Na0S's taxonomy/test corpus and
external benchmarks (JailbreakBench, HarmBench). Produces structured
data for the dashboard visualization.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from na0s.layer15.base import SourceSnapshot, TechniqueEntry
from na0s.layer15.config import SNAPSHOTS_DIR, TAXONOMY_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[\s\-_/.,;:()]+")
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "of", "in", "for", "to", "and", "or", "via",
    "is", "are", "with", "by", "on", "at", "from", "that", "this",
    "it", "as", "be", "was", "were",
}


def _tokenize(text: str) -> Set[str]:
    """Lowercase, split, and remove stop words."""
    tokens = set()
    for tok in _SPLIT_RE.split(text.lower()):
        tok = tok.strip()
        if tok and tok not in _STOP_WORDS and len(tok) > 1:
            tokens.add(tok)
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CategoryCoverage:
    """Coverage data for a single Na0S category against benchmarks."""

    category_id: str
    category_name: str
    na0s_technique_count: int
    benchmark_matches: Dict[str, int]  # {benchmark_name: match_count}
    coverage_level: str  # "strong", "partial", "none"
    gaps: List[str]  # benchmark items with no Na0S match


@dataclass
class BenchmarkAnalysis:
    """Full result of a cross-benchmark validation analysis."""

    timestamp: datetime
    na0s_categories: int
    na0s_techniques: int
    benchmarks: Dict[str, int]  # {name: item_count}
    coverage: List[CategoryCoverage]
    na0s_unique: List[str]  # categories only Na0S covers
    benchmark_unique: List[str]  # benchmark items Na0S doesn't cover
    overall_overlap_pct: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class BenchmarkAnalyzer:
    """Analyzes overlap between Na0S taxonomy and external benchmarks.

    Parameters
    ----------
    taxonomy_path : Path, optional
        Path to the Na0S taxonomy YAML file. Defaults to the repo's
        ``data/taxonomy.yaml``.
    snapshots_dir : Path, optional
        Directory containing benchmark snapshot JSON files. Defaults to
        the repo's ``data/threat_intel_snapshots``.
    """

    # Threshold for Jaccard similarity to count as a match.
    MATCH_THRESHOLD = 0.15
    STRONG_THRESHOLD = 0.4

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        snapshots_dir: Optional[Path] = None,
    ):
        self.taxonomy_path = taxonomy_path or TAXONOMY_PATH
        self.snapshots_dir = snapshots_dir or SNAPSHOTS_DIR

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_taxonomy(self) -> Dict[str, Any]:
        """Load Na0S taxonomy from YAML.

        Returns the parsed ``categories`` dict keyed by category ID.
        """
        import yaml  # optional dep (threatintel)

        with open(self.taxonomy_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("categories", {})

    def load_benchmark_snapshots(self) -> Dict[str, SourceSnapshot]:
        """Load latest snapshots for each benchmark source.

        Scans ``snapshots_dir`` for ``*_snapshot.json`` files whose
        ``source_name`` contains *jailbreakbench* or *harmbench*.
        Returns ``{source_name: SourceSnapshot}``.

        If no snapshot files exist (first run), returns an empty dict.
        """
        snapshots: Dict[str, SourceSnapshot] = {}
        if not self.snapshots_dir.exists():
            logger.info("Snapshots directory does not exist: %s", self.snapshots_dir)
            return snapshots

        # Only load benchmark-related snapshots (not ATLAS, Garak, etc.)
        benchmark_names = {"jailbreakbench", "harmbench"}
        for path in sorted(self.snapshots_dir.glob("*_snapshot.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                snap = SourceSnapshot.from_dict(data)
                # Filter: only include snapshots with benchmark techniques
                if snap.source_name in benchmark_names or any(
                    t.id.startswith(("jailbreakbench", "harmbench"))
                    for t in snap.techniques[:5]  # check first few
                ):
                    snapshots[snap.source_name] = snap
            except (json.JSONDecodeError, KeyError, TypeError, Exception) as exc:
                logger.warning("Skipping invalid snapshot %s: %s", path.name, exc)
        return snapshots

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------

    def _compute_overlap(
        self,
        taxonomy_cats: Dict[str, Any],
        benchmark_techniques: Dict[str, List[TechniqueEntry]],
    ) -> Tuple[List[CategoryCoverage], List[str], List[str]]:
        """Match benchmark items to Na0S categories via fuzzy keywords.

        Returns (coverage_list, na0s_unique_ids, benchmark_unique_names).
        """
        # Build token sets for each Na0S category (name + description +
        # technique names).
        cat_tokens: Dict[str, Set[str]] = {}
        for cat_id, cat_data in taxonomy_cats.items():
            tokens = _tokenize(cat_data.get("name", ""))
            tokens |= _tokenize(cat_data.get("description", ""))
            for tech in (cat_data.get("techniques") or {}).values():
                if isinstance(tech, dict):
                    tokens |= _tokenize(tech.get("name", ""))
            cat_tokens[cat_id] = tokens

        # Build token sets for each benchmark technique.
        bench_item_tokens: Dict[str, Dict[str, Set[str]]] = {}
        for bench_name, techniques in benchmark_techniques.items():
            bench_item_tokens[bench_name] = {}
            for t in techniques:
                tok = _tokenize(t.name) | _tokenize(t.description)
                bench_item_tokens[bench_name][t.id] = tok

        coverage_list: List[CategoryCoverage] = []
        all_matched_bench_ids: Set[str] = set()
        cats_with_any_match: Set[str] = set()

        for cat_id, cat_data in taxonomy_cats.items():
            cat_toks = cat_tokens[cat_id]
            tech_count = len(cat_data.get("techniques") or {})
            matches_per_bench: Dict[str, int] = {}
            gaps: List[str] = []
            best_score = 0.0

            for bench_name, items in bench_item_tokens.items():
                match_count = 0
                for item_id, item_toks in items.items():
                    score = _jaccard(cat_toks, item_toks)
                    if score >= self.MATCH_THRESHOLD:
                        match_count += 1
                        all_matched_bench_ids.add(item_id)
                        if score > best_score:
                            best_score = score
                    else:
                        # Potential gap — but only record if it matches
                        # NO category at all (handled below).
                        pass
                matches_per_bench[bench_name] = match_count

            total_matches = sum(matches_per_bench.values())
            if total_matches > 0:
                cats_with_any_match.add(cat_id)

            if best_score >= self.STRONG_THRESHOLD:
                level = "strong"
            elif best_score >= self.MATCH_THRESHOLD:
                level = "partial"
            else:
                level = "none"

            coverage_list.append(
                CategoryCoverage(
                    category_id=cat_id,
                    category_name=cat_data.get("name", cat_id),
                    na0s_technique_count=tech_count,
                    benchmark_matches=matches_per_bench,
                    coverage_level=level,
                    gaps=gaps,  # filled below
                )
            )

        # Determine benchmark-unique items (no Na0S match at all).
        benchmark_unique: List[str] = []
        for bench_name, items in bench_item_tokens.items():
            for item_id, item_toks in items.items():
                if item_id not in all_matched_bench_ids:
                    # Find the original technique name.
                    for t in benchmark_techniques[bench_name]:
                        if t.id == item_id:
                            benchmark_unique.append(f"[{bench_name}] {t.name}")
                            break

        # Determine Na0S-unique categories (no benchmark match at all).
        na0s_unique = [
            cat_id
            for cat_id in taxonomy_cats
            if cat_id not in cats_with_any_match
        ]

        # Fill per-category gaps: benchmark items that did NOT match
        # this category but matched no other category either.
        unmatched_bench_ids = set()
        for bench_name, items in bench_item_tokens.items():
            for item_id in items:
                if item_id not in all_matched_bench_ids:
                    unmatched_bench_ids.add(item_id)

        # Assign unmatched items as gaps to the *closest* category,
        # but only if similarity is above a minimum floor (0.05) to
        # avoid assigning completely unrelated items.
        GAP_MIN_SIMILARITY = 0.05
        for item_id in unmatched_bench_ids:
            best_cat_idx = -1
            best_sim = -1.0
            item_toks: Set[str] = set()
            bench_name_for_item = ""
            for bn, items in bench_item_tokens.items():
                if item_id in items:
                    item_toks = items[item_id]
                    bench_name_for_item = bn
                    break
            for idx, cov in enumerate(coverage_list):
                sim = _jaccard(cat_tokens[cov.category_id], item_toks)
                if sim > best_sim:
                    best_sim = sim
                    best_cat_idx = idx
            if best_cat_idx >= 0 and best_sim >= GAP_MIN_SIMILARITY:
                for t in benchmark_techniques.get(bench_name_for_item, []):
                    if t.id == item_id:
                        coverage_list[best_cat_idx].gaps.append(
                            f"[{bench_name_for_item}] {t.name}"
                        )
                        break

        return coverage_list, na0s_unique, benchmark_unique

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self) -> BenchmarkAnalysis:
        """Run full analysis: taxonomy vs benchmarks.

        Works even when no benchmark snapshots exist — returns an
        analysis showing "0 benchmarks loaded".
        """
        taxonomy_cats = self.load_taxonomy()
        snapshots = self.load_benchmark_snapshots()

        # Count Na0S techniques.
        total_techniques = 0
        for cat_data in taxonomy_cats.values():
            total_techniques += len(cat_data.get("techniques") or {})

        # Organize benchmark techniques by source name.
        benchmark_techniques: Dict[str, List[TechniqueEntry]] = {}
        benchmark_counts: Dict[str, int] = {}
        for snap_name, snap in snapshots.items():
            # Split combined snapshots (jailbreakbench stores both
            # jailbreakbench and harmbench items together).
            jbb_items: List[TechniqueEntry] = []
            hb_items: List[TechniqueEntry] = []
            other_items: List[TechniqueEntry] = []

            for t in snap.techniques:
                if t.id.startswith("jailbreakbench"):
                    jbb_items.append(t)
                elif t.id.startswith("harmbench"):
                    hb_items.append(t)
                else:
                    other_items.append(t)

            if jbb_items:
                benchmark_techniques["jailbreakbench"] = jbb_items
                benchmark_counts["jailbreakbench"] = len(jbb_items)
            if hb_items:
                benchmark_techniques["harmbench"] = hb_items
                benchmark_counts["harmbench"] = len(hb_items)
            if other_items:
                benchmark_techniques[snap_name] = other_items
                benchmark_counts[snap_name] = len(other_items)

        # Compute overlap.
        if benchmark_techniques:
            coverage, na0s_unique, benchmark_unique = self._compute_overlap(
                taxonomy_cats, benchmark_techniques
            )
        else:
            # No benchmarks — every category shows "none" coverage.
            coverage = [
                CategoryCoverage(
                    category_id=cid,
                    category_name=cdata.get("name", cid),
                    na0s_technique_count=len(cdata.get("techniques") or {}),
                    benchmark_matches={},
                    coverage_level="none",
                    gaps=[],
                )
                for cid, cdata in taxonomy_cats.items()
            ]
            na0s_unique = list(taxonomy_cats.keys())
            benchmark_unique = []

        # Overall overlap percentage: fraction of Na0S categories that
        # have at least partial benchmark coverage.
        cats_with_coverage = sum(
            1 for c in coverage if c.coverage_level != "none"
        )
        total_cats = len(taxonomy_cats)
        overlap_pct = (
            round(cats_with_coverage / total_cats * 100, 1)
            if total_cats
            else 0.0
        )

        return BenchmarkAnalysis(
            timestamp=datetime.now(timezone.utc),
            na0s_categories=total_cats,
            na0s_techniques=total_techniques,
            benchmarks=benchmark_counts,
            coverage=coverage,
            na0s_unique=na0s_unique,
            benchmark_unique=benchmark_unique,
            overall_overlap_pct=overlap_pct,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, analysis: BenchmarkAnalysis) -> str:
        """Serialize analysis for dashboard consumption."""
        data = {
            "timestamp": analysis.timestamp.isoformat(),
            "na0s_categories": analysis.na0s_categories,
            "na0s_techniques": analysis.na0s_techniques,
            "benchmarks": analysis.benchmarks,
            "overall_overlap_pct": analysis.overall_overlap_pct,
            "na0s_unique": analysis.na0s_unique,
            "benchmark_unique": analysis.benchmark_unique,
            "coverage": [
                {
                    "category_id": c.category_id,
                    "category_name": c.category_name,
                    "na0s_technique_count": c.na0s_technique_count,
                    "benchmark_matches": c.benchmark_matches,
                    "coverage_level": c.coverage_level,
                    "gaps": c.gaps,
                }
                for c in analysis.coverage
            ],
        }
        return json.dumps(data, indent=2)
