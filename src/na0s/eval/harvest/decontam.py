"""Eval decontamination set for the harvester ingestion bridge.

Builds the canonical set of *eval* stable_ids — every text the model is
measured against — so that newly-fetched candidate training rows that match
an eval row can be dropped BEFORE they reach quarantine/staging/training.

This is a thin, importable wrapper around the exact contract already pinned by
``tests/test_no_holdout_leakage.py`` and implemented by
``scripts/check_eval_decontamination.py``:

* ``stable_id`` is ``SHA-256(NFKC-normalized, whitespace-collapsed text)`` —
  identical to :meth:`na0s.dataset.schema.Na0SSample.__post_init__` and
  :meth:`Scenario._compute_stable_id`, so a candidate row's hash compares
  equal to an eval row's hash.
* Eval sources are the F14 scenario library (``data/eval/scenarios/v0.1``,
  per-turn for multi-turn scenarios) PLUS the holdout / benchmark JSONL sets
  (``data/holdout/*.jsonl``, ``data/benchmark/*.jsonl``), keyed off the
  ``text`` / ``prompt`` / ``stable_id`` fields the leakage test reads.

The bridge calls :func:`build_eval_decontam_set` once, then
:meth:`EvalDecontaminator.is_contaminated` per candidate row.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

# Repo root: this file is src/na0s/eval/harvest/decontam.py -> 4 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_SCENARIOS_DIR = _REPO_ROOT / "data" / "eval" / "scenarios" / "v0.1"
DEFAULT_HOLDOUT_DIR = _REPO_ROOT / "data" / "holdout"
DEFAULT_BENCHMARK_DIR = _REPO_ROOT / "data" / "benchmark"


def compute_stable_id(text: str) -> str:
    """Return ``SHA-256(NFKC + whitespace-collapsed text)``.

    Mirrors ``Na0SSample.__post_init__`` / ``Scenario._compute_stable_id`` /
    ``check_eval_decontamination.compute_stable_id`` so candidate-row hashes
    compare equal to eval-row hashes across pipeline stages.
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _scenario_ids(scenarios_dir: Path) -> set[str]:
    """Collect stable_ids from the F14 scenario library.

    Delegates to the same loader-based collector used by the standalone
    decontam check so the two stay in lock-step. Falls back to an empty set
    if the eval loader / scenarios dir is unavailable (e.g. minimal fixture
    checkouts) — the holdout/benchmark JSONL path still applies.
    """
    if not scenarios_dir.exists():
        return set()
    try:
        # Reuse the exact scenario-id collector from the check script.
        import sys

        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from check_eval_decontamination import _collect_scenario_ids  # type: ignore

        return set(_collect_scenario_ids(scenarios_dir).keys())
    except Exception:
        return set()


def _jsonl_eval_ids(directory: Path) -> set[str]:
    """Stable_ids from holdout/benchmark JSONL rows.

    Reads the same fields the leakage test keys off: ``text`` (preferred),
    then ``prompt``; an explicit ``stable_id`` field is also honored directly.
    """
    ids: set[str] = set()
    if not directory.exists():
        return ids
    for path in sorted(directory.rglob("*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    sid = row.get("stable_id")
                    if sid:
                        ids.add(str(sid).strip())
                    text = row.get("text") or row.get("prompt")
                    if text:
                        ids.add(compute_stable_id(str(text)))
        except OSError:
            continue
    return ids


class EvalDecontaminator:
    """Holds the eval stable_id set and answers membership queries."""

    def __init__(self, eval_ids: Optional[Iterable[str]] = None):
        self.eval_ids: set[str] = set(eval_ids or ())

    def __len__(self) -> int:
        return len(self.eval_ids)

    def is_contaminated(self, text: str) -> bool:
        """True if ``text`` matches any eval row (by normalized hash)."""
        if not text:
            return False
        return compute_stable_id(text) in self.eval_ids

    def filter_rows(self, rows, text_getter=lambda r: r.get("text", "")):
        """Split rows into (accepted, dropped) by contamination.

        ``rows`` is any iterable; ``text_getter`` extracts the comparable
        text from each row. Returns two lists.
        """
        accepted, dropped = [], []
        for row in rows:
            if self.is_contaminated(text_getter(row)):
                dropped.append(row)
            else:
                accepted.append(row)
        return accepted, dropped


def build_eval_decontam_set(
    scenarios_dir: Optional[Path] = None,
    holdout_dir: Optional[Path] = None,
    benchmark_dir: Optional[Path] = None,
) -> EvalDecontaminator:
    """Build the eval decontamination set from scenarios + holdout + benchmark.

    All three sources default to the repo's standard locations. Missing
    sources are skipped (not an error) so the bridge runs against partial
    checkouts and tiny local fixtures.
    """
    scenarios_dir = Path(scenarios_dir) if scenarios_dir else DEFAULT_SCENARIOS_DIR
    holdout_dir = Path(holdout_dir) if holdout_dir else DEFAULT_HOLDOUT_DIR
    benchmark_dir = Path(benchmark_dir) if benchmark_dir else DEFAULT_BENCHMARK_DIR

    ids: set[str] = set()
    ids |= _scenario_ids(scenarios_dir)
    ids |= _jsonl_eval_ids(holdout_dir)
    ids |= _jsonl_eval_ids(benchmark_dir)
    return EvalDecontaminator(ids)
