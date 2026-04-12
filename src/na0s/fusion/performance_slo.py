"""Performance SLO (Service Level Objective) tracking for Na0S pipeline stages.

Records per-stage timing data and checks against configurable latency budgets.
Enabled via the ``NA0S_SLO_TRACKING=1`` environment variable to ensure zero
overhead in production when not needed.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class SLOTracker:
    """Track per-stage latency and flag SLO violations.

    Parameters
    ----------
    whitelist_ms : float
        Maximum acceptable latency for the whitelist stage (default 1.0 ms).
    weighted_ms : float
        Maximum acceptable latency for the weighted classifier stage (default 10.0 ms).
    judge_ms : float
        Maximum acceptable latency for the LLM judge stage (default 5000.0 ms).
    """

    def __init__(
        self,
        whitelist_ms: float = 1.0,
        weighted_ms: float = 10.0,
        judge_ms: float = 5000.0,
    ):
        self._budgets: dict[str, float] = {
            "whitelist": whitelist_ms,
            "weighted": weighted_ms,
            "judge": judge_ms,
        }
        self._history: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    # -- Configuration -------------------------------------------------------

    def set_budget(self, stage: str, budget_ms: float) -> None:
        """Set or update the SLO budget for a stage."""
        with self._lock:
            self._budgets[stage] = budget_ms

    # -- Recording -----------------------------------------------------------

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Record a timing observation for *stage*."""
        with self._lock:
            self._history[stage].append(elapsed_ms)

    # -- Querying ------------------------------------------------------------

    def check_violations(self) -> list[dict]:
        """Return a list of SLO violation dicts.

        Each dict has keys ``stage``, ``budget_ms``, ``actual_ms``, ``index``
        identifying which observation violated the SLO.
        """
        violations: list[dict] = []
        with self._lock:
            for stage, observations in self._history.items():
                budget = self._budgets.get(stage)
                if budget is None:
                    continue
                for idx, elapsed in enumerate(observations):
                    if elapsed > budget:
                        violations.append({
                            "stage": stage,
                            "budget_ms": budget,
                            "actual_ms": elapsed,
                            "index": idx,
                        })
        return violations

    def get_stats(self) -> dict:
        """Return ``{stage: {count, p50, p95, p99, mean, max}}`` for each stage."""
        stats: dict[str, dict] = {}
        with self._lock:
            for stage, observations in self._history.items():
                if not observations:
                    continue
                sorted_obs = sorted(observations)
                n = len(sorted_obs)
                stats[stage] = {
                    "count": n,
                    "p50": _percentile(sorted_obs, 50),
                    "p95": _percentile(sorted_obs, 95),
                    "p99": _percentile(sorted_obs, 99),
                    "mean": sum(sorted_obs) / n,
                    "max": sorted_obs[-1],
                }
        return stats

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded history."""
        with self._lock:
            self._history.clear()


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile of already-sorted *sorted_data*."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    k = (pct / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
