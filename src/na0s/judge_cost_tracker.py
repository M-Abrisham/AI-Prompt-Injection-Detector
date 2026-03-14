"""Cost tracking for LLM Judge API calls.

Records token usage per model and computes USD costs.  Thread-safe.
"""

import threading
from collections import defaultdict


# Pricing: USD per million tokens
_COST_TABLE = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama-3.3-70b": {"input": 0.59, "output": 0.79},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

# Fallback for unknown models
_DEFAULT_COST = {"input": 0.50, "output": 1.00}


class CostTracker:
    """Track token usage and estimated cost across LLM Judge calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        self._budget: float | None = None

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a single API call."""
        with self._lock:
            entry = self._usage[model]
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["calls"] += 1

    def _cost_for_model(self, model: str) -> float:
        """Compute USD cost for a single model's accumulated usage."""
        entry = self._usage[model]
        rates = _COST_TABLE.get(model, _DEFAULT_COST)
        input_cost = (entry["input_tokens"] / 1_000_000) * rates["input"]
        output_cost = (entry["output_tokens"] / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def get_total_cost(self) -> float:
        """Return total estimated USD cost across all models."""
        with self._lock:
            return sum(self._cost_for_model(m) for m in self._usage)

    def get_breakdown(self) -> dict:
        """Return per-model breakdown of usage and cost."""
        with self._lock:
            result = {}
            for model, entry in self._usage.items():
                result[model] = {
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                    "calls": entry["calls"],
                    "cost_usd": self._cost_for_model(model),
                }
            return result

    def set_budget(self, max_usd: float) -> None:
        """Set a monthly budget cap in USD."""
        with self._lock:
            self._budget = max_usd

    def is_over_budget(self) -> bool:
        """Return True if total cost exceeds the configured budget."""
        with self._lock:
            if self._budget is None:
                return False
            return sum(self._cost_for_model(m) for m in self._usage) > self._budget

    def reset(self) -> None:
        """Clear all usage data (e.g. for a new billing period)."""
        with self._lock:
            self._usage.clear()
