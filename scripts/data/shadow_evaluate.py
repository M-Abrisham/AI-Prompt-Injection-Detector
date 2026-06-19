#!/usr/bin/env python3
"""Shadow evaluation with active learning and Na0SSample integration.

Extends the shadow evaluation pipeline with:
- Batch evaluation through na0s.scan()
- Active learning sample selection strategies
- Model promotion gating based on shadow reports
- CLI for end-to-end evaluation and promotion decisions

Usage::

    # Evaluate samples
    python scripts/data/shadow_evaluate.py --input data/eval.jsonl --output report.json

    # Evaluate with active learning selection
    python scripts/data/shadow_evaluate.py --input data/eval.jsonl --output report.json \\
        --strategy uncertainty --budget 500

    # Compare two shadow reports for promotion
    python scripts/data/shadow_evaluate.py --promotion-gate \\
        --current report_prod.json --candidate report_cand.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from na0s.dataset.schema import DataLabel, Na0SSample

# Confidence threshold below which a false negative is considered "novel"
NOVEL_CONFIDENCE_THRESHOLD = 0.4

# Uncertainty band: samples with confidence in [0.5 - band, 0.5 + band]
UNCERTAINTY_BAND = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_sample(sample: Any) -> dict:
    """Convert a Na0SSample or dict into a uniform dict with text and true_label."""
    if isinstance(sample, Na0SSample):
        return {
            "text": sample.text,
            "true_label": 1 if sample.label == DataLabel.INJECTION else 0,
        }
    if isinstance(sample, dict):
        text = sample.get("text", "")
        raw_label = sample.get("label", sample.get("true_label", 0))
        if isinstance(raw_label, str):
            raw_label = raw_label.strip().lower()
            if raw_label in ("injection", "malicious", "true", "1"):
                true_label = 1
            else:
                true_label = 0
        else:
            true_label = int(raw_label)
        return {"text": text, "true_label": true_label}
    raise TypeError(f"Unsupported sample type: {type(sample)}")


def _evaluate_single(
    sample_dict: dict,
    scan_fn: Optional[Callable] = None,
) -> dict:
    """Evaluate a single sample through the scanner.

    Parameters
    ----------
    sample_dict : dict
        Must have ``text`` and ``true_label`` keys.
    scan_fn : callable or None
        If None, imports and uses ``na0s.scan``.  Accepts a string, returns
        an object with ``.label`` and ``.risk_score`` (or ``.ml_confidence``).
    """
    if scan_fn is None:
        from na0s import scan as _scan
        scan_fn = _scan

    text = sample_dict["text"]
    true_label = sample_dict["true_label"]

    start = time.perf_counter()
    try:
        result = scan_fn(text)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "text": text,
            "true_label": true_label,
            "predicted_label": -1,
            "confidence": 0.0,
            "correct": False,
            "is_fn": False,
            "is_fp": False,
            "is_novel": False,
            "is_uncertain": False,
            "latency_ms": elapsed,
            "error": str(exc),
        }
    elapsed = (time.perf_counter() - start) * 1000

    # Extract prediction
    if hasattr(result, "label"):
        pred_label = 1 if result.label in ("malicious", "injection") else 0
    elif isinstance(result, dict):
        rl = result.get("label", "safe")
        pred_label = 1 if rl in ("malicious", "injection") else 0
    else:
        pred_label = int(result)

    # Extract confidence
    if hasattr(result, "risk_score"):
        confidence = float(result.risk_score)
    elif hasattr(result, "ml_confidence"):
        confidence = float(result.ml_confidence)
    elif isinstance(result, dict):
        confidence = float(result.get("risk_score", result.get("confidence", 0.5)))
    else:
        confidence = 0.5

    correct = pred_label == true_label
    is_fn = true_label == 1 and pred_label == 0
    is_fp = true_label == 0 and pred_label == 1
    is_novel = is_fn and confidence < NOVEL_CONFIDENCE_THRESHOLD
    is_uncertain = abs(confidence - 0.5) < UNCERTAINTY_BAND

    return {
        "text": text,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": confidence,
        "correct": correct,
        "is_fn": is_fn,
        "is_fp": is_fp,
        "is_novel": is_novel,
        "is_uncertain": is_uncertain,
        "latency_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# 1. Batch evaluation
# ---------------------------------------------------------------------------

def shadow_evaluate_batch(
    samples: list,
    batch_size: int = 256,
    parallel: int = 8,
    scan_fn: Optional[Callable] = None,
) -> list[dict]:
    """Evaluate a batch of samples through the Na0S scanner.

    Parameters
    ----------
    samples : list
        List of dicts or :class:`Na0SSample` instances.
    batch_size : int
        Chunk size for processing (controls memory, not parallelism).
    parallel : int
        Number of threads for concurrent evaluation.
    scan_fn : callable or None
        Override the scanner function (useful for testing).

    Returns
    -------
    list[dict]
        Enriched result dicts with prediction metadata.
    """
    normalised = [_normalise_sample(s) for s in samples]
    results: list[dict] = []

    for chunk_start in range(0, len(normalised), batch_size):
        chunk = normalised[chunk_start : chunk_start + batch_size]

        if parallel <= 1:
            for item in chunk:
                results.append(_evaluate_single(item, scan_fn=scan_fn))
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {
                    pool.submit(_evaluate_single, item, scan_fn): idx
                    for idx, item in enumerate(chunk)
                }
                chunk_results = [None] * len(chunk)
                for future in as_completed(futures):
                    idx = futures[future]
                    chunk_results[idx] = future.result()
                results.extend(chunk_results)

    return results


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def compute_shadow_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from shadow evaluation results.

    Returns
    -------
    dict
        Keys: total, tp, tn, fp, fn, precision, recall, f1, fn_rate, fp_rate,
        novel_count, uncertain_count, easy_count, latency_p50_ms, latency_p95_ms.
    """
    # Filter out errored samples
    valid = [r for r in results if r.get("predicted_label", -1) != -1]

    total = len(valid)
    if total == 0:
        return {
            "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "fn_rate": 0.0, "fp_rate": 0.0,
            "novel_count": 0, "uncertain_count": 0, "easy_count": 0,
            "latency_p50_ms": 0.0, "latency_p95_ms": 0.0,
        }

    tp = sum(1 for r in valid if r["true_label"] == 1 and r["predicted_label"] == 1)
    tn = sum(1 for r in valid if r["true_label"] == 0 and r["predicted_label"] == 0)
    fp = sum(1 for r in valid if r["true_label"] == 0 and r["predicted_label"] == 1)
    fn = sum(1 for r in valid if r["true_label"] == 1 and r["predicted_label"] == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    n_positives = tp + fn
    n_negatives = tn + fp
    fn_rate = fn / n_positives if n_positives > 0 else 0.0
    fp_rate = fp / n_negatives if n_negatives > 0 else 0.0

    novel_count = sum(1 for r in valid if r.get("is_novel", False))
    uncertain_count = sum(1 for r in valid if r.get("is_uncertain", False))
    # "easy" = correct predictions that are not uncertain
    easy_count = sum(
        1 for r in valid
        if r["correct"] and not r.get("is_uncertain", False)
    )

    latencies = sorted(r["latency_ms"] for r in valid)
    p50_idx = max(0, int(len(latencies) * 0.50) - 1)
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "novel_count": novel_count,
        "uncertain_count": uncertain_count,
        "easy_count": easy_count,
        "latency_p50_ms": latencies[p50_idx] if latencies else 0.0,
        "latency_p95_ms": latencies[p95_idx] if latencies else 0.0,
    }


def evaluate(jsonl_path: str) -> dict:
    """Evaluate a JSONL dataset end-to-end and return aggregate metrics.

    Reads the JSONL at ``jsonl_path`` (one sample per line, each with at least
    ``text`` and ``label`` fields), runs the samples through
    :func:`shadow_evaluate_batch`, computes metrics via
    :func:`compute_shadow_metrics`, and augments the result with an
    ``accuracy`` field that the underlying helper does not provide.

    Parameters
    ----------
    jsonl_path : str
        Path to a JSONL file; each non-empty line is a JSON object with
        ``text`` and ``label`` (string or int) keys.

    Returns
    -------
    dict
        The metrics dict from :func:`compute_shadow_metrics` with an added
        ``accuracy`` key.  Guaranteed numeric keys include ``accuracy``,
        ``precision``, ``recall``, and ``f1``.
    """
    samples: list = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    results = shadow_evaluate_batch(samples)
    metrics = compute_shadow_metrics(results)

    total = metrics.get("total", 0)
    tp = metrics.get("tp", 0)
    tn = metrics.get("tn", 0)
    metrics["accuracy"] = (tp + tn) / total if total else 0.0

    return metrics


# ---------------------------------------------------------------------------
# 3. Active learning selection
# ---------------------------------------------------------------------------

def select_for_active_learning(
    results: list[dict],
    strategy: str = "uncertainty",
    budget: int = 1000,
) -> list[dict]:
    """Select samples for active learning based on the chosen strategy.

    Parameters
    ----------
    results : list[dict]
        Output from :func:`shadow_evaluate_batch`.
    strategy : str
        One of ``"uncertainty"``, ``"novel"``, ``"balanced"``, ``"fn_priority"``.
    budget : int
        Maximum number of samples to return.

    Returns
    -------
    list[dict]
        Ranked selection, length = min(budget, available).
    """
    valid = [r for r in results if r.get("predicted_label", -1) != -1]

    if strategy == "uncertainty":
        # Sort by distance to 0.5 (ascending — closest first)
        ranked = sorted(valid, key=lambda r: abs(r["confidence"] - 0.5))

    elif strategy == "novel":
        # Only FN samples, sorted by ascending confidence (least confident first)
        fn_samples = [r for r in valid if r["is_fn"]]
        ranked = sorted(fn_samples, key=lambda r: r["confidence"])

    elif strategy == "fn_priority":
        # All FN by ascending confidence
        fn_samples = [r for r in valid if r["is_fn"]]
        ranked = sorted(fn_samples, key=lambda r: r["confidence"])

    elif strategy == "balanced":
        # 50% uncertainty + 30% novel + 20% easy FP
        n_uncertain = max(1, int(budget * 0.50))
        n_novel = max(1, int(budget * 0.30))
        n_easy_fp = max(1, int(budget * 0.20))

        uncertain = sorted(valid, key=lambda r: abs(r["confidence"] - 0.5))[:n_uncertain]
        novel = sorted(
            [r for r in valid if r["is_novel"]],
            key=lambda r: r["confidence"],
        )[:n_novel]
        easy_fp = sorted(
            [r for r in valid if r["is_fp"] and not r.get("is_uncertain", False)],
            key=lambda r: r["confidence"],
            reverse=True,
        )[:n_easy_fp]

        # Deduplicate by text while preserving order
        seen_texts: set[str] = set()
        ranked = []
        for item in (*uncertain, *novel, *easy_fp):
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                ranked.append(item)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return ranked[: min(budget, len(ranked))]


# ---------------------------------------------------------------------------
# 4. Model promotion gate
# ---------------------------------------------------------------------------

def model_promotion_gate(
    current_report: dict,
    candidate_report: dict,
) -> dict:
    """Compare two shadow reports and decide whether to promote the candidate.

    Gates
    -----
    - Recall improved by >= 2 percentage points
    - FP rate no worse by > 1 percentage point
    - F1 improved by >= 1 percentage point

    Parameters
    ----------
    current_report : dict
        Metrics dict (must contain ``recall``, ``fp_rate``, ``f1``).
    candidate_report : dict
        Metrics dict for the candidate model.

    Returns
    -------
    dict
        Keys: promote, reason, current, candidate, deltas, gates.
    """
    cur = current_report
    cand = candidate_report

    recall_delta = cand["recall"] - cur["recall"]
    fp_rate_delta = cand["fp_rate"] - cur["fp_rate"]
    f1_delta = cand["f1"] - cur["f1"]

    gate_recall = recall_delta >= 0.02
    gate_fp_rate = fp_rate_delta <= 0.01
    gate_f1 = f1_delta >= 0.01

    gates = {
        "recall_improved_2pct": gate_recall,
        "fp_rate_no_worse_1pct": gate_fp_rate,
        "f1_improved_1pct": gate_f1,
    }

    promote = all(gates.values())

    reasons = []
    if not gate_recall:
        reasons.append(f"recall delta {recall_delta:+.4f} < 0.02 threshold")
    if not gate_fp_rate:
        reasons.append(f"fp_rate delta {fp_rate_delta:+.4f} > 0.01 threshold")
    if not gate_f1:
        reasons.append(f"f1 delta {f1_delta:+.4f} < 0.01 threshold")

    reason = "PROMOTE: all gates passed" if promote else "BLOCK: " + "; ".join(reasons)

    return {
        "promote": promote,
        "reason": reason,
        "current": cur,
        "candidate": cand,
        "deltas": {
            "recall": recall_delta,
            "fp_rate": fp_rate_delta,
            "f1": f1_delta,
        },
        "gates": gates,
    }


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shadow evaluation with active learning and promotion gating.",
    )
    parser.add_argument(
        "--input", dest="input_path",
        help="Path to JSONL file with samples to evaluate.",
    )
    parser.add_argument(
        "--output", dest="output_path", default="shadow_report.json",
        help="Path to write the shadow report JSON (default: shadow_report.json).",
    )
    parser.add_argument(
        "--strategy", default="uncertainty",
        choices=["uncertainty", "novel", "balanced", "fn_priority"],
        help="Active learning selection strategy (default: uncertainty).",
    )
    parser.add_argument(
        "--budget", type=int, default=1000,
        help="Maximum number of samples for active learning (default: 1000).",
    )
    parser.add_argument(
        "--parallel", type=int, default=8,
        help="Number of parallel threads (default: 8).",
    )
    parser.add_argument(
        "--promotion-gate", action="store_true",
        help="Run promotion gate comparison instead of evaluation.",
    )
    parser.add_argument(
        "--current", dest="current_path",
        help="Path to current model shadow report JSON (for --promotion-gate).",
    )
    parser.add_argument(
        "--candidate", dest="candidate_path",
        help="Path to candidate model shadow report JSON (for --promotion-gate).",
    )

    args = parser.parse_args()

    # ── Promotion gate mode ───────────────────────────────────────
    if args.promotion_gate:
        if not args.current_path or not args.candidate_path:
            parser.error("--promotion-gate requires --current and --candidate")

        with open(args.current_path) as f:
            current_report = json.load(f)
        with open(args.candidate_path) as f:
            candidate_report = json.load(f)

        gate_result = model_promotion_gate(current_report, candidate_report)

        print(json.dumps(gate_result, indent=2))
        with open(args.output_path, "w") as f:
            json.dump(gate_result, f, indent=2)

        sys.exit(0 if gate_result["promote"] else 1)

    # ── Evaluation mode ───────────────────────────────────────────
    if not args.input_path:
        parser.error("--input is required for evaluation mode")

    # Load samples from JSONL
    samples = []
    with open(args.input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.input_path}")

    # Run evaluation
    results = shadow_evaluate_batch(
        samples,
        parallel=args.parallel,
    )

    # Compute metrics
    metrics = compute_shadow_metrics(results)
    print(f"\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Active learning selection
    selected = select_for_active_learning(
        results,
        strategy=args.strategy,
        budget=args.budget,
    )
    print(f"\nActive learning ({args.strategy}): {len(selected)} samples selected")

    # Build report
    report = {
        **metrics,
        "strategy": args.strategy,
        "budget": args.budget,
        "selected_count": len(selected),
        "selected_samples": [
            {k: v for k, v in s.items() if k != "text"}
            for s in selected
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {args.output_path}")


if __name__ == "__main__":
    main()
