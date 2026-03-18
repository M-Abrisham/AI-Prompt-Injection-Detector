#!/usr/bin/env python3
"""Na0S data quality audit with cleanlab (primary) or sklearn (fallback).

Reads JSONL files containing Na0SSample records, computes per-sample
quality scores using cross-validated TF-IDF + LogisticRegression
probabilities, and writes enriched JSONL output plus a JSON report.

Two execution paths:
  1. Primary — cleanlab ``find_label_issues()`` + quality scores.
  2. Fallback — sklearn ``cross_val_predict`` confidence scoring
     (used when cleanlab is not installed).

Samples are NEVER deleted. The ``quality_score`` field (0.0-1.0) is set
on audited rows; rows beyond ``--max-rows`` pass through with
``quality_score`` left as ``None``.

Usage::

    python scripts/data/cleanlab_audit.py \\
        --input data/samples.jsonl \\
        --output data/samples_scored.jsonl \\
        --report data/audit_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Cleanlab availability
# ---------------------------------------------------------------------------
_CLEANLAB_AVAILABLE = False
_CLEANLAB_IMPORT_ERROR: str | None = None

try:
    from cleanlab.filter import find_label_issues  # noqa: F401
    from cleanlab.rank import get_label_quality_scores  # noqa: F401

    _CLEANLAB_AVAILABLE = True
except ImportError as exc:
    find_label_issues = None  # type: ignore[assignment]
    get_label_quality_scores = None  # type: ignore[assignment]
    _CLEANLAB_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_AUDIT_ROWS = 500_000
DEFAULT_THRESHOLD = 0.4
CV_FOLDS = 5

logger = logging.getLogger("cleanlab_audit_pipeline")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_to_int(label: str) -> int:
    """Convert a Na0SSample label string to int (0=benign, 1=injection)."""
    return 1 if label == "injection" else 0


def compute_out_of_sample_probs(
    texts: list[str],
    labels: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> np.ndarray:
    """Cross-validated TF-IDF + LogisticRegression predicted probabilities.

    Returns an (N, 2) array of class probabilities where column 0 is the
    probability for class 0 (benign) and column 1 for class 1 (injection).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(texts)

    pred_probs = np.zeros((len(labels), 2), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, labels), 1):
        logger.debug("  Fold %d/%d", fold_idx, n_folds)
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        clf.fit(X[train_idx], labels[train_idx])
        pred_probs[val_idx] = clf.predict_proba(X[val_idx])

    return pred_probs


def _quality_scores_cleanlab(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quality scores using cleanlab.

    Returns (quality_scores, issue_mask) where issue_mask is boolean.
    """
    assert _CLEANLAB_AVAILABLE
    quality_scores = get_label_quality_scores(labels=labels, pred_probs=pred_probs)

    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    issue_mask = np.zeros(len(labels), dtype=bool)
    if isinstance(issue_indices, np.ndarray):
        if issue_indices.dtype == bool:
            issue_mask = issue_indices
        else:
            issue_mask[issue_indices] = True

    # Also flag anything below threshold
    issue_mask |= quality_scores < threshold
    return quality_scores, issue_mask


def _quality_scores_sklearn(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quality scores using sklearn confidence (fallback).

    quality_score = predicted probability for the given label.
    """
    quality_scores = pred_probs[np.arange(len(labels)), labels]
    issue_mask = quality_scores < threshold
    return quality_scores, issue_mask


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def audit_samples(
    samples: list[dict[str, Any]],
    *,
    max_rows: int = MAX_AUDIT_ROWS,
    threshold: float = DEFAULT_THRESHOLD,
    n_folds: int = CV_FOLDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the quality audit on a list of Na0SSample dicts.

    Returns (enriched_samples, report_dict).
    Samples beyond *max_rows* pass through with quality_score=None.
    """
    total = len(samples)
    audit_slice = samples[:max_rows]
    passthrough_slice = samples[max_rows:]

    texts = [s.get("text", "") for s in audit_slice]
    labels = np.array([_label_to_int(s.get("label", "benign")) for s in audit_slice])

    # Need at least 2 classes and enough samples per class for CV
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(audit_slice) < n_folds:
        # Cannot cross-validate; assign neutral scores
        logger.warning(
            "Not enough class diversity or samples for CV (n=%d, classes=%d). "
            "Assigning neutral quality_score=0.5.",
            len(audit_slice),
            len(unique_labels),
        )
        for s in audit_slice:
            s["quality_score"] = 0.5
        for s in passthrough_slice:
            s["quality_score"] = None
        report = _build_report(audit_slice, passthrough_slice, threshold, "neutral")
        return samples, report

    # Compute out-of-sample probabilities
    logger.info("Computing out-of-sample probabilities for %d samples...", len(audit_slice))
    pred_probs = compute_out_of_sample_probs(texts, labels, n_folds=n_folds)

    # Choose path
    if _CLEANLAB_AVAILABLE:
        path_used = "cleanlab"
        logger.info("Using cleanlab path for quality scoring.")
        quality_scores, issue_mask = _quality_scores_cleanlab(labels, pred_probs, threshold)
    else:
        path_used = "sklearn_fallback"
        logger.info("Cleanlab unavailable (%s). Using sklearn fallback.", _CLEANLAB_IMPORT_ERROR)
        quality_scores, issue_mask = _quality_scores_sklearn(labels, pred_probs, threshold)

    # Set quality_score on audited samples
    for i, s in enumerate(audit_slice):
        s["quality_score"] = float(quality_scores[i])

    # Passthrough samples get None
    for s in passthrough_slice:
        s["quality_score"] = None

    enriched = audit_slice + passthrough_slice
    report = _build_report(audit_slice, passthrough_slice, threshold, path_used)
    return enriched, report


def _build_report(
    audited: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    threshold: float,
    path_used: str,
) -> dict[str, Any]:
    """Build a JSON-serialisable audit report."""
    total_audited = len(audited)
    total_skipped = len(skipped)

    low_quality = [s for s in audited if s.get("quality_score") is not None and s["quality_score"] < threshold]
    label_issues = low_quality  # in fallback mode, low quality == label issues

    # Worst sources
    source_counter: Counter[str] = Counter()
    for s in low_quality:
        src = s.get("source") or "unknown"
        source_counter[src] += 1
    worst_sources = [{"source": src, "count": cnt} for src, cnt in source_counter.most_common(10)]

    # Worst techniques
    technique_counter: Counter[str] = Counter()
    for s in low_quality:
        tech = s.get("augmentation_type") or s.get("technique_id") or "unknown"
        technique_counter[tech] += 1
    worst_techniques = [{"technique": tech, "count": cnt} for tech, cnt in technique_counter.most_common(10)]

    label_issue_rate = len(label_issues) / total_audited if total_audited > 0 else 0.0

    return {
        "path_used": path_used,
        "total_audited": total_audited,
        "total_skipped": total_skipped,
        "label_issues": len(label_issues),
        "label_issue_rate": round(label_issue_rate, 6),
        "low_quality": len(low_quality),
        "threshold": threshold,
        "worst_sources": worst_sources,
        "worst_techniques": worst_techniques,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str, max_lines: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file, returning a list of dicts."""
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def read_jsonl_streaming(path: str) -> list[dict[str, Any]]:
    """Read all lines from a JSONL file (streaming, no full load into pandas)."""
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_jsonl(path: str, samples: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Na0S data quality audit (cleanlab or sklearn fallback).",
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path (with quality_score).")
    parser.add_argument("--report", required=True, help="Output JSON report path.")
    parser.add_argument("--max-rows", type=int, default=MAX_AUDIT_ROWS, help="Max rows to audit (default 500000).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Quality threshold (default 0.4).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Reading input from %s", args.input)
    samples = read_jsonl_streaming(args.input)
    logger.info("Read %d samples.", len(samples))

    enriched, report = audit_samples(
        samples,
        max_rows=args.max_rows,
        threshold=args.threshold,
    )

    logger.info("Writing enriched output to %s", args.output)
    write_jsonl(args.output, enriched)

    logger.info("Writing report to %s", args.report)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info("Done. Audited=%d, Skipped=%d, Issues=%d",
                report["total_audited"], report["total_skipped"], report["label_issues"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
