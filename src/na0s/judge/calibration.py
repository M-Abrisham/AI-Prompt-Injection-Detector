"""Judge / detector calibration primitives.

Trustworthy metrics for an imbalanced binary security task: confusion-matrix
rates (TPR/TNR/precision/recall — never accuracy as a headline), Rogan-Gladen
prevalence correction (undo the *judge's own* error rate), and percentile
bootstrap confidence intervals.

Design / dependency posture:
  * Pure stdlib + ``numpy`` (a CORE dependency; ``pyproject.toml`` ``numpy>=1.24``).
    ``scipy`` is NOT a core dep and ``judgy`` is an unvetted third-party package,
    so neither is imported here — Rogan-Gladen is one line and the percentile
    bootstrap is a few; keeping ``src/`` free of an unvetted runtime dep matches
    Na0S's supply-chain/integrity posture. ``scipy.stats.bootstrap`` and ``judgy``
    are used only as TEST oracles.
  * Confusion arithmetic mirrors ``scripts/canary_eval.compute_metrics`` exactly so
    this can become the single shared primitive (the four script-local copies are a
    known dedup target).

Why no accuracy headline: on a benign-heavy slice accuracy is dominated by true
negatives and hides a weak detector. Lead with recall/TPR (missed attacks) and
TNR/precision (false alarms), each with a CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


# ── confusion-matrix metrics ────────────────────────────────────────────────


def _safe_div(num: float, den: float) -> float:
    """Mirror canary_eval._safe_div: 0.0 when the denominator is 0."""
    return num / den if den else 0.0


def confusion_metrics(y_true, y_pred) -> dict:
    """Confusion counts + rates for 0/1 label sequences.

    Key-compatible with ``scripts/canary_eval.compute_metrics`` (so callers can
    swap to this shared primitive), plus ``n`` and ``prevalence_apparent`` for
    calibration. Rates use 0.0 on a zero denominator (consistent with the rest
    of the eval stack); callers needing "N/A on empty" should branch on ``n``.
    """
    yt = [int(t) for t in y_true]
    yp = [int(p) for p in y_pred]
    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    n = tp + tn + fp + fn

    tpr = _safe_div(tp, tp + fn)            # recall / sensitivity
    tnr = _safe_div(tn, tn + fp)            # specificity
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "prevalence_apparent": _safe_div(tp + fp, n),  # fraction PREDICTED positive
    }


# stat functions for bootstrap (operate on numpy 0/1 arrays) ------------------

def tpr_stat(yt: np.ndarray, yp: np.ndarray) -> float:
    pos = yt == 1
    return float(np.mean(yp[pos] == 1)) if pos.any() else float("nan")


def tnr_stat(yt: np.ndarray, yp: np.ndarray) -> float:
    neg = yt == 0
    return float(np.mean(yp[neg] == 0)) if neg.any() else float("nan")


def precision_stat(yt: np.ndarray, yp: np.ndarray) -> float:
    pred_pos = yp == 1
    return float(np.mean(yt[pred_pos] == 1)) if pred_pos.any() else float("nan")


def recall_stat(yt: np.ndarray, yp: np.ndarray) -> float:
    return tpr_stat(yt, yp)


# ── Rogan-Gladen prevalence correction ──────────────────────────────────────


def rogan_gladen(apparent_prevalence: float, tpr: float, tnr: float) -> float:
    """Correct an apparent positive rate for a known-imperfect classifier.

    ``true = (apparent + tnr - 1) / (tpr + tnr - 1)`` (Rogan & Gladen, 1978),
    clamped to [0, 1]. ``tpr``/``tnr`` are the classifier's sensitivity /
    specificity estimated on a labeled calibration slice.

    Raises ``ValueError`` when ``tpr + tnr <= 1`` (Youden's J <= 0 — the
    classifier is no better than chance, the correction is undefined). We raise
    rather than silently return 0 so a degenerate judge is never mistaken for a
    clean prevalence.
    """
    denom = tpr + tnr - 1.0
    if denom <= 0:
        raise ValueError(
            f"Rogan-Gladen undefined: tpr+tnr-1={denom:.4g} <= 0 "
            "(classifier no better than chance)"
        )
    corrected = (apparent_prevalence + tnr - 1.0) / denom
    return min(1.0, max(0.0, corrected))


# ── percentile bootstrap CIs (numpy only) ───────────────────────────────────


def bootstrap_ci(y_true, y_pred, stat_fn, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a paired (y_true, y_pred) statistic.

    Resamples item indices with replacement ``n_boot`` times and returns the
    ``(alpha/2, 1-alpha/2)`` percentiles of ``stat_fn(yt_boot, yp_boot)``.
    Draws that yield NaN (e.g. a resample with no positives for TPR) are
    dropped. ``n_boot=2000`` is the conventional floor for a stable 95%
    percentile interval; exposed as a parameter, not buried. Fixed ``seed`` for
    reproducibility.
    """
    yt = np.asarray([int(t) for t in y_true])
    yp = np.asarray([int(p) for p in y_pred])
    n = len(yt)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stat_fn(yt[idx], yp[idx])
        if s is not None and not np.isnan(s):
            stats.append(float(s))
    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def rogan_gladen_ci(y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for the Rogan-Gladen-corrected prevalence.

    Each draw re-estimates apparent prevalence AND tpr/tnr from the resampled
    calibration slice, then propagates them through :func:`rogan_gladen`.
    Degenerate draws (tpr+tnr<=1) are dropped.
    """
    def stat(yt: np.ndarray, yp: np.ndarray) -> float:
        tpr = tpr_stat(yt, yp)
        tnr = tnr_stat(yt, yp)
        apparent = float(np.mean(yp == 1)) if len(yp) else float("nan")
        if np.isnan(tpr) or np.isnan(tnr):
            return float("nan")
        try:
            return rogan_gladen(apparent, tpr, tnr)
        except ValueError:
            return float("nan")

    return bootstrap_ci(y_true, y_pred, stat, n_boot=n_boot, alpha=alpha, seed=seed)


# ── top-level result ────────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    """Trustworthy calibration summary for a binary security classifier.

    Deliberately carries NO ``accuracy`` field as a headline — recall/TPR and
    TNR/precision (with CIs) are the honest metrics on imbalanced data.
    """

    n: int
    prevalence_apparent: float
    tpr: float
    tnr: float
    precision: float
    recall: float
    f1: float
    tpr_ci: tuple[float, float]
    tnr_ci: tuple[float, float]
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    prevalence_corrected: float | None
    prevalence_corrected_ci: tuple[float, float]
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05,
              seed: int = 0) -> CalibrationResult:
    """Full calibration: rates + bootstrap CIs + Rogan-Gladen-corrected prevalence."""
    m = confusion_metrics(y_true, y_pred)
    try:
        corrected = rogan_gladen(m["prevalence_apparent"], m["tpr"], m["tnr"])
    except ValueError:
        corrected = None  # degenerate judge — corrected prevalence not meaningful
    return CalibrationResult(
        n=m["n"],
        prevalence_apparent=m["prevalence_apparent"],
        tpr=m["tpr"], tnr=m["tnr"], precision=m["precision"],
        recall=m["recall"], f1=m["f1"],
        tpr_ci=bootstrap_ci(y_true, y_pred, tpr_stat, n_boot, alpha, seed),
        tnr_ci=bootstrap_ci(y_true, y_pred, tnr_stat, n_boot, alpha, seed),
        precision_ci=bootstrap_ci(y_true, y_pred, precision_stat, n_boot, alpha, seed),
        recall_ci=bootstrap_ci(y_true, y_pred, recall_stat, n_boot, alpha, seed),
        prevalence_corrected=corrected,
        prevalence_corrected_ci=rogan_gladen_ci(y_true, y_pred, n_boot, alpha, seed),
        counts={k: m[k] for k in ("tp", "tn", "fp", "fn")},
    )
