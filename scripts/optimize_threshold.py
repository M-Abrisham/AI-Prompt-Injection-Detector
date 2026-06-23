#!/usr/bin/env python
"""
Threshold Optimization Script

Finds the optimal classification threshold to reduce false positives
while preserving detection recall. Sweeps thresholds from 0.01 to 0.99,
computes metrics at each point, and identifies optimal operating points
using Youden's J statistic and a 95%-recall constraint.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold

from na0s.safe_pickle import safe_load

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_PLOTTING = True
except Exception:
    plt = None
    _HAS_PLOTTING = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'tfidf_vectorizer.pkl')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'combined_data.csv')
ROC_PLOT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'roc_curve.png')
PR_PLOT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'pr_curve.png')
THRESHOLD_JSON_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'optimal_threshold.json')
CV_FOLDS = 5


def _compute_oof_probabilities(model, X, labels, n_splits=CV_FOLDS):
    """Return out-of-fold P(malicious) probabilities via stratified k-fold CV."""
    labels = np.asarray(labels)
    n_samples = labels.shape[0]
    if n_samples == 0:
        return np.array([])

    class_counts = np.bincount(labels.astype(int), minlength=2)
    min_class = int(class_counts.min()) if class_counts.size else 0
    if min_class < 2:
        # Not enough samples for CV; fall back to in-sample probabilities.
        return model.predict_proba(X)[:, 1]

    folds = min(int(n_splits), min_class)
    if folds < 2:
        return model.predict_proba(X)[:, 1]

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    oof_probs = np.zeros(n_samples, dtype=float)
    for train_idx, valid_idx in splitter.split(np.zeros(n_samples), labels):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], labels[train_idx])
        oof_probs[valid_idx] = fold_model.predict_proba(X[valid_idx])[:, 1]
    return oof_probs


def main():
    # ------------------------------------------------------------------
    # 1. Load trained model and vectorizer
    # ------------------------------------------------------------------
    print("[1/7] Loading model and vectorizer ...")
    model = safe_load(MODEL_PATH)
    vectorizer = safe_load(VECTORIZER_PATH)
    print(f"      Model:      {MODEL_PATH}")
    print(f"      Vectorizer: {VECTORIZER_PATH}")

    # ------------------------------------------------------------------
    # 2. Load training data
    # ------------------------------------------------------------------
    print("[2/7] Loading training data ...")
    df = pd.read_csv(DATA_PATH)
    texts = df['text'].astype(str)
    labels = df['label'].values
    print(f"      Samples: {len(df)}  (safe={int((labels == 0).sum())}, "
          f"malicious={int((labels == 1).sum())})")

    # ------------------------------------------------------------------
    # 3. Compute out-of-fold predicted probabilities (k-fold CV)
    # ------------------------------------------------------------------
    print(f"[3/7] Computing out-of-fold predicted probabilities ({CV_FOLDS}-fold CV) ...")
    X = vectorizer.transform(texts)
    probs = _compute_oof_probabilities(model, X, labels, n_splits=CV_FOLDS)

    # ------------------------------------------------------------------
    # 4. Sweep thresholds 0.01 .. 0.99
    # ------------------------------------------------------------------
    print("[4/7] Sweeping thresholds (0.01 - 0.99) ...")
    thresholds = np.arange(0.01, 1.00, 0.01)

    positives = (labels == 1)
    negatives = (labels == 0)
    n_pos = positives.sum()
    n_neg = negatives.sum()

    records = []
    best_youden_j = -np.inf
    best_youden_thresh = 0.5
    best_youden_tpr = 0.0
    best_youden_fpr = 1.0

    best_r95_thresh = 0.99
    best_r95_fpr = 1.0
    best_r95_tpr = 1.0

    # GAP-03: FPR-anchored operating point — the highest-recall threshold whose
    # FPR stays within the deployment false-positive budget (Na0S is an embedded
    # defensive SDK; a fixed FP budget matters more than a fixed recall floor).
    target_fpr = float(os.environ.get("NA0S_TARGET_FPR", "0.012"))
    best_fpr_thresh = None
    best_fpr_tpr = -1.0
    best_fpr_fpr = 1.0

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tp = int(((preds == 1) & positives).sum())
        fp = int(((preds == 1) & negatives).sum())
        fn = int(((preds == 0) & positives).sum())

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)
              if (precision + tpr) > 0 else 0.0)
        j_stat = tpr - fpr

        records.append({
            'threshold': round(t, 2),
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'f1': f1,
            'j_stat': j_stat,
        })

        # Track Youden-optimal threshold
        if j_stat > best_youden_j:
            best_youden_j = j_stat
            best_youden_thresh = round(t, 2)
            best_youden_tpr = tpr
            best_youden_fpr = fpr

        # Track 95%-recall threshold with lowest FPR
        if tpr >= 0.95 and fpr < best_r95_fpr:
            best_r95_fpr = fpr
            best_r95_thresh = round(t, 2)
            best_r95_tpr = tpr

        # Track FPR-anchored threshold: highest recall with fpr <= target_fpr.
        if fpr <= target_fpr and tpr > best_fpr_tpr:
            best_fpr_tpr = tpr
            best_fpr_thresh = round(t, 2)
            best_fpr_fpr = fpr

    metrics_df = pd.DataFrame(records)

    # Default (0.5) metrics
    default_row = metrics_df.loc[metrics_df['threshold'] == 0.50].iloc[0]
    default_tpr = float(default_row['tpr'])
    default_fpr = float(default_row['fpr'])

    print(f"      Youden-optimal threshold: {best_youden_thresh}  "
          f"(J={best_youden_j:.4f}, TPR={best_youden_tpr:.4f}, "
          f"FPR={best_youden_fpr:.4f})")
    print(f"      95%-recall threshold:     {best_r95_thresh}  "
          f"(TPR={best_r95_tpr:.4f}, FPR={best_r95_fpr:.4f})")
    if best_fpr_thresh is not None:
        print(f"      target-FPR (<= {target_fpr:.3f}) threshold: {best_fpr_thresh}  "
              f"(TPR={best_fpr_tpr:.4f}, FPR={best_fpr_fpr:.4f})")
    else:
        print(f"      target-FPR (<= {target_fpr:.3f}): NO threshold meets the FPR budget")

    # ------------------------------------------------------------------
    # 5. Plot ROC and PR curves
    # ------------------------------------------------------------------
    if _HAS_PLOTTING:
        print("[5/7] Plotting ROC curve ...")
        sk_fpr, sk_tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(sk_fpr, sk_tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(sk_fpr, sk_tpr, color='steelblue', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

        # Mark Youden-optimal
        ax.scatter([best_youden_fpr], [best_youden_tpr],
                   marker='o', s=120, color='red', zorder=5,
                   label=f'Youden optimal (t={best_youden_thresh})')
        # Mark 95%-recall
        ax.scatter([best_r95_fpr], [best_r95_tpr],
                   marker='^', s=120, color='green', zorder=5,
                   label=f'95%-recall (t={best_r95_thresh})')
        # Mark default 0.5
        ax.scatter([default_fpr], [default_tpr],
                   marker='s', s=100, color='orange', zorder=5,
                   label=f'Default (t=0.50)')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title('ROC Curve - Prompt Injection Detector')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROC_PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"      Saved: {ROC_PLOT_PATH}")

        print("[5/7] Plotting PR curve ...")
        sk_precision, sk_recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(sk_recall, sk_precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(sk_recall, sk_precision, color='steelblue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')

        # Look up precision at the marked thresholds from sweep data
        youden_row = metrics_df.loc[
            metrics_df['threshold'] == best_youden_thresh
        ].iloc[0]
        r95_row = metrics_df.loc[
            metrics_df['threshold'] == best_r95_thresh
        ].iloc[0]

        ax.scatter([youden_row['tpr']], [youden_row['precision']],
                   marker='o', s=120, color='red', zorder=5,
                   label=f'Youden optimal (t={best_youden_thresh})')
        ax.scatter([r95_row['tpr']], [r95_row['precision']],
                   marker='^', s=120, color='green', zorder=5,
                   label=f'95%-recall (t={best_r95_thresh})')
        ax.scatter([default_tpr], [float(default_row['precision'])],
                   marker='s', s=100, color='orange', zorder=5,
                   label=f'Default (t=0.50)')

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Prompt Injection Detector')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PR_PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"      Saved: {PR_PLOT_PATH}")
    else:
        print("[5/7] matplotlib unavailable -- skipping ROC/PR plot generation.")

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print("[6/7] Summary table\n")
    key_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    special = {best_youden_thresh: 'optimal', best_r95_thresh: '95%-recall'}

    # Collect rows to display
    display_rows = []
    for t in key_thresholds:
        row = metrics_df.loc[metrics_df['threshold'] == t]
        if not row.empty:
            r = row.iloc[0]
            tag = special.pop(t, '')
            display_rows.append((r, tag))

    # Add special thresholds that were not already in the fixed list
    for t, tag in special.items():
        row = metrics_df.loc[metrics_df['threshold'] == t]
        if not row.empty:
            display_rows.append((row.iloc[0], tag))

    # Sort by threshold
    display_rows.sort(key=lambda x: x[0]['threshold'])

    header = (f"  {'Threshold':>10}  {'TPR':>8}  {'FPR':>8}  "
              f"{'Precision':>10}  {'F1':>8}  {'J-stat':>8}  {'Note'}")
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for r, tag in display_rows:
        note = f'  <-- {tag}' if tag else ''
        print(f"  {r['threshold']:>10.2f}  {r['tpr']:>8.4f}  "
              f"{r['fpr']:>8.4f}  {r['precision']:>10.4f}  "
              f"{r['f1']:>8.4f}  {r['j_stat']:>8.4f}{note}")

    print()

    # ------------------------------------------------------------------
    # 7. Save optimal threshold JSON
    # ------------------------------------------------------------------
    print("[7/7] Saving optimal threshold JSON ...")
    result = {
        'youden_threshold': best_youden_thresh,
        'youden_j': round(best_youden_j, 6),
        'youden_tpr': round(best_youden_tpr, 6),
        'youden_fpr': round(best_youden_fpr, 6),
        'recall95_threshold': best_r95_thresh,
        'recall95_tpr': round(best_r95_tpr, 6),
        'recall95_fpr': round(best_r95_fpr, 6),
        'default_threshold': 0.5,
        'default_tpr': round(default_tpr, 6),
        'default_fpr': round(default_fpr, 6),
        # GAP-03: FPR-anchored operating point preferred by the runtime
        # (na0s.fusion.voting.get_decision_threshold reads target_fpr_threshold
        # first, recall95_threshold second).  Only emitted when a threshold
        # actually meets the budget, so the runtime cleanly falls back otherwise.
        'target_fpr': target_fpr,
    }
    if best_fpr_thresh is not None:
        result['target_fpr_threshold'] = best_fpr_thresh
        result['target_fpr_tpr'] = round(best_fpr_tpr, 6)
        result['target_fpr_fpr'] = round(best_fpr_fpr, 6)

    with open(THRESHOLD_JSON_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"      Saved: {THRESHOLD_JSON_PATH}")

    # Also write a TRACKED, packaged copy so the calibrated threshold ships in the
    # wheel and `voting.get_decision_threshold()` reads it at runtime (instead of
    # the uncalibrated fallback).  data/processed/ above is gitignored; this one
    # is committed by the retrain's deploy/PR.
    packaged_path = os.path.join(PROJECT_ROOT, 'src', 'na0s', 'models', 'decision_threshold.json')
    os.makedirs(os.path.dirname(packaged_path), exist_ok=True)
    with open(packaged_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"      Saved (packaged): {packaged_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
