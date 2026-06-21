"""Feature extraction script — TF-IDF + structural features.

Reads ``data/processed/combined_data.csv``, fits a TF-IDF vectorizer and a
StandardScaler for Layer 3 structural features, then writes the combined
sparse feature matrix (TF-IDF columns + scaled structural columns) plus
label vector to ``data/processed/features.pkl``.

The fitted vectorizer and scaler are saved separately so they can be reused
at inference time.
"""

import sys
import os
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from na0s.safe_pickle import safe_dump
from na0s.structural import extract_structural_features_batch

__all__ = ["load_training_data", "apply_stratified_cap", "resolve_max_train_rows"]

# Paths
INPUT_PATH = "data/processed/combined_data.csv"
VECTORIZER_PATH = "data/processed/tfidf_vectorizer.pkl"
CHAR_VECTORIZER_PATH = "data/processed/char_tfidf_vectorizer.pkl"
SCALER_PATH = "data/processed/structural_scaler.pkl"
FEATURES_PATH = "data/processed/features.pkl"

# Minimum number of samples required for reliable training
_MIN_SAMPLES = 10
# TF-IDF vocabulary ceiling
_MAX_FEATURES = 10000
# Character-level TF-IDF vocabulary ceiling (supplementary, smaller)
_CHAR_MAX_FEATURES = 5000
# Drop n-grams that appear in only one document. For a multi-million-row
# corpus the vast majority of word-(1,3) and char_wb-(3,5) n-grams are
# singletons (typos, one-off tokens) that never survive the max_features
# prune anyway, but they still inflate the *intermediate* vocabulary the
# vectorizer materialises before pruning. min_df=2 trims that intermediate
# vocab — standard practice for large corpora, negligible quality impact.
_MIN_DF = 2

# ---------------------------------------------------------------------------
# Memory-safe training-set ceiling (OOM mitigation)
# ---------------------------------------------------------------------------
# The Auto-Retrain GitHub runner OOMs during feature extraction on the full
# ~2.2M-row combined_data.csv: word TfidfVectorizer(1,3) + char_wb(3,5) build
# large intermediate n-gram vocabularies, and extract_structural_features_batch
# materialises a dense (N x len(FEATURE_NAMES)) float64 array — all in RAM at
# once. Peak memory scales ~linearly with row count.
#
# Why 400,000 (not arbitrary): a linear TF-IDF + LogisticRegression model
# saturates on lexical n-gram signal well before millions of rows — the
# marginal AUC gain from 400k -> 2.2M rows of the same distribution is tiny,
# while peak memory keeps climbing. 400k rows keeps word(1,3) + char(3,5) +
# the dense structural array comfortably under a 7-16GB runner with headroom,
# while still being ~4x the rebalanced minority-driven set in practice.
# This is a *deliberate, reversible* model trade-off: set NA0S_MAX_TRAIN_ROWS
# to a larger value (or 0 to disable the cap entirely) on a bigger runner to
# train on the full set. Maintainer should tune to the runner's RAM.
# Lowered 400k -> 150k: 400k still pushed Train model (calibrated LogReg over a
# 150k+ x 15029 matrix) and the 2.3GB features.pkl over a standard runner's
# limits, and the >2GB pickle tripped the os.write truncation boundary (see
# safe_pickle fix). 150k keeps feature extraction + calibrated training fast and
# the features.pkl well under 2GB, while a linear TF-IDF+LogReg saturates far
# below this. Reversible: NA0S_MAX_TRAIN_ROWS=<n> (or 0 to disable) on a bigger
# runner trains on more/all rows.
_DEFAULT_MAX_TRAIN_ROWS = 150000
# Env var that overrides the default ceiling. "0" (or any value <= 0)
# disables the cap entirely (use the full dataset).
_MAX_TRAIN_ROWS_ENV = "NA0S_MAX_TRAIN_ROWS"
# Fixed seed so the downsample is reproducible across runs (matches the
# rebalancing block above, which also uses random_state=42).
_CAP_RANDOM_STATE = 42


def resolve_max_train_rows():
    """Return the effective training-row ceiling.

    Reads ``NA0S_MAX_TRAIN_ROWS`` from the environment; falls back to
    ``_DEFAULT_MAX_TRAIN_ROWS`` when unset or unparseable. A resolved value
    of ``0`` (or any non-positive integer) means *no cap*.
    """
    raw = os.environ.get(_MAX_TRAIN_ROWS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_MAX_TRAIN_ROWS
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(
            f"WARNING: {_MAX_TRAIN_ROWS_ENV}={raw!r} is not an integer; "
            f"falling back to default cap {_DEFAULT_MAX_TRAIN_ROWS}."
        )
        return _DEFAULT_MAX_TRAIN_ROWS


def apply_stratified_cap(dataset, max_rows):
    """Stratified-downsample *dataset* to at most *max_rows* rows.

    Preserves each class's proportion (stratified on ``label``) and is
    deterministic (``random_state=42``). Returns ``dataset`` unchanged when
    ``max_rows`` is non-positive (cap disabled) or when the dataset already
    fits within the ceiling — both no-ops.

    The per-class allocation floors then distributes any rounding remainder
    to the largest classes, so the result has exactly ``min(len(dataset),
    max_rows)`` rows and every class that had >=1 row keeps >=1 row.
    """
    n = len(dataset)
    if max_rows is None or max_rows <= 0 or n <= max_rows:
        return dataset

    # Per-class target counts proportional to class size, summing to max_rows.
    counts = dataset["label"].value_counts()
    targets = {}
    allocated = 0
    for lbl, cnt in counts.items():
        # Floor of the proportional share, but never drop a class to zero.
        share = int(cnt * max_rows // n)
        targets[lbl] = max(1, min(cnt, share))
        allocated += targets[lbl]

    # Distribute the remainder (max_rows - allocated) to the largest classes
    # that still have spare rows, so the totals land exactly on max_rows.
    remainder = max_rows - allocated
    for lbl in counts.index:  # value_counts() is sorted desc by count
        if remainder <= 0:
            break
        spare = int(counts[lbl]) - targets[lbl]
        take = min(spare, remainder)
        targets[lbl] += take
        remainder -= take

    sampled_parts = [
        dataset[dataset["label"] == lbl].sample(
            n=targets[lbl], random_state=_CAP_RANDOM_STATE
        )
        for lbl in counts.index
    ]
    return pd.concat(sampled_parts, ignore_index=True)


def load_training_data():
    # --- Guard: verify input file exists ---
    if not os.path.isfile(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        sys.exit(1)

    try:
        print("Loading...")
        dataset = pd.read_csv(INPUT_PATH)  #import the dataset

        # --- Guard: dataset must not be empty ---
        if dataset.empty or len(dataset) == 0:
            print("ERROR: Dataset is empty (0 rows). Cannot train.")
            sys.exit(1)

        # --- Guard: required columns must be present ---
        missing_cols = [c for c in ("text", "label") if c not in dataset.columns]
        if missing_cols:
            print(f"ERROR: Dataset is missing required column(s): {missing_cols}")
            sys.exit(1)

        dataset['text'] = dataset['text'].fillna('').astype(str)

        # --- Dataset rebalancing (max 3:1 majority:minority ratio) ---
        # Activated when minority class is < 20% of total.
        # Set SKIP_REBALANCE=1 to disable.
        if not os.environ.get("SKIP_REBALANCE", "") == "1":
            label_counts = dataset['label'].value_counts()
            total = len(dataset)
            minority_count = label_counts.min()
            majority_label = label_counts.idxmax()
            minority_label = label_counts.idxmin()

            print(f"Class distribution before rebalancing:")
            for lbl in sorted(label_counts.index):
                cnt = label_counts[lbl]
                pct = cnt / total * 100
                print(f"  Label {lbl}: {cnt} ({pct:.1f}%)")

            if minority_count < 0.20 * total:
                max_majority = minority_count * 3
                majority_df = dataset[dataset['label'] == majority_label]
                minority_df = dataset[dataset['label'] == minority_label]

                if len(majority_df) > max_majority:
                    majority_df = majority_df.sample(
                        n=max_majority, random_state=42
                    )
                    dataset = pd.concat(
                        [majority_df, minority_df], ignore_index=True
                    )
                    print(f"Rebalanced: undersampled label {majority_label} "
                          f"from {label_counts[majority_label]} to {max_majority} "
                          f"(3:1 ratio). Total samples: {len(dataset)}")
                else:
                    print("No rebalancing needed (majority already within 3:1).")
            else:
                print("No rebalancing needed (minority class >= 20% of total).")
        else:
            print("Rebalancing skipped (SKIP_REBALANCE=1).")

        # --- Memory-safe training-set cap (stratified downsample) ---
        # Applied AFTER rebalancing so the class proportions we preserve are
        # the rebalanced ones. Keeps feature extraction tractable on a
        # standard runner; see _DEFAULT_MAX_TRAIN_ROWS for the justification
        # and the NA0S_MAX_TRAIN_ROWS override (0 = disabled).
        max_train_rows = resolve_max_train_rows()
        rows_before_cap = len(dataset)
        if max_train_rows <= 0:
            print(
                f"Training-set cap disabled ({_MAX_TRAIN_ROWS_ENV}=0); "
                f"using all {rows_before_cap} rows."
            )
        elif rows_before_cap > max_train_rows:
            dataset = apply_stratified_cap(dataset, max_train_rows)
            print(
                f"Stratified cap: downsampled from {rows_before_cap} to "
                f"{len(dataset)} rows (ceiling={max_train_rows}, "
                f"{_MAX_TRAIN_ROWS_ENV} override available)."
            )
            capped_counts = dataset["label"].value_counts()
            for lbl in sorted(capped_counts.index):
                cnt = capped_counts[lbl]
                pct = cnt / len(dataset) * 100
                print(f"  Label {lbl}: {cnt} ({pct:.1f}%)")
        else:
            print(
                f"No training-set cap applied ({rows_before_cap} rows "
                f"<= ceiling {max_train_rows})."
            )

        # --- Guard: both class labels must be present ---
        unique_labels = set(dataset['label'].dropna().unique())
        if not {0, 1}.issubset(unique_labels):
            print(
                f"ERROR: Dataset must contain both label 0 (Safe) and label 1 "
                f"(Malicious). Found only: {sorted(unique_labels)}"
            )
            sys.exit(1)

        # --- Guard: minimum sample count ---
        n_samples = len(dataset)
        if n_samples < _MIN_SAMPLES:
            print(
                f"ERROR: Dataset has only {n_samples} sample(s). "
                f"At least {_MIN_SAMPLES} are required for meaningful training."
            )
            sys.exit(1)

        # Warn when vocabulary ceiling exceeds the dataset size
        if _MAX_FEATURES > n_samples:
            print(
                f"WARNING: max_features={_MAX_FEATURES} exceeds dataset size "
                f"({n_samples}). Vocabulary will be limited to unique terms."
            )

        # --- TF-IDF features ---
        vec = TfidfVectorizer(
            lowercase=True,
            max_features=_MAX_FEATURES,
            min_df=_MIN_DF,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )
        X_tfidf = vec.fit_transform(dataset['text'])
        print(f"TF-IDF features: {X_tfidf.shape}")

        # --- Layer 4: Character-level TF-IDF features ---
        char_vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=_CHAR_MAX_FEATURES,
            min_df=_MIN_DF,
            sublinear_tf=True,
        )
        X_char_tfidf = char_vec.fit_transform(dataset['text'])
        print(f"Char TF-IDF features: {X_char_tfidf.shape}")

        # --- Layer 3: Structural features ---
        texts = dataset['text'].tolist()
        print(f"Extracting structural features for {len(texts)} samples...")
        X_structural_raw = extract_structural_features_batch(texts)
        print(f"Structural features: {X_structural_raw.shape}")

        # Fit StandardScaler on structural features (zero mean, unit variance)
        scaler = StandardScaler()
        X_structural_scaled = scaler.fit_transform(X_structural_raw)

        # Combine: sparse word TF-IDF + char TF-IDF + dense structural (converted to sparse)
        X_structural_sparse = scipy.sparse.csr_matrix(X_structural_scaled)
        X = scipy.sparse.hstack([X_tfidf, X_char_tfidf, X_structural_sparse], format="csr")
        y = dataset['label']  # 0 = Safe, 1 = Malicious
        print(f"Combined features: {X.shape} "
              f"({X_tfidf.shape[1]} word TF-IDF + {X_char_tfidf.shape[1]} char TF-IDF"
              f" + {X_structural_raw.shape[1]} structural)")

        # Save word vectorizer
        print(f"Saving vectorizer to {VECTORIZER_PATH}...")
        safe_dump(vec, VECTORIZER_PATH)

        # Save char vectorizer
        print(f"Saving char vectorizer to {CHAR_VECTORIZER_PATH}...")
        safe_dump(char_vec, CHAR_VECTORIZER_PATH)

        # Save structural scaler
        print(f"Saving structural scaler to {SCALER_PATH}...")
        safe_dump(scaler, SCALER_PATH)

        # Save combined features
        print(f"Saving features to {FEATURES_PATH}...")
        safe_dump((X, y), FEATURES_PATH)

    except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError, ValueError) as e:
        print(f"ERROR: Failed to process training data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure during feature extraction: {e}")
        sys.exit(1)

    # --- Verify output files exist and exit non-zero if missing ---
    all_ok = True
    for path in (VECTORIZER_PATH, CHAR_VECTORIZER_PATH, SCALER_PATH, FEATURES_PATH):
        if os.path.isfile(path):
            size = os.path.getsize(path)
            print(f"Verified: {path} ({size:,} bytes)")
        else:
            print(f"ERROR: Expected output file was not created: {path}")
            all_ok = False
    if not all_ok:
        sys.exit(1)

    print(f"Successfully created! Shape: {X.shape}")

# Test
if __name__ == "__main__":
    load_training_data()
