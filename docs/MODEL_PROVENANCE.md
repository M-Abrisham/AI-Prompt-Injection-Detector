# Model Provenance

This document records the training-time environment for the bundled
classifier weights in `src/na0s/models/` and the supported runtime range.

## Bundled artefacts

| File                      | Estimator                                  |
|---------------------------|--------------------------------------------|
| `model.pkl`               | `CalibratedClassifierCV(LogisticRegression)` |
| `tfidf_vectorizer.pkl`    | `TfidfVectorizer` (word n-grams)           |
| `structural_scaler.pkl`   | `StandardScaler` (29 structural features)  |
| `model_embedding.pkl`     | optional embedding-space classifier        |

Authoritative SHA-256 digests live in `src/na0s/models/__init__.py`
(`KNOWN_HASHES`) and are verified on every load by `safe_load()`.

## Training environment

- **scikit-learn**: `1.8.0`
- **numpy**: `>=1.24`
- **Python**: `3.10+`

## Supported runtime range

`pyproject.toml` pins `scikit-learn>=1.3,<2`. Loading the bundled pickles on
an sklearn version that differs from `1.8.0` is safe — sklearn keeps the
internal estimator state stable across minor versions — but `predict_proba`
calibration may drift slightly. Na0S logs a single INFO line on first model
load when the runtime version differs:

    Bundled model trained on sklearn 1.8.0; running <version>. See
    docs/MODEL_PROVENANCE.md for retrain instructions.

The corresponding `sklearn.exceptions.InconsistentVersionWarning` is
suppressed inside `na0s.integrity.safe_pickle.safe_load` so it does not
spam stderr on every cold start. Other warnings still propagate.

## Retraining

To rebuild the model with the runtime sklearn version, regenerate the
pipeline outputs end to end:

    python3 scripts/dataset.py        # gather labelled data
    python3 scripts/process_data.py   # clean + split
    python3 scripts/features.py       # TF-IDF + structural features
    python3 scripts/model.py          # train + calibrate + save

Outputs land in `data/processed/`. Copy the resulting `.pkl` files into
`src/na0s/models/`, then update the SHA-256 digests in
`src/na0s/models/__init__.py` (`KNOWN_HASHES`) and bump the
`_TRAINED_SKLEARN` constant in `src/na0s/predict.py`.

The training data lives outside the repo (DVC-tracked); see `data/raw.dvc`.
