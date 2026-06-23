# Spec: `_transform` dimension fail-loud guards (P2)

Two related fail-loud hardenings that complete the F-AR8 feature-contract. Both make a
feature-dimension mismatch surface LOUDLY instead of as a cryptic per-request error or
silently-wrong scores.

## Finding A — no load-time width reconciliation (LOW/MED)
Nothing asserts that the loaded model's `n_features_in_` (currently 10029) equals
`word_vocab(10000) + char(0 or char_vocab) + structural(29)`. The components are loaded by
independent cached loaders (word vectorizer, optional char vectorizer, optional scaler,
classifier) with no cross-check, so a missing/stale/mismatched artifact only surfaces as a
per-request `ValueError: X has N features` deep in `model.predict`.

**Fix:** a one-time validation at `preload_models` / first scan:
```
expected = len(word_vec.get_feature_names_out())
         + (len(char_vec.get_feature_names_out()) if char_vec is not None else 0)
         + (29 if (scaler is not None and _HAS_STRUCTURAL_FEATURES) else 0)
assert expected == model.n_features_in_, <clear error naming the missing/extra component>
```
Raise a clear, named error (which component is missing/extra) — no arbitrary constants
(29 is the documented structural-feature count; cite its source).

## Finding B — `_HAS_STRUCTURAL_FEATURES=False` + non-None scaler silently drops structural (LOW)
`_transform` guards the structural branch with `if scaler is not None and
_HAS_STRUCTURAL_FEATURES`. If a bundle ships `structural_scaler.pkl` (so the cached scaler
is a real object) but the structural module failed to import (`_HAS_STRUCTURAL_FEATURES`
False), the 29 structural columns are silently dropped → an under-width vector with no
fail-loud error.

**Fix:** when `scaler is not None` but `_HAS_STRUCTURAL_FEATURES` is False, log at `error`
and raise — a shipped scaler implies the model expects structural features this build
cannot produce (consistent with F-AR8). Preserve the legit `scaler is None` skip.

## Applicable general-prompt steps
1. Explore `_transform` + `preload_models` + the cached loaders + `_HAS_STRUCTURAL_FEATURES`
   import guard (predict.py ~111-112). Find edge cases.
2. Full picture: `scripts/features.py` (29 structural features, train width), `model.pkl`
   `n_features_in_` decomposition, F-AR8 docstring.
3. Root-cause plan (above).
4. Implement (predict.py); confirm cascade path benefits via the shared model/loaders.
6. Tests for both A and B.
7. Cleanup; 8. Roadmap; 11. PR.

## N/A steps
5, 10, Q1/Q5/Q6/Q9/Q10/Q12.

## Tests
- A: a deliberately mismatched bundle (e.g. scaler absent against a 10029-model, or a
  wrong-vocab word vectorizer) is caught AT LOAD with a clear component-naming error.
- B: `scaler is not None` + `_HAS_STRUCTURAL_FEATURES` monkeypatched False → `_transform`
  raises (does not silently return an under-width vector). The legit `scaler is None` skip
  still returns word(+char)-only.

## Acceptance criteria
- Stale/mismatched artifact bundles fail at load with a clear, component-named error.
- A shipped-scaler-but-no-structural-module build fails loud rather than under-width.
- No false positives on the valid shipped bundle (load guard passes for 10029).
- No regression; verified shared venv (numpy 2.4).

## Q&A checks
- Q3 (wiring): both guards apply to the shared loaders/model used by predict AND cascade.
- Q4: both A and B have tests.
- Q13: ensure the load guard runs once (not per-request) and does not break the legitimate
  charless/backward-compat bundle.
