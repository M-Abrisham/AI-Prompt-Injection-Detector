# Spec: cascade `_transform` char_vectorizer parity (P0)

## Finding (verdict: OVERSIGHT_BUG — confirmed by two independent agent investigations + git archaeology)
`src/na0s/cascade.py:441` (`WeightedClassifier.classify`) calls
`X = _transform(text, vectorizer, scaler)` **without** `char_vectorizer`, while all
5 `predict.py` sites (676, 787, 810, 905, 1683) pass
`char_vectorizer=_get_cached_char_vectorizer()`.

**Root cause:** commit `c53e3ec` ("L2+L4 complete") added char-TF-IDF to `_transform`'s
signature and threaded `char_vectorizer=_char_vec` into every predict.py call site, but
never updated the structurally identical cascade call. `git log --all -S 'char_vectorizer'
-- src/na0s/cascade.py` is EMPTY: the string has never appeared in cascade.py on any
branch — there is no deliberate "charless cascade" design. cascade shares the SAME model
+ loaders as predict (`cascade._ensure_model` → `predict._get_cached_models()`,
`predict._get_cached_scaler()`), so the model expects identical features from both paths.

**Severity:** HIGH (latent). Dormant today only because `char_tfidf_vectorizer.pkl` is
absent from the shipped bundle, so both paths build the same `[word(10000)|struct(29)]
=10029` vector. The instant a char-trained model is deployed (a fully-supported workflow:
`scripts/features.py` builds the char vectorizer, `scripts/deploy_model.py` ships it
conditionally "to avoid a feature-dimension mismatch at inference time"), cascade builds a
too-narrow AND mis-ordered (`[word|struct]` vs trained `[word|char|struct]`) vector →
`ValueError: X has 10000 features, but ... expecting 10029` or silently-wrong scores. This
is the exact failure mode F-AR8 hardened predict's `_transform` against — cascade was left
out of that fix. It is entirely untested (every cascade test mocks `na0s.cascade._transform`).

## Applicable general-prompt steps
1. Explore cascade.py + predict.py `_transform` call sites; confirm the parity gap + edge
   cases (what if `_get_cached_char_vectorizer` import path differs; what other cascade
   stages call `_transform`).
2. Full picture: read `_transform` (predict.py:441), the 5 predict call sites, cascade
   `_ensure_model`/`WeightedClassifier`, `scripts/features.py` (train order
   `[word|char|struct]`), `scripts/deploy_model.py` (char artifact contract), and
   ROADMAP_V2 parity precedent (L5-centroid, g7, D8-G11, RAG-W2).
3. Root-cause implementation plan (below).
4. Implement from root cause + wire into the pipeline with **predict↔cascade PARITY** (the
   governing Na0S rule: a feature/signal must be wired identically into BOTH predict.py and
   cascade.py).
6. Test: add a cascade↔predict feature-dimension PARITY test (code + use-case).
7. Cleanup: no clutter; follow existing patterns.
8. Roadmap: add a DONE entry citing the parity precedent.
9. README/benchmark: not needed (no behavior change today).
11. PR with held-out tests passing.

## N/A steps (code-correctness/parity fix, NOT a new attack type)
5 (harvested datasets), 10 (taxonomy/coverage-matrix/scorer per attack),
Q1/Q5/Q6(harvester audit)/Q9/Q10/Q12. There is no attack-type, dataset, taxonomy code,
or scorer to touch — only feature-assembly parity.

## Implementation (root-cause fix)
In `src/na0s/cascade.py` (`WeightedClassifier.classify`, the line-441 call):
- Extend the existing `from na0s.predict import ...` to also import
  `_get_cached_char_vectorizer`.
- Replace `X = _transform(text, vectorizer, scaler)` with:
  ```python
  char_vec = _get_cached_char_vectorizer()
  X = _transform(text, vectorizer, scaler, char_vectorizer=char_vec)
  ```
This is a **no-op with the current charless bundle** (`char_vec is None` → char block
skipped, exactly as predict behaves today) and correct against any char-trained bundle.
Do NOT add arbitrary thresholds or change decision logic — assembly parity only.

## Test (the test that would have caught this)
Add a parity test (e.g. `tests/structural/test_cascade_predict_parity.py` or in
`tests/test_cascade.py`) that does NOT mock `na0s.cascade._transform`:
- With a fitted word vectorizer + structural scaler (+ a fitted/mock CHAR vectorizer),
  build the feature matrix via the predict path and via the cascade path and assert
  `predict_X.shape[1] == cascade_X.shape[1] == word + char + struct`.
- Run it once with NO char vectorizer (both = 10029) and once with a char vectorizer
  present (both include the char block) — proving parity in both bundle states.

## Acceptance criteria
- cascade and predict produce identical-width feature vectors for the same input under
  both charless and char-present bundles.
- New parity test fails on the pre-fix code (char-present case) and passes after.
- No regression: `tests/test_cascade*.py`, `tests/test_predict*.py`,
  `tests/structural/test_integration.py`, `tests/integrity/` all green.
- Verified in shared venv (`/Users/mehrnoosh/na0s-venv312/bin/python`, numpy 2.4, model
  loads natively).

## Q&A checks
- Q3 (pipeline wiring): cascade now mirrors predict's `_transform` args — parity restored.
- Q4 (tested code+use-case): the parity test exercises the REAL cascade assembly (no mock).
- Q11 (predict/cascade references): cascade now references `_get_cached_char_vectorizer`
  like predict.
- Q13: confirm no OTHER `_transform` caller omits an artifact (grep all call sites).
