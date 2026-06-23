# Spec: `_transform` edge-case regression tests (P2, test-only)

## Finding
No current bug, but a coverage gap: none of `TestTransformFailLoud` /
`tests/structural/test_integration.py` / `tests/test_subword_features.py` feed
empty / whitespace-only / very-long input through `_transform` with REAL fitted
vectorizers + scaler. Live checks confirm `''`, `'   '`, and a 50,000-word input all
yield a `(1, 10029)` float64 csr matrix and `model.predict` succeeds — so behavior is
correct today, but unlocked (a future `extract_structural_features_batch` change on empty
input could regress silently).

## Applicable general-prompt steps
1. Explore `_transform` behavior on edge-content inputs (empty, whitespace, very long).
2. Full picture: `extract_structural_features_batch` empty-input handling, csr dtype.
4. (test-only — no source change unless an edge actually breaks; if one does, fix from root
   cause and note it.)
6. Add parametrized regression tests.
7. Cleanup; 8. Roadmap; 11. PR.

## N/A steps
3 (no root-cause bug to plan unless found), 5, 9, 10, Q1/Q5/Q6/Q9/Q10/Q12.

## Tests
Parametrized over `''`, `'   '`, `'\n\t'`, and a ~50k-word input, using REAL fitted word
vectorizer + structural scaler (+ optional char vectorizer):
- assert the result is a `scipy.sparse` csr matrix, dtype float64,
- assert `shape == (1, word(+char)+struct)`,
- assert `model.predict(X)` runs without raising.
Place in `tests/structural/` or `tests/test_predict_unit.py` next to the existing
`_transform` tests.

## Acceptance criteria
- New parametrized edge tests pass on current code (locking behavior).
- If any edge input reveals a real bug, it is fixed from root cause and the test asserts
  the corrected behavior (never weaken an assertion to pass).
- Verified shared venv (numpy 2.4).

## Q&A checks
- Q4: edge cases now covered for code + use-case.
- Q13: confirm the very-long input does not trip the scan timeout / does not OOM.
