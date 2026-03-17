# Ensemble Pipeline Research: TF-IDF (L4) + Embeddings (L5)

**Date**: 2026-02-18
**Scope**: Best approach for combining L4 TF-IDF and L5 Sentence Embedding classifiers
**Status**: Research complete, ready for implementation

---

## Key Finding: Confidence Scale Mismatch Bug

**BUG (MEDIUM)** in cascade.py line 472-474:
- `confidence` from WeightedClassifier = composite score (ML + rules + obfuscation) or 1-composite
- `emb_conf` from classify_prompt_embedding = P(malicious)
- These are on DIFFERENT scales -- blending them is mathematically unsound
- **Fix**: Both inputs must be raw P(malicious) from predict_proba()

## Recommended Approach

**Adaptive Confidence-Weighted Averaging with Disagreement Resolution**

1. Extract raw P(malicious) from both models BEFORE any composite scoring
2. Weight each model by its own confidence (dynamically per sample)
3. On disagreement: favor embedding when TF-IDF triggers keywords but embedding says safe (FP reduction)
4. On disagreement with corroboration: trust TF-IDF when rules/structural signals confirm (attack detection)
5. Apply rule/obfuscation/structural boosts AFTER ensemble (not mixed into ML probability)
6. Graceful degradation to TF-IDF-only when embedding unavailable

**Fallback**: Fixed-weight averaging (alpha=0.55) with raw P(malicious)

## Methods Evaluated

| Method | Verdict | Why |
|--------|---------|-----|
| Simple Averaging | Baseline/fallback | Too naive, fixed weights |
| Confidence-Weighted | **RECOMMENDED** | Dynamic, no training needed |
| Stacking (meta-learner) | Phase 3 upgrade | Best accuracy but needs held-out training data |
| Bayesian Fusion | Rejected | Independence assumption violated (same training data) |
| Dempster-Shafer | Rejected | Over-engineered, no production precedent |

## Architecture Decisions

1. Use raw P(malicious), not composite scores
2. Asymmetric disagreement: favor SAFE when TF-IDF=MAL, Embed=SAFE (unless rules corroborate)
3. Post-ensemble rule boosting (not pre-ensemble composite)
4. Standard 0.50 threshold (calibrated models don't need shifted threshold)
5. New file: `src/na0s/ensemble.py` (Single Responsibility)

## Calibration Status

- Both models use CalibratedClassifierCV(cv=5, method='isotonic') -- EXCELLENT
- Isotonic is non-parametric, best calibration method available
- Key requirement: ensemble inputs MUST be the calibrated probabilities, not composites

## Competitive Insight

- NO competitor uses principled ensemble of heterogeneous classifiers
- LLM Guard: single DeBERTa model
- Vigil: OR-voting (any method triggers = malicious) -- maximizes recall, kills precision
- Meta Prompt Guard 2: single model
- Na0S ensemble is a genuine differentiator

## Implementation Phases

- Phase 1: Confidence-weighted averaging + disagreement resolution in ensemble.py
- Phase 2: Grid search for optimal weights on held-out data
- Phase 3: Stacking meta-learner + Prompt Guard 2 as third classifier
